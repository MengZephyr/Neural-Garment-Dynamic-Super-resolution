from models import *

Dyna_IF_NORM = False
Statistic_IF_NORM = True


'''
----------------------------Mesh-based-graph-convolutional-network------------------
'''


class CoarseGeo_Update(nn.Module):
    def __init__(self, vin_dim, ein_dim, num_mesh_layer, if_selfcc):
        super().__init__()
        self.mesh_edge_net = nn.ModuleList([Edge_Block(vin_dim, ein_dim, ein_dim) for _ in range(num_mesh_layer)])
        self.selfcc_edge_net = Edge_Block(vin_dim, ein_dim, ein_dim)

        if if_selfcc:
            self.vertex_block = Vertex_Block(vin_dim, ein_dim, num_mesh_layer + 1, vin_dim)
        else:
            self.vertex_block = Vertex_Block(vin_dim, ein_dim, num_mesh_layer, vin_dim)
        self.num_mesh_layers = num_mesh_layer
        self.node_dim = vin_dim
        self.edge_dim = ein_dim
        self.if_selfcc = if_selfcc

    def forward(self, mesh_node_feat, mesh_edge_feats, selfcc_edge_feats,
                mesh_layer_edges, selfcc_edges):

        b, num_node, dim = mesh_node_feat.shape

        sumedge_feats = []
        update_mesh_edge_feats = []
        for l in range(self.num_mesh_layers):
            edges = mesh_layer_edges[l]
            x0 = mesh_node_feat[:, edges[:, 0], :]
            x1 = mesh_node_feat[:, edges[:, 1], :]
            e01 = mesh_edge_feats[l]   # [b, num_edges, e_dim]

            ex = self.mesh_edge_net[l](x0, x1, e01, if_norm=Statistic_IF_NORM)
            sum_ex = torch.zeros(b, num_node, self.edge_dim).to(mesh_node_feat)
            sum_ex.index_add_(1, edges[:, 0], ex)

            sumedge_feats.append(sum_ex)
            update_mesh_edge_feats.append(ex)

        edge_feats = torch.cat(sumedge_feats, dim=-1)

        if self.if_selfcc:
            full_selfcc_edge_feats = torch.zeros(b, num_node, self.edge_dim).to(mesh_node_feat)
            update_selfcc_edge_feats = None
            if selfcc_edges is not None:
                if selfcc_edges.shape[0] > 0:
                    sx0 = mesh_node_feat[:, selfcc_edges[:, 0], :]
                    sx1 = mesh_node_feat[:, selfcc_edges[:, 1], :]
                    update_selfcc_edge_feats = self.selfcc_edge_net(sx0, sx1, selfcc_edge_feats, if_norm=Dyna_IF_NORM)
                    full_selfcc_edge_feats.index_add_(1, selfcc_edges[:, 0], update_selfcc_edge_feats)
            edge_feats = torch.cat([edge_feats, full_selfcc_edge_feats], dim=-1)
        else:
            update_selfcc_edge_feats = None

        update_mesh_node_feat = self.vertex_block(mesh_node_feat, edge_feats, if_norm=Statistic_IF_NORM)

        return update_mesh_node_feat, update_mesh_edge_feats, update_selfcc_edge_feats


class CoarseGeo_GraphNet(nn.Module):
    def __init__(self, node_dim=23,  # [n,v,a,m,p,hm,hp,d,hv]
                 edge_dim=9,  # edge vector & length & pre_edge vector & pre_length
                 graph_v_dim=128, graph_e_dim=128, num_graph=5, num_mesh_layers=3,
                 coarse_out_dim=640, if_selfcc=False):
        super().__init__()
        self.geo_node_encoder = NodeEncoder(in_dim=node_dim, mid_dim=8, out_dim=graph_v_dim)
        self.geo_edge_encoder = FeatEncoder(in_dim=edge_dim, out_dim=graph_e_dim)
        self.geo_ccedge_encoder = FeatEncoder(in_dim=edge_dim, out_dim=graph_e_dim)

        self.if_selfcc = if_selfcc

        self.geo_graph = nn.ModuleList([CoarseGeo_Update(vin_dim=graph_v_dim, ein_dim=graph_e_dim,
                                                         num_mesh_layer=num_mesh_layers, if_selfcc=if_selfcc)
                                        for _ in range(num_graph)])

        self.geo_node_decoder = FeatDecoder(in_dim=graph_v_dim, out_dim=coarse_out_dim)

        self.num_graph = num_graph
        self.num_mesh_layers = num_mesh_layers
        self.graph_edge_dim = graph_e_dim
        self.graph_node_dim = graph_v_dim

    def forward(self, vert_feat, mesh_edge_feat, mesh_layer_edges, cc_edge_feat, selfcc_edges):

        mesh_node_feat = self.geo_node_encoder(vert_feat.unsqueeze(0), if_norm=Statistic_IF_NORM)
        mesh_edge_feat = self.geo_edge_encoder(mesh_edge_feat.unsqueeze(0), if_norm=Statistic_IF_NORM)

        #print(selfcc_edges.shape)
        selfcc_edge_feat = None
        if self.if_selfcc and selfcc_edges is not None:
            if selfcc_edges.shape[0] > 0:
                selfcc_edge_feat = self.geo_ccedge_encoder(cc_edge_feat.unsqueeze(0), if_norm=Dyna_IF_NORM)

        # to initialize the edge features of each layers as ZERO but the first layer
        layer_mesh_edge_feats = []
        layer_mesh_edge_feats.append(mesh_edge_feat)
        for i in range(self.num_mesh_layers-1):
            num_e = mesh_layer_edges[i+1].shape[0]
            l_edge = torch.zeros(1, num_e, self.graph_edge_dim).to(vert_feat)
            layer_mesh_edge_feats.append(l_edge)

        for i in range(self.num_graph):
            mesh_node_feat, layer_mesh_edge_feats, selfcc_edge_feats = \
                self.geo_graph[i](mesh_node_feat, layer_mesh_edge_feats, selfcc_edge_feat,
                                  mesh_layer_edges, selfcc_edges)

        out = self.geo_node_decoder(mesh_node_feat, if_norm=Statistic_IF_NORM)
        return out


'''
----------------------------Detail-residual-hyper-network------------------
'''


class FineDisplacement_Synthesis_WIRE(nn.Module):
    def __init__(self, coarse_feat_dim=512, face_mid_dim=256,
                 hyper_net_in_dim=3,  # [a, b, c]
                 hyper_net_mid_dim=64, hyper_net_num_layers=4, hyper_net_out_dim=3, scale_ref=0.01):
        super().__init__()
        self.coarse_face_hypernet = HyperNet_Complex(in_dim=coarse_feat_dim*3,
                                                     mid_dim=face_mid_dim, hyper_in_dim=hyper_net_in_dim,
                                                     hyper_mid_dim=hyper_net_mid_dim, hyper_out_dim=hyper_net_out_dim,
                                                     num_layers=hyper_net_num_layers)
        #print(self.coarse_face_hypernet)

        self.hypernet_dims = self.coarse_face_hypernet.dims
        self.hypernet_dim_in = self.coarse_face_hypernet.dim_in
        self.hypernet_dim_out = self.coarse_face_hypernet.dim_out

        self.scale = nn.Parameter(scale_ref * torch.randn(3).unsqueeze(0), requires_grad=True)

        self.omega_0 = 5.
        self.scale_0 = 10.

    def forward(self, coarse_faceVertices):
        vf0, vf1, vf2 = coarse_faceVertices
        face_vfeats = torch.cat([vf0, vf1, vf2], dim=-1)
        face_W = self.coarse_face_hypernet(face_vfeats)

        return face_W

    def decode_details(self, face_W, detail_coord):
        dim_beg = 0
        dim_in = self.hypernet_dim_in[0]
        dim_out = self.hypernet_dim_out[0]
        dim_end = self.hypernet_dims[0]

        detail_x = Hyper_WIRE(feat=detail_coord,
                              weights=face_W[:, dim_beg:dim_end],
                              in_dim=dim_in, out_dim=dim_out, ifreal=True, omega_scale=(self.omega_0, self.scale_0))

        dim_beg = dim_end

        for i in range(len(self.hypernet_dims) - 1):
            dim_in = self.hypernet_dim_in[i + 1]
            dim_out = self.hypernet_dim_out[i + 1]
            dim_end = self.hypernet_dims[i + 1]
            omega_scale = None if dim_end == self.hypernet_dims[-1] else (self.omega_0, self.scale_0)
            detail_x = Hyper_WIRE(feat=detail_x,
                                  weights=face_W[:, dim_beg:dim_end],
                                  in_dim=dim_in, out_dim=dim_out, ifreal=False, omega_scale=omega_scale)
            dim_beg = dim_end

        detail_x = torch.tanh(detail_x.real) * self.scale
        return detail_x


'''
----------------------------Correction-displacement-decoder------------------
'''


class Coarse_faceVertex_Decoder(nn.Module):
    def __init__(self, node_dim=128, num_fv=3, out_dim=3):
        super().__init__()
        self.mlp_0 = MLP_Block(node_dim * num_fv, node_dim, groups=-1)
        self.mlp_1 = MLP_Block(node_dim, 256, groups=-1)
        self.mlp_2 = MLP_Block(256, 128, groups=-1)
        self.mlp_3 = MLP_Block(128, 64, groups=-1)
        self.out_linear = nn.Linear(64, out_dim * num_fv)
        self.scale = nn.Parameter(0.1 * torch.randn(3).unsqueeze(0), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(3).unsqueeze(0), requires_grad=True)
        self.num_fv = num_fv

    def forward(self, x):
        x = self.mlp_0(x, if_norm=False)
        x = self.mlp_1(x, if_norm=False)
        x = self.mlp_2(x, if_norm=False)
        x = self.mlp_3(x, if_norm=False)
        x = torch.tanh(self.out_linear(x)) * self.scale.repeat(1, self.num_fv) \
            + self.bias.repeat(1, self.num_fv)
        return x


# class Coarse_faceVertex_Decoder_wire(nn.Module):
#     def __init__(self, node_dim=128, num_fv=3, out_dim=3):
#         super().__init__()
#         self.D0 = WIRE_layer(node_dim*num_fv, node_dim)
#         self.D1 = WIRE_layer(node_dim, node_dim)
#         self.D2 = WIRE_layer(node_dim, node_dim//2)
#         self.D3 = WIRE_layer(node_dim//2, node_dim//4)
#         self.D4 = WIRE_layer(node_dim//4, out_dim * num_fv)
#
#         self.omega_0 = 5.
#         self.scale_0 = 10.
#
#         self.scale = nn.Parameter(0.1 * torch.randn(3).unsqueeze(0), requires_grad=True)
#         self.bias = nn.Parameter(torch.zeros(3).unsqueeze(0), requires_grad=True)
#         self.num_fv = num_fv
#
#     def forward(self, x):
#         x = self.D0(x, if_real=True, omega_scale=(self.omega_0, self.scale_0))
#         x = self.D1(x, if_real=False, omega_scale=(self.omega_0, self.scale_0))
#         x = self.D2(x, if_real=False, omega_scale=(self.omega_0, self.scale_0))
#         x = self.D3(x, if_real=False, omega_scale=(self.omega_0, self.scale_0))
#         x = self.D4(x, if_real=False, omega_scale=None)
#         x = torch.tanh(x.real) * self.scale.repeat(1, self.num_fv) \
#             + self.bias.repeat(1, self.num_fv)
#         return x



