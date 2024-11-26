import torch
from torch import nn, einsum
from einops import rearrange


class MLP_Block(nn.Module):
    def __init__(self, dim_in, dim_out, groups=1):
        super().__init__()
        self.MLP_0 = nn.Linear(dim_in, dim_out)
        self.MLP_1 = nn.Linear(dim_out, dim_out)

        self.act = nn.LeakyReLU(0.02, inplace=True)

        '''
        dim_out should be divisible by groups
        when groups == dim_out, group_norm is equivalent with instance_norm
        when groups == 1, group_norm is equivalent with layer_norm  
        '''
        if groups > 0:
            self.norm = nn.GroupNorm(groups, dim_out, affine=False)
        else:
            self.norm = None

    def forward(self, x, if_norm=False, scale_shift=None):
        x = self.act(self.MLP_0(x))  # [b, num, dim]
        x = self.MLP_1(x)

        if if_norm and self.norm is not None:
            x = torch.transpose(x, -1, -2)  # [b, dim, num]
            x = self.norm(x)
            x = torch.transpose(x, -1, -2)
            if scale_shift is not None:
                scale, shift = scale_shift
                x = x * (1. + scale) + shift
        x = self.act(x)

        return x


class Vertex_Block(nn.Module):
    def __init__(self, vertex_dim, edge_dim, cedges, groups):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(vertex_dim), requires_grad=True)
        self.shift = nn.Parameter(torch.randn(vertex_dim), requires_grad=True)

        self.vertex_mlp = MLP_Block(vertex_dim + edge_dim*cedges, vertex_dim, groups)

    def forward(self, vx, exs, if_norm):
        vf = torch.cat([vx, exs], dim=-1)
        vf = self.vertex_mlp(vf, if_norm, scale_shift=[self.scale, self.shift])
        vx = vx + vf
        return vx


class Edge_Block(nn.Module):
    def __init__(self, vertex_dim, edge_dim, groups, if_edge_input=True):
        super().__init__()
        self.edge_MLP = MLP_Block(2 * vertex_dim + edge_dim if if_edge_input else 2 * vertex_dim, edge_dim, groups)
        self.if_edge_input = if_edge_input

    def forward(self, v0, v1, ex, if_norm):
        if self.if_edge_input and ex is not None:
            x = torch.cat([v0, v1, ex], dim=-1)
        else:
            x = torch.cat([v0, v1], dim=-1)
        x = self.edge_MLP(x, if_norm)
        ex = ex + x
        return ex


class FeatEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.mlp_0 = MLP_Block(in_dim, min(4 * out_dim, 1024), groups=min(4 * out_dim, 1024))
        self.mlp_1 = MLP_Block(min(4 * out_dim, 1024), min(2 * out_dim, 1024), groups=min(2 * out_dim, 1024))
        self.mlp_2 = MLP_Block(min(2 * out_dim, 1024), out_dim, groups=out_dim)

    def forward(self, x, if_norm):
        x = self.mlp_0(x, if_norm)
        x = self.mlp_1(x, if_norm)
        x = self.mlp_2(x, if_norm)
        return x

class NodeEncoder(nn.Module):
    def __init__(self, in_dim, mid_dim=8, out_dim=128):
        super().__init__()

        self.dim_down_0 = MLP_Block(in_dim, mid_dim * 4, groups=mid_dim * 4)
        self.dim_down_1 = MLP_Block(mid_dim * 4, mid_dim * 2, groups=mid_dim * 2)
        self.dim_down_2 = MLP_Block(mid_dim * 2, mid_dim, groups=mid_dim)

        self.mid_block = nn.Linear(mid_dim, mid_dim)
        #self.norm = nn.GroupNorm(1, mid_dim, affine=False)
        self.norm = nn.GroupNorm(mid_dim, mid_dim, affine=False)

        self.dim_up_0 = MLP_Block(mid_dim, out_dim//8, groups=out_dim//8)
        self.dim_up_1 = MLP_Block(out_dim//8, out_dim//4, groups=out_dim//4)
        self.dim_up_2 = MLP_Block(out_dim // 4, out_dim // 2, groups=out_dim // 2)
        self.dim_up_3 = MLP_Block(out_dim // 2, out_dim, groups=out_dim)

    def forward(self, x, if_norm):
        x = self.dim_down_0(x, if_norm)
        x = self.dim_down_1(x, if_norm)
        x = self.dim_down_2(x, if_norm)

        x = self.mid_block(x)
        x = torch.transpose(x, -1, -2)  # [b, dim, num]
        x = self.norm(x)
        x = torch.transpose(x, -1, -2)

        x = self.dim_up_0(x, if_norm)
        x = self.dim_up_1(x, if_norm)
        x = self.dim_up_2(x, if_norm)
        x = self.dim_up_3(x, if_norm)
        return x


class FeatDecoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp_0 = MLP_Block(in_dim, min(2 * in_dim, out_dim), groups=min(2 * in_dim, out_dim))
        self.mlp_1 = MLP_Block(min(2 * in_dim, out_dim), min(4 * in_dim, out_dim), groups=min(4 * in_dim, out_dim))
        self.mlp_2 = MLP_Block(min(4 * in_dim, out_dim), out_dim, groups=out_dim)

    def forward(self, vx, if_norm):
        vx = self.mlp_0(vx, if_norm)
        vx = self.mlp_1(vx, if_norm)
        vx = self.mlp_2(vx, if_norm)
        return vx


class HyperNet_Complex(nn.Module):
    def __init__(self, in_dim, mid_dim, hyper_in_dim, hyper_mid_dim, hyper_out_dim, num_layers):
        super().__init__()
        self.dim_in = []
        self.dim_out = []
        self.dims = []

        dim_end = hyper_in_dim * hyper_mid_dim + hyper_mid_dim
        self.dims.append(dim_end)
        self.dim_in.append(hyper_in_dim)
        self.dim_out.append(hyper_mid_dim)

        for i in range(num_layers-1):
            dim_end = dim_end + hyper_mid_dim * hyper_mid_dim + hyper_mid_dim
            self.dims.append(dim_end)
            self.dim_in.append(hyper_mid_dim)
            self.dim_out.append(hyper_mid_dim)

        dim_end = dim_end + hyper_mid_dim * hyper_out_dim + hyper_out_dim
        self.dims.append(dim_end)
        self.dim_in.append(hyper_mid_dim)
        self.dim_out.append(hyper_out_dim)

        self.mlp_0 = MLP_Block(in_dim, min(mid_dim*4, 1024), groups=-1)
        self.mlp_1 = MLP_Block(min(mid_dim*4, 1024), min(mid_dim*2, 1024), groups=-1)
        self.mlp_2 = MLP_Block(min(mid_dim*2, 1024), mid_dim, groups=-1)

        self.paraperter_proj_real = nn.Linear(mid_dim, self.dims[-1], bias=False)
        self.paraperter_proj_image = nn.Linear(mid_dim, self.dims[-1], bias=False)

    def forward(self, x):
        x = self.mlp_0(x, if_norm=False)
        x = self.mlp_1(x, if_norm=False)
        x = self.mlp_2(x, if_norm=False)

        WR = self.paraperter_proj_real(x)  # [numf, self.dims[-1]]
        WI = self.paraperter_proj_image(x)
        W = WR + 1j * WI
        return W


def Hyper_WIRE(feat, weights, in_dim, out_dim, ifreal, omega_scale=None):
    wh = weights[:, 0: in_dim * out_dim]  # [numV, in_dim*out_dim]
    wb = weights[:, in_dim * out_dim: in_dim * out_dim + out_dim]  # [numV, out_dim]
    wh = rearrange(wh, 'n (o i)->n o i', i=in_dim, o=out_dim)  # [numV, in_dim, out_dim]
    if ifreal:
        wh = wh.real
        wb = wb.real

    x = torch.matmul(wh, feat.unsqueeze(-1)).squeeze(-1) + wb  # [numV, out_dim]

    if omega_scale is not None:
        omega_0, scale_0 = omega_scale
        omega_x = omega_0 * x
        scale_x = scale_0 * x
        x = torch.exp(1j*omega_x - scale_x.abs().square())

    return x


class WIRE_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.wr = nn.Parameter(torch.randn(in_dim, out_dim), requires_grad=True)
        self.wc = nn.Parameter(torch.randn(in_dim, out_dim), requires_grad=True)
        self.br = nn.Parameter(torch.randn(1, out_dim), requires_grad=True)
        self.bc = nn.Parameter(torch.randn(1, out_dim), requires_grad=True)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x, if_real=False, omega_scale=None):
        if if_real:
            w = self.wr
            b = self.br
        else:
            w = self.wr + 1j*self.wc
            b = self.br + 1j*self.bc
        #w = rearrange(w, '(i o)-> i o', i=self.in_dim, o=self.out_dim)
        x = torch.matmul(x, w) + b

        if omega_scale is not None:
            omega_0, scale_0 = omega_scale
            omega_x = omega_0 * x
            scale_x = scale_0 * x
            x = torch.exp(1j * omega_x - scale_x.abs().square())

        return x


class NeuralFeatMap(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, H, W, pX, pY, vertMask, feature):
        out = torch.zeros(feature.size()[0], H, W).to(self.device)
        out[:, pY, pX] = torch.mm(feature, vertMask)  # [dimFeat, numVerts] * [numVerts, len(pX)] = [dimFeat, len(pX)]
        return out