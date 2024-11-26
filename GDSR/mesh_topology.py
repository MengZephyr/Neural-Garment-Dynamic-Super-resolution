import numpy as np
import torch

from Data_IO import *
from geometry import *
from UVImg_Sampling import *
from random import shuffle


class garment_mesh_topologies(object):
    def __init__(self, file_pref, garment_name, device,
                 coarse_PD, edgelayers,
                 target_PD, ifcloth_layered):
        self.device = device
        self.file_pref = file_pref
        self.coarse_PD = coarse_PD
        self.target_PD = target_PD
        self.ifcloth_layered = ifcloth_layered

        ''' coarse'''
        self.coarse_faces_np = readObj_faces(file_pref + 'PD' + str(coarse_PD) + '_C.obj')
        self.coarse_faces_tensor = int_numpy_to_tensor(self.coarse_faces_np, device)
        # self.coarse_edges = get_edges(self.coarse_faces_np)
        # print(self.coarse_edges.shape[0])

        self.coarse_vm = readObj_vert_feats(file_pref + 'PD' + str(coarse_PD) + '_Flatten.obj', device)
        self.coarse_fm = readObj_faces(file_pref + 'PD' + str(coarse_PD) + '_Flatten.obj')
        c_dm = get_shape_matrix(self.coarse_vm[:, 0:2], self.coarse_fm)
        self.coarse_dm_inv = torch.linalg.inv(c_dm)

        self.coarse_edges, edgeLens = self.create_edge_topology()
        self.layer_edges, self.coarse_numVerts = create_layer_mesh_edges(self.coarse_edges, edgelayers, device)
        self.coarse_edgeLens = torch.cat([edgeLens, edgeLens], dim=0).unsqueeze(-1)

        self.coarse_average_weights = create_vertex_averay_weights(self.coarse_faces_np, self.coarse_numVerts)
        self.coarse_average_weights = float_numpy_to_tensor(self.coarse_average_weights, device).unsqueeze(-1)

        self.coarse_face_adjacent = faces_to_edges_and_adjacency(self.coarse_faces_np, ifAll=False)

        print(garment_name)
        print('coarse # of verts == ', self.coarse_numVerts)
        print('coarse # of edges == ', self.coarse_edges.shape[0])
        print('coarse # of faces == ', self.coarse_faces_tensor.shape[0])

        ''' target '''
        self.target_face_np = readObj_faces(file_pref + 'PD' + str(target_PD) + '_C.obj')
        self.target_face_tensor = int_numpy_to_tensor(self.target_face_np, device)
        self.target_in_coarse_faces, self.target_in_coarse_abc = \
            load_Geo_Different_Resolution_Sampling(file_pref, coarse_PD, target_PD)
        self.target_in_coarse_faces = int_numpy_to_tensor(self.target_in_coarse_faces, device)
        self.target_in_coarse_abc = float_numpy_to_tensor(self.target_in_coarse_abc, device)
        self.target_in_coarse_fVIDs = self.coarse_faces_tensor[self.target_in_coarse_faces, :]
        self.targe_numVerts = self.target_in_coarse_abc.shape[0]

        self.target_average_weights = create_vertex_averay_weights(self.target_face_np, self.targe_numVerts)
        self.target_average_weights = float_numpy_to_tensor(self.target_average_weights, device).unsqueeze(-1)

        self.target_vm = readObj_vert_feats(file_pref + 'PD' + str(target_PD) + '_Flatten.obj', device)
        self.target_fm = readObj_faces(file_pref + 'PD' + str(target_PD) + '_Flatten.obj')
        f_dm = get_shape_matrix(self.target_vm[:, 0:2], self.target_fm)
        self.target_dm_inv = torch.linalg.inv(f_dm)

        print("target # of verts == ", self.targe_numVerts)
        print('target # of faces == ', self.target_face_tensor.shape[0])

        self.target_PatchSampling = None
        self.patchIDs = None
        self.batch_PatchKK = 0
        self.batch_count = 0
        self.batch_size = 0

    def NearestCoarseVIDs(self):
        abc_index = torch.max(self.target_in_coarse_abc, dim=-1)[-1]
        VIDs = self.target_in_coarse_fVIDs[[i for i in range(self.target_in_coarse_fVIDs.shape[0])], abc_index]
        return VIDs

    def create_target_patch_sample(self, target_UVSize, target_PatchSize, if_overlap):
        self.target_PatchSampling = UVImg_PatchPool(self.file_pref + '/test/PD' + str(self.target_PD),
                                                    target_UVSize, target_PatchSize, 0.1, self.target_face_tensor,
                                                    self.device, if_prepare_patch=True, if_overlap=if_overlap)
        print("# of training patches = ", self.target_PatchSampling.ValidPatchCenters.shape)

        self.patchIDs = [i for i in range(self.target_PatchSampling.ValidPatchCenters.shape[0])]

    def set_patch_batch(self, pbSize):
        if len(self.patchIDs) == 0:
            print('please run create_target_patch_sample(...) before set_patch_batch(...).')
            return
        self.batch_size = pbSize
        self.batch_PatchKK = len(self.patchIDs) // self.batch_size
        self.batch_count = 0

    def get_batchSampleIDs(self):
        if self.batch_size == 0:
            print('please run set_patch_batch(...) before get_batchSampleIDs(...).')
            return []

        self.batch_count = self.batch_count % self.batch_PatchKK
        if self.batch_count == 0:
            shuffle(self.patchIDs)
        sample_PatchIDs = self.patchIDs[self.batch_count * self.batch_size: (self.batch_count+1) * self.batch_size]
        self.batch_count = self.batch_count + 1
        return sample_PatchIDs

    def create_mesh_edge_feats(self, verts, edges):
        edge_vec = verts[edges[:, 1], :]-verts[edges[:, 0], :]
        edge_len = torch.norm(edge_vec, dim=-1)
        return edge_vec, edge_len

    def create_edge_topology(self):
        m_edges = get_edges(self.coarse_fm)
        _, m_edgeLens = self.create_mesh_edge_feats(self.coarse_vm, m_edges)

        m_g_map = np.array(readVertMapFile(self.file_pref + 'PD' + str(self.coarse_PD) + '_u_to_g.txt'))
        e0 = m_g_map[m_edges[:, 0], :]
        e1 = m_g_map[m_edges[:, 1], :]
        g_edges = np.concatenate([e0, e1], axis=1)

        edges = np.sort(g_edges, axis=-1)
        edges, id = np.unique(edges, axis=0, return_index=True)
        rest_edgeLens = m_edgeLens[id]

        return edges, rest_edgeLens

    def load_graph_feats_R(self, fname):
        selfcc_edgeIDs = None
        selfcc_edge_feats = None
        with open(fname, 'rb') as f:
            '''for edge info.'''
            CPos = float_numpy_to_tensor(np.load(f), self.device)
            pre_CPos = float_numpy_to_tensor(np.load(f), self.device)

            '''for node info'''
            CV_normals = float_numpy_to_tensor(np.load(f), self.device)
            CVelocity = float_numpy_to_tensor(np.load(f), self.device)
            CAcceleration = float_numpy_to_tensor(np.load(f), self.device)
            C_m = float_numpy_to_tensor(np.load(f), self.device)
            C_p = float_numpy_to_tensor(np.load(f), self.device)
            preC_m = float_numpy_to_tensor(np.load(f), self.device)
            preC_p = float_numpy_to_tensor(np.load(f), self.device)
            # pre_displace = float_numpy_to_tensor(np.load(f), self.device)
            # pre_CGVelocity = float_numpy_to_tensor(np.load(f), self.device)

            if self.ifcloth_layered:
                selfcc_edgeIDs = np.load(f)

            '''ground truth'''
            # gt_CGPos = float_numpy_to_tensor(np.load(f), self.device)

            '''Input local'''
            CF_normals = float_numpy_to_tensor(np.load(f), self.device)
            CF_bases = float_numpy_to_tensor(np.load(f), self.device)
            CF_tangents = float_numpy_to_tensor(np.load(f), self.device)

            f.close()

        if self.ifcloth_layered:
            if selfcc_edgeIDs.shape[0] > 0:
                rev_ccedges = selfcc_edgeIDs[:, ::-1]
                selfcc_edgeIDs = np.concatenate([selfcc_edgeIDs, rev_ccedges], axis=0)
                selfcc_edgeIDs = int_numpy_to_tensor(selfcc_edgeIDs, device=self.device)
                cc_edge_vec = CPos[selfcc_edgeIDs[:, 1], :] - CPos[selfcc_edgeIDs[:, 0], :]
                cc_edge_len = torch.norm(cc_edge_vec, dim=-1).unsqueeze(-1)
                pre_cc_edge_vec = pre_CPos[selfcc_edgeIDs[:, 1], :] - pre_CPos[selfcc_edgeIDs[:, 0], :]
                pre_cc_edge_len = torch.norm(pre_cc_edge_vec, dim=-1).unsqueeze(-1)
                cc_len = 0.001 * torch.ones(pre_cc_edge_len.shape[0], 1).to(self.device)
                selfcc_edge_feats = \
                    torch.cat([cc_edge_vec, cc_edge_len, pre_cc_edge_vec, pre_cc_edge_len, cc_len], dim=-1)
            else:
                selfcc_edge_feats = None

        node_feats = [CV_normals, CVelocity, CAcceleration,
                      C_m.unsqueeze(-1), C_p, preC_m.unsqueeze(-1), preC_p]
        node_feats = torch.cat(node_feats, dim=-1)

        edge_vec, edge_len = self.create_mesh_edge_feats(CPos, self.layer_edges[0])
        preedge_vec, preedge_len = self.create_mesh_edge_feats(pre_CPos, self.layer_edges[0])
        edge_feats = [edge_vec, edge_len.unsqueeze(-1), preedge_vec, preedge_len.unsqueeze(-1), self.coarse_edgeLens]
        edge_feats = torch.cat(edge_feats, dim=-1)

        return node_feats, edge_feats, selfcc_edgeIDs, selfcc_edge_feats, \
            CPos, CF_normals, CF_bases, CF_tangents

    def load_graph_feats_B(self, fname, if_noise=False, noise_std=0.003):
        selfcc_edgeIDs = None
        selfcc_edge_feats = None
        with open(fname, 'rb') as f:
            '''for edge info.'''
            CPos = float_numpy_to_tensor(np.load(f), self.device)
            pre_CPos = float_numpy_to_tensor(np.load(f), self.device)

            '''for node info'''
            CV_normals = float_numpy_to_tensor(np.load(f), self.device)
            CVelocity = float_numpy_to_tensor(np.load(f), self.device)
            CAcceleration = float_numpy_to_tensor(np.load(f), self.device)
            C_m = float_numpy_to_tensor(np.load(f), self.device)
            C_p = float_numpy_to_tensor(np.load(f), self.device)
            preC_m = float_numpy_to_tensor(np.load(f), self.device)
            preC_p = float_numpy_to_tensor(np.load(f), self.device)
            pre_displace = float_numpy_to_tensor(np.load(f), self.device)
            pre_CGVelocity = float_numpy_to_tensor(np.load(f), self.device)

            if self.ifcloth_layered:
                selfcc_edgeIDs = np.load(f)

            '''ground truth'''
            gt_CGPos = float_numpy_to_tensor(np.load(f), self.device)

            '''Input local'''
            CF_normals = float_numpy_to_tensor(np.load(f), self.device)
            CF_bases = float_numpy_to_tensor(np.load(f), self.device)
            CF_tangents = float_numpy_to_tensor(np.load(f), self.device)

            f.close()

        if if_noise:
            noise = noise_std * torch.randn_like(pre_displace).to(pre_displace)
            pre_displace = pre_displace + noise
            pre_CGVelocity = pre_CGVelocity + noise

        if self.ifcloth_layered:
            if selfcc_edgeIDs.shape[0] > 0:
                rev_ccedges = selfcc_edgeIDs[:, ::-1]
                selfcc_edgeIDs = np.concatenate([selfcc_edgeIDs, rev_ccedges], axis=0)
                selfcc_edgeIDs = int_numpy_to_tensor(selfcc_edgeIDs, device=self.device)
                cc_edge_vec = CPos[selfcc_edgeIDs[:, 1], :] - CPos[selfcc_edgeIDs[:, 0], :]
                cc_edge_len = torch.norm(cc_edge_vec, dim=-1).unsqueeze(-1)
                pre_cc_edge_vec = pre_CPos[selfcc_edgeIDs[:, 1], :] - pre_CPos[selfcc_edgeIDs[:, 0], :]
                pre_cc_edge_len = torch.norm(pre_cc_edge_vec, dim=-1).unsqueeze(-1)
                cc_len = 0.001 * torch.ones(pre_cc_edge_len.shape[0], 1).to(self.device)
                selfcc_edge_feats = \
                    torch.cat([cc_edge_vec, cc_edge_len, pre_cc_edge_vec, pre_cc_edge_len, cc_len], dim=-1)
            else:
                selfcc_edge_feats = None

        node_feats = [CV_normals, CVelocity, CAcceleration,
                      C_m.unsqueeze(-1), C_p, preC_m.unsqueeze(-1), preC_p,
                      pre_displace, pre_CGVelocity]
        node_feats = torch.cat(node_feats, dim=-1)

        edge_vec, edge_len = self.create_mesh_edge_feats(CPos, self.layer_edges[0])
        preedge_vec, preedge_len = self.create_mesh_edge_feats(pre_CPos, self.layer_edges[0])
        edge_feats = [edge_vec, edge_len.unsqueeze(-1), preedge_vec, preedge_len.unsqueeze(-1), self.coarse_edgeLens]
        edge_feats = torch.cat(edge_feats, dim=-1)

        return node_feats, edge_feats, selfcc_edgeIDs, selfcc_edge_feats, \
            CPos, gt_CGPos, CF_normals, CF_bases, CF_tangents


if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device("cuda" if USE_CUDA else "cpu")

    DataFolderRoot = './Data'
    garment_pref = 'tshirt'
    file_prifix = DataFolderRoot + garment_pref + '/Canonical/weld/'
    t_mesh = garment_mesh_topologies(file_prifix, garment_pref, device, 30, 2, 10)

    UV_PatchSize = 80
    UVImgSize = 1024
    PatchOverlap = True
    t_mesh.create_target_patch_sample(UVImgSize, UV_PatchSize, PatchOverlap)

    import cv2
    from torchvision.utils import save_image

    refImg = cv2.imread(DataFolderRoot + garment_pref + '/Canonical/Weld/test/PD10_1024_uvmask.png', cv2.IMREAD_COLOR)
    for cc in t_mesh.target_PatchSampling.ValidPatchCenters:
        minX, minY, maxX, maxY = BBox_PatchCenter(cc, UV_PatchSize, 1024, 1024)
        cv2.rectangle(refImg, (minX, minY), (maxX, maxY), (0, 0, 255), 1)

    cv2.imwrite('./test/recSample.png', refImg)

    target_fname = DataFolderRoot + garment_pref + '/silk_chamuse/7_Swing_Dancing/PD10/PD10_' \
                   + str(10).zfill(7) + '.obj'
    detail_ground_truth = readObj_vert_feats(target_fname, device)

    for i in range(t_mesh.target_PatchSampling.num_valid_patches()):
        target_vids = t_mesh.target_PatchSampling.get_patch_UniqueVIDs(i)
        patch_localfvids = t_mesh.target_PatchSampling.get_patch_LocalfVIDs(i)

        patch_gt = detail_ground_truth[target_vids, :]
        gt_normals = compute_mesh_vert_normals(patch_gt, patch_localfvids)
        gt_nmap = t_mesh.target_PatchSampling.draw_patch_feats(i, 0.5 * gt_normals + 0.5)
        save_image(gt_nmap, './test/patches/' + str(i) + '.png')



