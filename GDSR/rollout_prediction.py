import torch.backends.cudnn

from architectures import CoarseGeo_GraphNet, FineDisplacement_Synthesis_WIRE, Coarse_faceVertex_Decoder
from mesh_topology import garment_mesh_topologies
from collision import obj_collision_handling, mesh_collision_handling

from geometry import *
from Data_IO import *

import time


USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device("cuda" if USE_CUDA else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

######pd30
num_mesh_layer = 2
graph_feat_dim = 128
num_graphs = 6
residualZ_dim = 512
folderZ_dim = 128
hyper_Layers = 5
hyper_inDim = 3
outDim = 3

if_self_collision = True


H_Net = CoarseGeo_GraphNet(node_dim=23,  # [n,v,a,m,p,hm,hp,d,hv]
                           edge_dim=9,  # edge vector & length & pre_edge vector & pre_length & rest_length
                           graph_v_dim=graph_feat_dim, graph_e_dim=graph_feat_dim,
                           num_graph=num_graphs, num_mesh_layers=num_mesh_layer,
                           coarse_out_dim=residualZ_dim+folderZ_dim, if_selfcc=if_self_collision).to(device)

A_Net = FineDisplacement_Synthesis_WIRE(coarse_feat_dim=residualZ_dim, face_mid_dim=256,
                                        hyper_net_in_dim=hyper_inDim,
                                        hyper_net_mid_dim=64, hyper_net_num_layers=hyper_Layers,
                                        hyper_net_out_dim=outDim, scale_ref=0.01).to(device)

C_Net = Coarse_faceVertex_Decoder(node_dim=folderZ_dim, num_fv=3, out_dim=outDim).to(device)

ckpName = '../All_d_300000.ckp'
ckp = torch.load(ckpName, map_location=lambda storage, loc: storage)
H_Net.load_state_dict(ckp['H_Net'])
A_Net.load_state_dict(ckp['A_Net'])
C_Net.load_state_dict(ckp['C_Net'])

H_Net.eval()
A_Net.eval()
C_Net.eval()

'''
---------------------------------------------------------------------------------------------------------
'''


def run_A_Net_patch_details(G_TMesh: garment_mesh_topologies, sample_vids, cpos, node_feats, coarse_faceH):
    sample_coarse_facesID = G_TMesh.target_in_coarse_faces[sample_vids]
    sample_coarse_fVIDs = G_TMesh.target_in_coarse_fVIDs[sample_vids, :]
    sample_coarse_inABC = G_TMesh.target_in_coarse_abc[sample_vids, :]

    face_coarse_feats = [node_feats[sample_coarse_fVIDs[:, 0], :],
                         node_feats[sample_coarse_fVIDs[:, 1], :],
                         node_feats[sample_coarse_fVIDs[:, 2], :]]

    sample_face_W = A_Net(face_coarse_feats)
    sample_displacements = A_Net.decode_details(sample_face_W, sample_coarse_inABC)

    sample_face_oriSpace = coarse_faceH[sample_coarse_facesID, :]

    sample_ori = sample_from_faces(cpos, sample_coarse_fVIDs, sample_coarse_inABC)
    result_details = torch.matmul(sample_displacements.unsqueeze(1), sample_face_oriSpace).squeeze(1) + sample_ori

    return result_details


def run_hyper_net(G_TMesh: garment_mesh_topologies, node_positions, node_feats, coarse_faceH):
    t = time.time()
    face_coarse_feats = [node_feats[G_TMesh.coarse_faces_tensor[:, 0], :],
                         node_feats[G_TMesh.coarse_faces_tensor[:, 1], :],
                         node_feats[G_TMesh.coarse_faces_tensor[:, 2], :]]

    sample_face_W = A_Net(face_coarse_feats)[G_TMesh.target_in_coarse_faces, :]

    sample_displacements = A_Net.decode_details(sample_face_W, G_TMesh.target_in_coarse_abc)
    t1 = time.time() - t
    sample_face_oriSpace = coarse_faceH[G_TMesh.target_in_coarse_faces, :]
    sample_ori = sample_from_faces(node_positions, G_TMesh.target_in_coarse_fVIDs, G_TMesh.target_in_coarse_abc)

    result_details = torch.matmul(sample_displacements.unsqueeze(1), sample_face_oriSpace).squeeze(1) + sample_ori
    return result_details, t1


def run_C_Net_correction_displacements(G_TMesh: garment_mesh_topologies, cpos, node_feats, coarse_faceH):
    face_coarse_feats = torch.cat([node_feats[G_TMesh.coarse_faces_tensor[:, 0], :],
                                   node_feats[G_TMesh.coarse_faces_tensor[:, 1], :],
                                   node_feats[G_TMesh.coarse_faces_tensor[:, 2], :]], dim=-1)
    fv_displacements = C_Net(face_coarse_feats)
    f_d0 = fv_displacements[:, 0:3]
    f_d1 = fv_displacements[:, 3:6]
    f_d2 = fv_displacements[:, 6:9]

    ori_v0 = cpos[G_TMesh.coarse_faces_tensor[:, 0], :]
    ori_v1 = cpos[G_TMesh.coarse_faces_tensor[:, 1], :]
    ori_v2 = cpos[G_TMesh.coarse_faces_tensor[:, 2], :]

    f_v0 = torch.matmul(f_d0.unsqueeze(1), coarse_faceH).squeeze(1) + ori_v0
    f_v1 = torch.matmul(f_d1.unsqueeze(1), coarse_faceH).squeeze(1) + ori_v1
    f_v2 = torch.matmul(f_d2.unsqueeze(1), coarse_faceH).squeeze(1) + ori_v2

    # f_v0 = f_d0 + ori_v0
    # f_v1 = f_d1 + ori_v1
    # f_v2 = f_d2 + ori_v2

    result_position = torch.zeros_like(cpos).to(device)
    result_position.index_add_(0, G_TMesh.coarse_faces_tensor[:, 0], f_v0)
    result_position.index_add_(0, G_TMesh.coarse_faces_tensor[:, 1], f_v1)
    result_position.index_add_(0, G_TMesh.coarse_faces_tensor[:, 2], f_v2)

    result_position = G_TMesh.coarse_average_weights * result_position

    return result_position


def niaeve_collision_resolve(gpnts, bpnts, bnorms, thr, if_mask=False):
    m, pvec = position_sdf(float_tensor_to_numpy(gpnts), bpnts, bnorms, epsilon=thr)
    if if_mask:
        geo_dist = torch.norm(gpnts-float_numpy_to_tensor(bpnts, device), dim=-1)
        geo_m = torch.where(geo_dist > 5 * thr, 0., 1.).unsqueeze(-1)
        cgpnts = gpnts + geo_m * float_numpy_to_tensor(pvec, device)
        return cgpnts, geo_m
    else:
        cgpnts = gpnts + float_numpy_to_tensor(pvec, device)
        return cgpnts


def filter_layeredCollsion_with_kdTree(CGPos, CGNormals, CGSplit_id, GPos, GSplit_id, Thr):
    verts_1 = CGPos[0:CGSplit_id, :]
    L1_tree = neighbors.KDTree(float_tensor_to_numpy(verts_1))

    G2 = GPos[GSplit_id:, :]
    dist, ccID = L1_tree.query(float_tensor_to_numpy(G2), k=1)
    nearestCG1_id = [i[0] for i in ccID]
    nearest_CG1_pos = CGPos[nearestCG1_id, :]
    nearest_CG1_normal = CGNormals[nearestCG1_id, :]
    sdf = torch.einsum('nd, nd -> n', GPos[GSplit_id:, :] - nearest_CG1_pos, nearest_CG1_normal)
    m = torch.relu(Thr-sdf)
    GPos[GSplit_id:, :] = G2 + m.unsqueeze(-1) * nearest_CG1_normal
    return GPos


'''
---------------------------------------------------------------------------------------------------------
'''

DataFolderRoot = './Data/'
coarse_garment_PD = str(30)
targe_garment_PD = str(10)

garment_type = 'short_skirt'
bodyName = 'body/'
if garment_type == 'multi-layered':
    '''multi-layered'''
    coarse_layerSplit_ID = 2209
    coarse_faceSplit_ID = 4338

    target_layerSplit_ID = 19404
    ifLayered = True
elif garment_type == 'layered_dress':
    '''layered_dress'''
    coarse_layerSplit_ID = 2229
    coarse_faceSplit_ID = 4362

    target_layerSplit_ID = 19749
    ifLayered = True
else:
    coarse_layerSplit_ID = -1
    target_layerSplit_ID = -1
    coarse_faceSplit_ID = -1
    ifLayered = False

file_prifix = DataFolderRoot + garment_type + '/Canonical/weld/'
GMeshes = garment_mesh_topologies(file_pref=file_prifix, garment_name=garment_type, device=device,
                                  coarse_PD=coarse_garment_PD, edgelayers=num_mesh_layer,
                                  target_PD=targe_garment_PD, ifcloth_layered=ifLayered)
target_in_coarse_NeiVIDs = GMeshes.NearestCoarseVIDs().cpu().numpy()

'''
coarse sample from target
'''
coarse_in_target_faces, coarse_in_target_abc = \
    load_Geo_Different_Resolution_Sampling(file_prifix, targe_garment_PD, coarse_garment_PD)
coarse_in_target_faces = int_numpy_to_tensor(coarse_in_target_faces, device)
coarse_in_target_abc = float_numpy_to_tensor(coarse_in_target_abc, device)
coarse_in_target_fVIDs = GMeshes.target_face_tensor[coarse_in_target_faces, :]


motion_name = '/0_House_Dancing/'
motion_sequence = '/silk_chamuse' + motion_name
Frame_0 = 20
Frame_1 = 20
Frame_ID = [i for i in range(Frame_0, Frame_1 + 1)]
saveRoot = '../rst/' + garment_type + motion_sequence + 'ro/'
if_Layer_CCFlat = True
if_collision_handling = False

if_roll_out = True  # set False when debugging in the case of 1-frame prediction.
if if_roll_out:
    coarse_pref = '/' + targe_garment_PD + '_' + coarse_garment_PD + '_R/'
else:
    coarse_pref = '/' + targe_garment_PD + '_' + coarse_garment_PD + '_B/'

with torch.no_grad():
    caseName = DataFolderRoot + garment_type + motion_sequence
    pre_CPos = readObj_vert_feats(
        caseName + '/PD' + coarse_garment_PD + '/PD' + coarse_garment_PD + '_' +
        str(max(Frame_0 - 1, 0)).zfill(7) + '.obj', device)

    before_pre_GPos = readObj_vert_feats(
        caseName + '/PD' + targe_garment_PD + '/PD' + targe_garment_PD + '_' +
        str(max(Frame_0 - 2, 0)).zfill(7) + '.obj', device)
    before_pre_CGPos = sample_from_faces(before_pre_GPos, coarse_in_target_fVIDs, coarse_in_target_abc)
    pre_GPos = readObj_vert_feats(
        caseName + '/PD' + targe_garment_PD + '/PD' + targe_garment_PD + '_' +
        str(max(Frame_0 - 1, 0)).zfill(7) + '.obj', device)
    pre_CGPos = sample_from_faces(pre_GPos, coarse_in_target_fVIDs, coarse_in_target_abc)
    pre_CGVelocity = pre_CGPos - before_pre_CGPos

    pre_displace = pre_CGPos - pre_CPos

    for fID in Frame_ID:
        cg_fname = DataFolderRoot + garment_type + motion_sequence + coarse_pref + str(fID).zfill(7) + '.npy'
        if if_roll_out:
            node_feats, edge_feats, selfcc_edgeIDs, selfcc_edge_feats, \
                CPos, CF_normals, CF_bases, CF_tangents = GMeshes.load_graph_feats_R(cg_fname)
        else:
            node_feats, edge_feats, selfcc_edgeIDs, selfcc_edge_feats, \
                CPos, gt_CGPos, CF_normals, CF_bases, CF_tangents = GMeshes.load_graph_feats_B(cg_fname)

        face_oriSpace = torch.cat([CF_bases.unsqueeze(1),
                                   CF_tangents.unsqueeze(1),
                                   CF_normals.unsqueeze(1)], dim=1)

        if if_roll_out:
            dynamic_feats = torch.cat([node_feats[:, 0:17], pre_displace, pre_CGVelocity], dim=-1)
        else:
            dynamic_feats = node_feats

        Z_out = H_Net(vert_feat=dynamic_feats, mesh_edge_feat=edge_feats, mesh_layer_edges=GMeshes.layer_edges,
                      cc_edge_feat=selfcc_edge_feats, selfcc_edges=selfcc_edgeIDs).squeeze(0)
        Z_R = Z_out[:, 0:residualZ_dim]
        Z_D = Z_out[:, residualZ_dim:residualZ_dim + folderZ_dim]

        _CGPos = run_C_Net_correction_displacements(GMeshes, CPos, Z_D, face_oriSpace)

        bodyFileName = DataFolderRoot + bodyName + motion_name + '/b' + str(fID + 1) + '.obj'
        body_cchandling = obj_collision_handling(bodyFileName)
        bcc_points, bcc_normals = body_cchandling.querry_nearest_points(float_tensor_to_numpy(_CGPos))

        if if_collision_handling:
            _CGPos = niaeve_collision_resolve(_CGPos, bcc_points, bcc_normals, thr=0.003)

        # save_obj('./test/_CGPos_p1.obj', _CGPos.detach().cpu().numpy(), GMeshes.coarse_faces_np)
        # exit(1)

        if GMeshes.ifcloth_layered:
            if if_collision_handling:
                L1_verts = _CGPos[0:coarse_layerSplit_ID, :]
                L2_verts = _CGPos[coarse_layerSplit_ID:, :]
                Layer_collision_cc = \
                    mesh_collision_handling(float_tensor_to_numpy(L1_verts),
                                            GMeshes.coarse_faces_np[0:coarse_faceSplit_ID, :])
                Lcc_pnts, Lcc_norms = \
                    Layer_collision_cc.querry_nearest_points(float_tensor_to_numpy(L2_verts))

                _CGPos[coarse_layerSplit_ID:, :], geo_m = \
                    niaeve_collision_resolve(L2_verts, Lcc_pnts, Lcc_norms, thr=0.005, if_mask=True)
            # save_obj('./test/_CGPos_p2.obj', _CGPos.detach().cpu().numpy(), GMeshes.coarse_faces_np)

            CGF_rst_Vnormals, CGF_rst_fnormals, CGF_rst_fbases, CGF_rst_ftangents = \
                compute_mesh_surface(_CGPos, GMeshes.coarse_faces_tensor)
            CGF_oriSpace = torch.cat([CGF_rst_fbases.unsqueeze(1),
                                      CGF_rst_ftangents.unsqueeze(1),
                                      CGF_rst_fnormals.unsqueeze(1)], dim=1)

            GPos, ct = run_hyper_net(GMeshes, _CGPos, Z_R, CGF_oriSpace)

            # L2_coarseVerts = _CGPos[coarse_layerSplit_ID:, :]
            # save_obj('./test/L2c_pass2.obj', L2_coarseVerts.detach().cpu().numpy())
            # L2_targetVerts = GPos[target_layerSplit_ID:, :]
            # save_obj('./test/L2_pass2.obj', L2_targetVerts.detach().cpu().numpy())
            if if_collision_handling:
                GPos = \
                    niaeve_collision_resolve(GPos, bcc_points[target_in_coarse_NeiVIDs, :],
                                             bcc_normals[target_in_coarse_NeiVIDs, :], thr=0.003)
            # save_obj('./test/G_pass3.obj', GPos.detach().cpu().numpy(), GMeshes.target_face_np)

            '''-----------------'''
            if if_Layer_CCFlat and if_collision_handling:
                L2_targetVerts = GPos[target_layerSplit_ID:, :]
                # print(L2_targetVerts.shape)
                # save_obj('./test/L2_pass3.obj', L2_targetVerts)
                # exit(1)

                L2_tINcNVIDs = target_in_coarse_NeiVIDs[target_layerSplit_ID:] - coarse_layerSplit_ID
                L2_tccpnts = Lcc_pnts[L2_tINcNVIDs, :]
                L2_tccnorms = Lcc_norms[L2_tINcNVIDs, :]
                L2_tm = geo_m[L2_tINcNVIDs, :]
                m, pvec = position_sdf(float_tensor_to_numpy(L2_targetVerts), L2_tccpnts, L2_tccnorms, epsilon=0.005)
                GPos[target_layerSplit_ID:, :] = L2_targetVerts + L2_tm * float_numpy_to_tensor(pvec, device)
                # save_obj('./test/L2_pass4.obj', L2_targetVerts.detach().cpu().numpy())
                # save_obj('./test/G_pass4.obj', GPos.detach().cpu().numpy(), GMeshes.target_face_np)
                # exit(1)

            '''-----------------'''

            CGPos = sample_from_faces(GPos, coarse_in_target_fVIDs, coarse_in_target_abc)

        else:
            CGF_rst_Vnormals, CGF_rst_fnormals, CGF_rst_fbases, CGF_rst_ftangents = \
                compute_mesh_surface(_CGPos, GMeshes.coarse_faces_tensor)
            CGF_oriSpace = torch.cat([CGF_rst_fbases.unsqueeze(1),
                                      CGF_rst_ftangents.unsqueeze(1),
                                      CGF_rst_fnormals.unsqueeze(1)], dim=1)

            GPos, ct = run_hyper_net(GMeshes, _CGPos, Z_R, CGF_oriSpace)

            if if_collision_handling:
                GPos = \
                    niaeve_collision_resolve(GPos, bcc_points[target_in_coarse_NeiVIDs, :],
                                             bcc_normals[target_in_coarse_NeiVIDs, :], thr=0.001)
            #
            # if GMeshes.ifcloth_layered:
            #     GPos = filter_layeredCollsion_with_kdTree(_CGPos, CGF_rst_Vnormals, coarse_layerSplit_ID,
            #                                               GPos, target_layerSplit_ID, Thr=0.003)

            CGPos = sample_from_faces(GPos, coarse_in_target_fVIDs, coarse_in_target_abc)

        save_obj(saveRoot + 'f/' + str(fID).zfill(7) + '.obj', GPos.detach().cpu().numpy(), GMeshes.target_face_np)
        save_obj(saveRoot + 'c/' + str(fID).zfill(7) + '.obj', CGPos.detach().cpu().numpy(), GMeshes.coarse_faces_np)

        pre_CGVelocity = CGPos - pre_CGPos
        pre_displace = CGPos - CPos
        pre_CGPos = CGPos.clone()

