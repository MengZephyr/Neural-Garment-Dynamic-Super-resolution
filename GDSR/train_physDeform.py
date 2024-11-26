import torch.backends.cudnn

from architectures import CoarseGeo_GraphNet, FineDisplacement_Synthesis_WIRE, Coarse_faceVertex_Decoder
from mesh_topology import garment_mesh_topologies
from collision import kdTree_nearest_neighbor
from vgg_model import VGGFeat_Loss

from geometry import *
from Data_IO import *

from random import shuffle
from einops import rearrange
from torchvision.utils import save_image


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

'''
---------------------------------------------------------------------------------------------------------
'''

ini_lr = 1.e-4
Optimizer = torch.optim.AdamW(params=list(H_Net.parameters()) + list(C_Net.parameters()) + list(A_Net.parameters()),
                              lr=ini_lr, amsgrad=True)


def set_models_train():
    H_Net.train()
    C_Net.train()
    A_Net.train()


def set_models_eval():
    H_Net.eval()
    C_Net.eval()
    A_Net.eval()


def save_ckp(fname, itt):
    torch.save({'itt': itt,
                'H_Net': H_Net.state_dict(),
                'C_Net': C_Net.state_dict(),
                'A_Net': A_Net.state_dict()}, fname)


def load_ckp(fname):
    ckp = torch.load(fname, map_location=lambda storage, loc: storage)
    H_Net.load_state_dict(ckp['H_Net'])
    C_Net.load_state_dict(ckp['C_Net'])
    A_Net.load_state_dict(ckp['A_Net'])


def set_optimizer_lr(lr):
    for i, param_group in enumerate(Optimizer.param_groups):
        param_group['lr'] = lr


'''
---------------------------------------------------------------------------------------------------------
'''

DataFolderRoot = './Data/'
coarse_garment_PD = 30
targe_garment_PD = 10
UV_PatchSize = 80
UVImgSize = 1024
patch_BatchSize = 8
PatchOverlap = False

'''
body
'''
body_face_np = readObj_faces(DataFolderRoot + '/body/Canonical/body_C.obj')
body_face_tensor = int_numpy_to_tensor(body_face_np, device)

garment_types = ['short_skirt', 'tshirt', 'single-dress', 'pants', 'multi-layered', 'layered_dress']
garment_ifLayred = [False, False, False, False, True, True]

GMesh_Dict = {}
for gname, iflayered in zip(garment_types, garment_ifLayred):
    file_prifix = DataFolderRoot + gname + '/Canonical/weld/'
    GMesh_Dict[gname] = garment_mesh_topologies(file_pref=file_prifix, garment_name=gname, device=device,
                                                coarse_PD=coarse_garment_PD, edgelayers=num_mesh_layer,
                                                target_PD=targe_garment_PD, ifcloth_layered=iflayered)
    GMesh_Dict[gname].create_target_patch_sample(UVImgSize, UV_PatchSize, PatchOverlap)
    GMesh_Dict[gname].set_patch_batch(patch_BatchSize)
    print('--------------------------------------------------')


train_motion_cases = [['short_skirt', '/silk_chamuse/', '0_House_Dancing/', i, 0] for i in range(1, 795)] + \
                     [['short_skirt', '/silk_chamuse/', '7_Swing_Dancing/', i, 7] for i in range(1, 715)] + \
                     [['tshirt', '/silk_chamuse/', '0_House_Dancing/', i, 0] for i in range(1, 795)] + \
                     [['tshirt', '/silk_chamuse/', '7_Swing_Dancing/', i, 7] for i in range(1, 715)] + \
                     [['pants', '/silk_chamuse/', '0_House_Dancing/', i, 0] for i in range(1, 795)] + \
                     [['pants', '/silk_chamuse/', '7_Swing_Dancing/', i, 7] for i in range(1, 715)] + \
                     [['multi-layered', '/silk_chamuse/', '0_House_Dancing/', i, 0] for i in range(1, 795)] + \
                     [['multi-layered', '/silk_chamuse/', '7_Swing_Dancing/', i, 7] for i in range(1, 715)] + \
                     [['layered_dress', '/silk_chamuse/', '0_House_Dancing/', i, 0] for i in range(1, 795)] + \
                     [['layered_dress', '/silk_chamuse/', '7_Swing_Dancing/', i, 7] for i in range(1, 715)]

garmID = 0
matID = 1
motionID = 2
frameID = 3

shuffle(train_motion_cases)
num_trainCases = len(train_motion_cases)

test_motion_cases = [['short_skirt', '/silk_chamuse/', '5_Samba_Dancing/', i, 5] for i in range(1, 610)] + \
                    [['single-dress', '/silk_chamuse/', '5_Samba_Dancing/', i, 5] for i in range(1, 610)] + \
                    [['layered_dress', '/silk_chamuse/', '5_Samba_Dancing/', i, 5] for i in range(1, 610)]

shuffle(test_motion_cases)
num_testCases = len(test_motion_cases)

coarse_pref = '/' + str(targe_garment_PD) + '_' + str(coarse_garment_PD) + '_B/'
target_pref = '/PD10/PD10_'

'''
---------------------------------------------------------------------------------------------------------
'''
L1Loss = torch.nn.L1Loss().to(device)
#L2Loss = torch.nn.MSELoss().to(device)
VGGSimilarityLoss = VGGFeat_Loss(device).to(device)


def strain_loss(verts, gt_verts, dm_inv, vfaces):
    u, v, suv = decomposed_strain_elements(verts, vfaces, dm_inv)
    gu, gv, g_suv = decomposed_strain_elements(gt_verts, vfaces, dm_inv)
    l_u = L1Loss(u, gu)
    l_v = L1Loss(v, gv)
    l_uv = L1Loss(suv, g_suv)
    return l_u+l_v+l_uv


def smooth_loss(verts, faceVIDs, fweights):
    cvec = smooth_mesh_centercal(verts, faceVIDs, fweights)
    loss_smooth = torch.bmm(cvec.unsqueeze(1), cvec.unsqueeze(-1))
    loss_smooth = torch.sum(loss_smooth) / float(verts.shape[0])
    return loss_smooth


def layered_cc(ledges, verts, norms, laycc_thr):
    if ledges.shape[0] > 0:
        edge_vec = verts[ledges[:, 1], :] - verts[ledges[:, 0]]
        sdf = torch.einsum('nd, nd -> n', edge_vec, norms[ledges[:, 0], :])
        cc = torch.relu(laycc_thr - sdf)
        loss_layercc = torch.sum(cc)
    else:
        loss_layercc = 0.
    return loss_layercc


def bend_loss(fnorms, gt_fnorms, f_adj):
    angle = dihedral_angle_adjacent_faces(fnorms, f_adj)
    gt_angle = dihedral_angle_adjacent_faces(gt_fnorms, f_adj)
    return L1Loss(angle, gt_angle)


norm_coeff = 0.01
strain_coeff = 0.01
bend_coeff = 0.003
c_smooth_coeff = 100.
f_smooth_coeff = 0.05
if_render_norm = True

'''
---------------------------------------------------------------------------------------------------------
'''


def draw_patch_images(fname, imgs, a=4):
    imgs = rearrange(imgs, '(a b) c h w -> c (a h) (b w)', a=a)
    save_image(imgs, fname)


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


def one_iteration_network_training(caseInfo, coeff=[1., 1.]):
    G_TMesh = GMesh_Dict[caseInfo[0]]

    c_fname = DataFolderRoot + \
              caseInfo[garmID] + caseInfo[matID] + caseInfo[motionID] + \
              coarse_pref + str(caseInfo[frameID]).zfill(7) + '.npy'
    node_feats, edge_feats, selfcc_edgeIDs, selfcc_edge_feats, \
        CPos, gt_CGPos, CF_normals, CF_bases, CF_tangents = \
        G_TMesh.load_graph_feats_B(c_fname, if_noise=True, noise_std=0.003)

    gt_CGNormal, gt_CGFNorms = compute_mesh_vert_normals(gt_CGPos, G_TMesh.coarse_faces_tensor, if_face=True)

    tg_fname = DataFolderRoot + \
               caseInfo[garmID] + caseInfo[matID] + caseInfo[motionID] + \
               target_pref + str(caseInfo[frameID]).zfill(7) + '.obj'
    detail_ground_truth = readObj_vert_feats(tg_fname, device)

    face_oriSpace = torch.cat([CF_bases.unsqueeze(1),
                               CF_tangents.unsqueeze(1),
                               CF_normals.unsqueeze(1)], dim=1)

    # body_fname = DataFolderRoot + '/body/' + case_info[motionID] + '/b' + str(case_info[frameID] + 1) + '.obj'
    # a_bverts = readObj_vert_feats(body_fname, device)
    # a_bnormals = compute_mesh_vert_normals(a_bverts, body_face_tensor)

    '''Initialize training'''
    set_models_train()
    Optimizer.zero_grad()

    '''mesh_based_graph_net'''
    Z_out = H_Net(vert_feat=node_feats, mesh_edge_feat=edge_feats, mesh_layer_edges=G_TMesh.layer_edges,
                  cc_edge_feat=selfcc_edge_feats, selfcc_edges=selfcc_edgeIDs).squeeze(0)
    Z_R = Z_out[:, 0:residualZ_dim]
    Z_D = Z_out[:, residualZ_dim:residualZ_dim + folderZ_dim]

    '''coarse_correctiopn_net'''
    coarse_results = run_C_Net_correction_displacements(G_TMesh, CPos, Z_D, face_oriSpace)

    CGF_rst_Vnormals, CGF_rst_fnormals, CGF_rst_fbases, CGF_rst_ftangents = \
        compute_mesh_surface(coarse_results, G_TMesh.coarse_faces_tensor)
    CGF_oriSpace = torch.cat([CGF_rst_fbases.unsqueeze(1),
                              CGF_rst_ftangents.unsqueeze(1),
                              CGF_rst_fnormals.unsqueeze(1)], dim=1)

    '''reconstruction constrain loss'''
    loss_c_geo = L1Loss(coarse_results, gt_CGPos)
    loss_c_normal = norm_coeff * L1Loss(CGF_rst_Vnormals, gt_CGNormal)

    '''deformation constrain loss'''
    loss_c_strain = strain_coeff * strain_loss(coarse_results, gt_CGPos,
                                               G_TMesh.coarse_dm_inv, G_TMesh.coarse_faces_tensor)
    loss_c_smooth = \
        c_smooth_coeff * smooth_loss(coarse_results, G_TMesh.coarse_faces_tensor, G_TMesh.coarse_average_weights)
    loss_c_bend = bend_coeff * bend_loss(CGF_rst_fnormals, gt_CGFNorms, G_TMesh.coarse_face_adjacent)

    '''layered collision loss'''
    # if selfcc_edgeIDs is not None:
    #     loss_c_layered = layered_cc(selfcc_edgeIDs, coarse_results, CGF_rst_Vnormals, laycc_thr=0.)
    #     print('cc: ', loss_c_layered.item())
    # else:
    #     loss_c_layered = 0

    Loss_coarse = loss_c_geo + loss_c_normal + loss_c_strain + loss_c_smooth + loss_c_bend

    #print('c_smooth: ', loss_c_smooth.item())

    '''hyper net'''
    SPatch_IDs = G_TMesh.get_batchSampleIDs()
    pgeo_rst_array = []
    pgeo_gt_array = []
    pnorm_rst_array = []
    pnorm_gt_array = []

    pimg_rst_array = []
    pimg_gt_array = []

    f_pstrain_loss = 0
    f_psmooth_loss = 0

    for pid in SPatch_IDs:
        target_vids = G_TMesh.target_PatchSampling.get_patch_UniqueVIDs(pid)
        patch_localfvids = G_TMesh.target_PatchSampling.get_patch_LocalfVIDs(pid)

        patch_gt = detail_ground_truth[target_vids, :]
        gt_normals = compute_mesh_vert_normals(patch_gt, patch_localfvids)
        gt_nmap = G_TMesh.target_PatchSampling.draw_patch_feats(pid, 0.5 * gt_normals + 0.5)

        detail_results = run_A_Net_patch_details(G_TMesh, target_vids, coarse_results, Z_R, CGF_oriSpace)
        results_normals = compute_mesh_vert_normals(detail_results, patch_localfvids)
        rst_nmap = G_TMesh.target_PatchSampling.draw_patch_feats(pid, 0.5 * results_normals + 0.5)

        target_fids = G_TMesh.target_PatchSampling.get_patch_UniqueFIDs(pid)
        patch_dm_inv = G_TMesh.target_dm_inv[target_fids, :, :]
        f_pstrain_loss = f_pstrain_loss + strain_loss(detail_results, patch_gt, patch_dm_inv, patch_localfvids)

        target_vweights = G_TMesh.target_average_weights[target_vids, :]
        f_psmooth_loss = f_psmooth_loss + smooth_loss(detail_results, patch_localfvids, target_vweights)

        pgeo_gt_array.append(patch_gt)
        pgeo_rst_array.append(detail_results)

        if if_render_norm:
            pimg_rst_array.append(rst_nmap.unsqueeze(0))
            pimg_gt_array.append(gt_nmap.unsqueeze(0))

        pnorm_gt_array.append(gt_normals)
        pnorm_rst_array.append(results_normals)

    pgeo_rst_array = torch.cat(pgeo_rst_array, dim=0)
    pgeo_gt_array = torch.cat(pgeo_gt_array, dim=0)
    pnorm_rst_array = torch.cat(pnorm_rst_array, dim=0)
    pnorm_gt_array = torch.cat(pnorm_gt_array, dim=0)

    if if_render_norm:
        pimg_rst_array = torch.cat(pimg_rst_array, dim=0)
        pimg_gt_array = torch.cat(pimg_gt_array, dim=0)

    '''losses'''
    loss_f_geo = L1Loss(pgeo_gt_array, pgeo_rst_array)
    loss_f_norm = norm_coeff * L1Loss(pnorm_gt_array, pnorm_rst_array)
    if if_render_norm:
        loss_f_normImg = VGGSimilarityLoss(pimg_rst_array, pimg_gt_array, patch_BatchSize) + \
                         L1Loss(pimg_rst_array, pimg_gt_array)
        Loss_fine = loss_f_geo + norm_coeff * loss_f_normImg
    else:
        Loss_fine = loss_f_geo + loss_f_norm

    Loss_fine = Loss_fine + \
                strain_coeff * f_pstrain_loss / patch_BatchSize + f_smooth_coeff * f_psmooth_loss / patch_BatchSize

    #print('f_smooth: ',  f_smooth_coeff * f_psmooth_loss.item() / patch_BatchSize)

    Loss = coeff[0] * Loss_coarse + coeff[1] * Loss_fine

    Loss.backward()
    Optimizer.step()

    return Loss, loss_c_geo, loss_c_normal, loss_f_geo, loss_f_norm


def one_iteration_network_testing(caseInfo):
    G_TMesh = GMesh_Dict[caseInfo[0]]

    c_fname = DataFolderRoot + \
              caseInfo[garmID] + caseInfo[matID] + caseInfo[motionID] + \
              coarse_pref + str(caseInfo[frameID]).zfill(7) + '.npy'
    node_feats, edge_feats, selfcc_edgeIDs, selfcc_edge_feats, \
        CPos, gt_CGPos, CF_normals, CF_bases, CF_tangents = \
        G_TMesh.load_graph_feats_B(c_fname, if_noise=False)

    tg_fname = DataFolderRoot + \
               caseInfo[garmID] + caseInfo[matID] + caseInfo[motionID] + \
               target_pref + str(caseInfo[frameID]).zfill(7) + '.obj'
    detail_ground_truth = readObj_vert_feats(tg_fname, device)

    face_oriSpace = torch.cat([CF_bases.unsqueeze(1),
                               CF_tangents.unsqueeze(1),
                               CF_normals.unsqueeze(1)], dim=1)

    set_models_eval()

    with torch.no_grad():
        '''mesh_based_graph_net'''
        Z_out = H_Net(vert_feat=node_feats, mesh_edge_feat=edge_feats, mesh_layer_edges=G_TMesh.layer_edges,
                      cc_edge_feat=selfcc_edge_feats, selfcc_edges=selfcc_edgeIDs).squeeze(0)
        Z_R = Z_out[:, 0:residualZ_dim]
        Z_D = Z_out[:, residualZ_dim:residualZ_dim + folderZ_dim]

        '''coarse_correctiopn_net'''
        coarse_results = run_C_Net_correction_displacements(G_TMesh, CPos, Z_D, face_oriSpace)
        CGF_rst_Vnormals, CGF_rst_fnormals, CGF_rst_fbases, CGF_rst_ftangents = \
            compute_mesh_surface(coarse_results, G_TMesh.coarse_faces_tensor)
        CGF_oriSpace = torch.cat([CGF_rst_fbases.unsqueeze(1),
                                  CGF_rst_ftangents.unsqueeze(1),
                                  CGF_rst_fnormals.unsqueeze(1)], dim=1)

        loss_c_geo = L1Loss(coarse_results, gt_CGPos)

        '''hyper net'''
        SPatch_IDs = G_TMesh.get_batchSampleIDs()
        pgeo_rst_array = []
        pgeo_gt_array = []
        pimg_rst_array = []
        pimg_gt_array = []
        for pid in SPatch_IDs:
            target_vids = G_TMesh.target_PatchSampling.get_patch_UniqueVIDs(pid)
            patch_localfvids = G_TMesh.target_PatchSampling.get_patch_LocalfVIDs(pid)

            patch_gt = detail_ground_truth[target_vids, :]
            gt_normals = compute_mesh_vert_normals(patch_gt, patch_localfvids)
            gt_nmap = G_TMesh.target_PatchSampling.draw_patch_feats(pid, 0.5 * gt_normals + 0.5)

            detail_results = run_A_Net_patch_details(G_TMesh, target_vids, coarse_results, Z_R, CGF_oriSpace)
            results_normals = compute_mesh_vert_normals(detail_results, patch_localfvids)
            rst_nmap = G_TMesh.target_PatchSampling.draw_patch_feats(pid, 0.5 * results_normals + 0.5)

            pgeo_gt_array.append(patch_gt)
            pgeo_rst_array.append(detail_results)

            # pnorm_gt_array.append(gt_normals)
            # pnorm_rst_array.append(results_normals)

            pimg_rst_array.append(rst_nmap.unsqueeze(0))
            pimg_gt_array.append(gt_nmap.unsqueeze(0))

        pgeo_rst_array = torch.cat(pgeo_rst_array, dim=0)
        pgeo_gt_array = torch.cat(pgeo_gt_array, dim=0)
        # pnorm_rst_array = torch.cat(pnorm_rst_array, dim=0)
        # pnorm_gt_array = torch.cat(pnorm_gt_array, dim=0)
        pimg_rst_array = torch.cat(pimg_rst_array, dim=0)
        pimg_gt_array = torch.cat(pimg_gt_array, dim=0)

        '''losses'''
        loss_f_geo = L1Loss(pgeo_gt_array, pgeo_rst_array)

        return loss_c_geo, loss_f_geo, coarse_results, G_TMesh.coarse_faces_np, pimg_rst_array, pimg_gt_array


'''
---------------------------------------------------------------------------------------------------------
'''

ckpName = None
if ckpName is not None:
    load_ckp(ckpName)

total_iterations = 500000
print(total_iterations)

if_adj_lr = True
if_lr_adj_start_slow = False
lr_scale = 0.5
lr_adjust_freq = 50000
print('lr_adj_freq:', lr_adjust_freq)
saveKK = 10000

if_adj_coeff = False
adj_coeff_freq = 50000
train_coeff = [1., 1.]

IFSumWriter = True
if IFSumWriter:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()


bit = -1
for itt in range(bit+1, total_iterations+1):
    '''
        Adjust learning rate
    '''
    if if_adj_lr:
        if itt % lr_adjust_freq == 0:
            if if_lr_adj_start_slow:
                lr = max(ini_lr * (lr_scale ** max(itt // lr_adjust_freq - 1, 0)), 1.e-6)
            else:
                lr = max(ini_lr * (lr_scale ** itt // lr_adjust_freq), 1.e-6)
            set_optimizer_lr(lr)

    '''
        Adjust coeff
    '''
    if if_adj_coeff:
        if itt % adj_coeff_freq == 0:
            train_coeff[0] = max(train_coeff[0] - 0.2 * (itt // adj_coeff_freq), 1.)
            train_coeff[1] = min(train_coeff[1] + 0.2 * (itt // adj_coeff_freq), 1.)
            print('>> update coeff: ', train_coeff)

    '''
        Training
    '''
    fk = itt % num_trainCases
    if fk == 0:
        shuffle(train_motion_cases)
    case_info = train_motion_cases[fk]
    Loss, loss_c_geo, loss_c_normal, loss_f_geo, loss_f_norm = \
        one_iteration_network_training(case_info, coeff=train_coeff)
    print('Iter_{} --> Loss: {:.4f}, c_geo_loss: {:.4f}, f_geo_loss: {:.4f}, f_norm_loss: {:.4f}'.
          format(itt, Loss.item(), loss_c_geo.item(), loss_f_geo.item(), loss_f_norm.item()))

    '''
        save check point
    '''
    if itt % 1000 == 0 or itt == total_iterations:
        if itt % 1000 == 0 or itt == total_iterations:
            ckp_fname = '../ckp/all/All_d_temp.ckp'
            if itt % saveKK == 0 or itt == total_iterations:
                ckp_fname = '../ckp/all/All_d_' + str(itt) + '.ckp'
            save_ckp(ckp_fname, itt)

        if IFSumWriter:
            writer.add_scalar('Train_geoDist', loss_c_geo, itt)
            writer.add_scalar('Train_normDist', loss_f_norm, itt)
            writer.add_scalar('Train_optLoss', Loss, itt)

    '''
        Testing
    '''
    if itt % saveKK == 0 or itt == total_iterations:
        loss_c_geo, loss_f_geo, coarse_results, coarse_faces, pimg_rst_array, pimg_gt_array = \
            one_iteration_network_testing(case_info)
        save_obj('../test/' + str(itt) + '_' + str(case_info[frameID]) + '_' + str(case_info[-1]) + '_t.obj',
                 coarse_results.detach().cpu().numpy(), coarse_faces)
        draw_patch_images('../test/gt_p_' + str(itt) + '_t.jpg', pimg_gt_array)
        draw_patch_images('../test/p_' + str(itt) + '_t.jpg', pimg_rst_array)

        ek = itt % num_testCases
        if ek == 0:
            shuffle(test_motion_cases)
        t_case_info = test_motion_cases[ek]
        loss_c_geo, loss_f_geo, coarse_results, coarse_faces, pimg_rst_array, pimg_gt_array = \
            one_iteration_network_testing(t_case_info)
        print('Eval_Iter_{} --> c_geo_loss: {:.4f}, f_geo_loss: {:.4f}'.
              format(itt, loss_c_geo.item(), loss_f_geo.item()))

        save_obj('../test/' + str(itt) + '_' + str(t_case_info[frameID]) + '_' + str(t_case_info[-1]) + '_e.obj',
                 coarse_results.detach().cpu().numpy(), coarse_faces)
        draw_patch_images('../test/gt_p_' + str(itt) + '_e.jpg', pimg_gt_array)
        draw_patch_images('../test/p_' + str(itt) + '_e.jpg', pimg_rst_array)

        if IFSumWriter:
            writer.add_scalar('Eval_geoDist', loss_c_geo, itt)






