from Data_IO import *
from geometry import *
from collision import obj_collision_handling

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device("cuda" if USE_CUDA else "cpu")

finePD = str(10)
coarsePD = str(30)
projRoot = './Data/'
garmentName = 'short_skirt/'
bodyName = 'body/'
materialName = 'silk_chamuse/'

IFLayered = False
layerSplit_vID = -1

FineFacesArray = readObj_faces(projRoot + garmentName + 'Canonical/weld/PD' + finePD + '_C.obj')
CoarseFaceArray = readObj_faces(projRoot + garmentName + 'Canonical/weld/PD' + coarsePD + '_C.obj')
tensor_CoarseFaceArray = torch.from_numpy(CoarseFaceArray).type(torch.int).to(device)

'''the coarse down sampled from the fine'''
FSampleFaceID, FSampleABC = readFaceSampleFile(projRoot + garmentName + 'Canonical/weld/test/' +
                                               coarsePD + '_from_' + finePD + '_Sampling.txt')
FSampleABC = torch.from_numpy(FSampleABC).type(torch.FloatTensor).to(device)  # Coarse_Verts_N * 3
print(FSampleABC.shape)
SampleInFVIDs = FineFacesArray[FSampleFaceID, :]
CoarseG2U_map = readVertMapFile(projRoot + garmentName + 'Canonical/weld/PD' + coarsePD + '_g_to_u.txt')

body_collision_Thred = 0.015
self_collision_Thred = 0.020



def downsample_from_fine(feats):
    sample_feats = sample_from_faces(feats, SampleInFVIDs, FSampleABC)
    rst_feats = re_topo_verts(sample_feats, CoarseG2U_map)
    return rst_feats


def layered_prepare_B(device, motionName, Frame0, Frame1, coarse_layerSplit=-1):
    caseName = projRoot + garmentName + materialName + motionName
    saveRoot = caseName + finePD + '_' + coarsePD + '_R/'

    FrameIDs = [i for i in range(Frame0, Frame1 + 1)]

    before_pre_CPos = readObj_vert_feats(
        caseName + '/PD' + coarsePD + '/PD' + coarsePD + '_' + str(max(Frame0 - 2, 0)).zfill(7) + '.obj', device)
    pre_CPos = readObj_vert_feats(
        caseName + '/PD' + coarsePD + '/PD' + coarsePD + '_' + str(max(Frame0 - 1, 0)).zfill(7) + '.obj', device)
    pre_CVelocity = pre_CPos - before_pre_CPos

    num_C = pre_CPos.shape[0]

    # before_pre_GPos = readObj_vert_feats(
    #     caseName + '/PD' + finePD + '/PD' + finePD + '_' + str(max(Frame0 - 2, 0)).zfill(7) + '.obj', device)
    # before_pre_CGPos = downsample_from_fine(before_pre_GPos)
    # pre_GPos = readObj_vert_feats(
    #     caseName + '/PD' + finePD + '/PD' + finePD + '_' + str(max(Frame0 - 1, 0)).zfill(7) + '.obj', device)
    # pre_CGPos = downsample_from_fine(pre_GPos)
    # pre_CGVelocity = pre_CGPos - before_pre_CGPos
    #
    # pre_displace = pre_CGPos - pre_CPos

    for fI in FrameIDs:
        CPos = readObj_vert_feats(
            caseName + '/PD' + coarsePD + '/PD' + coarsePD + '_' + str(fI).zfill(7) + '.obj', device)

        CV_normals, CF_bases, CF_tangents, CF_normals = compute_mesh_surface(CPos, tensor_CoarseFaceArray)
        CVelocity = CPos - pre_CPos
        CAcceleration = CVelocity - pre_CVelocity

        bodyFileName = projRoot + bodyName + motionName + '/b' + str(fI + 1) + '.obj'
        body_cchandling = obj_collision_handling(bodyFileName)
        bcc_points, bcc_normals = body_cchandling.querry_nearest_points(float_tensor_to_numpy(CPos))

        C_m, C_p = position_sdf(float_tensor_to_numpy(CPos), bcc_points, bcc_normals, body_collision_Thred)
        preC_m, preC_p = position_sdf(float_tensor_to_numpy(pre_CPos), bcc_points, bcc_normals, body_collision_Thred)

        if IFLayered and coarse_layerSplit > 0:
            selfcollision_edges = layered_collision_detection_with_kdTree(float_tensor_to_numpy(CPos),
                                                                          coarse_layerSplit, self_collision_Thred)
        else:
            selfcollision_edges = None

        with open(saveRoot + str(fI).zfill(7) + '.npy', 'wb') as f:
            '''for edge info.'''
            np.save(f, float_tensor_to_numpy(CPos))
            np.save(f, float_tensor_to_numpy(pre_CPos))

            '''for node info'''
            np.save(f, float_tensor_to_numpy(CV_normals))
            np.save(f, float_tensor_to_numpy(CVelocity))
            np.save(f, float_tensor_to_numpy(CAcceleration))
            np.save(f, C_m)
            np.save(f, C_p)
            np.save(f, preC_m)
            np.save(f, preC_p)
            # np.save(f, float_tensor_to_numpy(pre_displace))
            # np.save(f, float_tensor_to_numpy(pre_CGVelocity))

            if IFLayered and coarse_layerSplit > 0:
                '''layered_collision'''
                np.save(f, selfcollision_edges)

            '''Input local'''
            np.save(f, float_tensor_to_numpy(CF_normals))
            np.save(f, float_tensor_to_numpy(CF_bases))
            np.save(f, float_tensor_to_numpy(CF_tangents))

        pre_CVelocity = CVelocity.clone()
        # pre_CGVelocity = CGPos - pre_CGPos
        # pre_displace = CGPos - CPos

        pre_CPos = CPos.clone()
        #pre_CGPos = CGPos.clone()


'''
--------------------------------------------------------------------------------------------------------
'''

motionName = '/5_Samba_Dancing/'
Frame0 = 10
Frame1 = 310

layered_prepare_B(device, motionName, Frame0, Frame1, layerSplit_vID)
