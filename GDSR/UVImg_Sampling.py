import torch
import numpy as np
from einops import rearrange

from models import NeuralFeatMap
from Data_IO import readFaceSampleFile, readValidPixel, float_numpy_to_tensor

import cv2


class UVImg_NormalRendering(object):
    def __init__(self, file_prefix, imgSize, meshFaces, numVerts, device):
        self.Render_OP = NeuralFeatMap(device).to(device)
        self.device = device

        pixelInFace, pixelInABC = \
            readFaceSampleFile(file_prefix + '_' + str(imgSize) + '_pixelGeoSample.txt')
        pixelFaceVIds = meshFaces[pixelInFace, :]
        validPixels = readValidPixel(file_prefix + '_' + str(imgSize) + '_validPixel.txt')
        numValidPixels = validPixels.shape[0]

        self.map_matrix = \
            self.gen_map_matrix(pixelFaceVIds, float_numpy_to_tensor(pixelInABC, device), numVerts, numValidPixels)
        self.ValidPixel_X = validPixels[:, 0]
        self.ValidPixel_Y = validPixels[:, 1]
        self.imgSize = imgSize

    def gen_map_matrix(self, pixelFaceVIDs, pixelInABC, numVerts, numValidPixels):
        H_matrix = torch.zeros(numVerts, numValidPixels).to(self.device)
        for i in range(numValidPixels):
            H_matrix[pixelFaceVIDs[i, :], i] = pixelInABC[i, :]
        return H_matrix

    def render_normal_map(self, normals):
        normalMap = self.Render_OP(self.imgSize, self.imgSize, self.ValidPixel_X, self.ValidPixel_Y,
                                   self.map_matrix, torch.transpose(normals, 0, 1))
        return normalMap


'''
-------------------------------------------------------------------------------------------------------------------
'''


def BBox_PatchCenter(center, patchsize, imgH, imgW):
    minX = center[0]-patchsize//2
    minY = center[1]-patchsize//2
    maxX = center[0]+patchsize//2
    maxY = center[1]+patchsize//2
    if minX < 0 or minY < 0:
        return -1, -1, -1, -1
    if maxX > imgW or maxY > imgH:
        return -1, -1, -1, -1
    return minX, minY, maxX, maxY


def calc_ValidArea(mask, minX, minY, maxX, maxY):
    p_mask = mask[minY:maxY, minX:maxX]
    area = np.sum(p_mask) / 255.
    return area


def check_pickcenter(pC, PSize, imgH, imgW, mask, areaRatioThr = 0.9):
    minX, minY, maxX, maxY = BBox_PatchCenter(pC, PSize, imgH, imgW)
    if minX >= 0 and minY >= 0:
        area = calc_ValidArea(mask, minX, minY, maxX, maxY)
        if area > PSize * PSize * areaRatioThr:
            return True
    return False


def genValidCenterArray(PSize, imgH, imgW, mask, areaRatioThr = 0.9):
    validCenter = []
    for y in range(imgH):
        for x in range(imgW):
            pCenter = [x, y]
            if check_pickcenter(pCenter, PSize, imgH, imgW, mask, areaRatioThr):
                validCenter.append(pCenter)
    validCenter = np.array(validCenter)
    return validCenter


def gen_no_overlap_patch_center_array(PSize, imgH, imgW, mask, beg_X, beg_Y, bbx, bby, areaThre=0.1):
    validCenter = []
    halph_PSize = PSize // 2

    end_X = beg_X + bbx
    end_Y = beg_Y + bby

    beg_X = beg_X - halph_PSize if beg_X - halph_PSize > 0 else 0
    beg_Y = beg_Y - halph_PSize if beg_Y - halph_PSize > 0 else 0

    bbx = end_X - beg_X
    bby = end_Y - beg_Y

    K_W = bbx // PSize + 1 if bbx % PSize > 0 else bbx // PSize
    K_H = bby // PSize + 1 if bby % PSize > 0 else bby // PSize

    area_bar = areaThre * PSize * PSize
    for ky in range(K_H):
        end_y = imgH if beg_Y + (ky + 1) * PSize > imgH else beg_Y + (ky + 1) * PSize
        cy = end_y - halph_PSize
        for kx in range(K_W):
            end_x = imgW if beg_X + (kx + 1) * PSize > imgW else beg_X + (kx + 1) * PSize
            cx = end_x - halph_PSize
            #validCenter.append([cx, cy])

            area = calc_ValidArea(mask, cx-halph_PSize, cy-halph_PSize, end_x, end_y)
            if area > area_bar:
                validCenter.append([cx, cy])

    validCenter = np.array(validCenter)
    return validCenter


def gen_half_overlap_patch_center_array(PSize, imgH, imgW, mask, beg_X, beg_Y, bbx, bby, areaThre=0.1):
    validCenter = []
    halph_PSize = PSize // 2

    end_X = beg_X + bbx
    end_Y = beg_Y + bby

    beg_X = beg_X - halph_PSize if beg_X - halph_PSize > 0 else 0
    beg_Y = beg_Y - halph_PSize if beg_Y - halph_PSize > 0 else 0

    bbx = end_X - beg_X
    bby = end_Y - beg_Y

    K_W = bbx // halph_PSize + 1 if bbx % halph_PSize > 0 else bbx // halph_PSize
    K_H = bby // halph_PSize + 1 if bby % halph_PSize > 0 else bby // halph_PSize

    area_bar = areaThre * PSize * PSize
    for ky in range(K_H):
        end_y = imgH if beg_Y + (ky + 2) * halph_PSize > imgH else beg_Y + (ky + 2) * halph_PSize
        cy = end_y - halph_PSize
        for kx in range(K_W):
            end_x = imgW if beg_X + (kx + 2) * halph_PSize > imgW else beg_X + (kx + 2) * halph_PSize
            cx = end_x - halph_PSize

            area = calc_ValidArea(mask, cx - halph_PSize, cy - halph_PSize, end_x, end_y)
            if area > area_bar:
                validCenter.append([cx, cy])

    # b0_x = 0 if beg_X-halph_PSize < 0 else beg_X-halph_PSize
    # b0_y = 0 if beg_Y-halph_PSize < 0 else beg_Y-halph_PSize
    # cx = b0_x + halph_PSize
    # cy = b0_y + halph_PSize
    # area = calc_ValidArea(mask, b0_x, b0_y, cx+halph_PSize, cy+halph_PSize)
    # if area > 0:
    #     validCenter.append([cx, cy])
    validCenter = np.array(validCenter)

    return validCenter


def load_training_target_UVImgSampling_info(file_prifix, imgSize, patchsize, patchAreaThr=0.1, if_overlap=False):
    pixelInFace, pixelInABC = readFaceSampleFile(file_prifix + '_' + str(imgSize) + '_pixelGeoSample.txt')

    validPixels = readValidPixel(file_prifix + '_' + str(imgSize) + '_validPixel.txt')
    numValidPixels = validPixels.shape[0]
    ValidPixel_X = validPixels[:, 0]
    ValidPixel_Y = validPixels[:, 1]

    ValidP_beg_X = np.min(ValidPixel_X)
    ValidP_end_X = np.max(ValidPixel_X) + 1
    ValidP_beg_Y = np.min(ValidPixel_Y)
    ValidP_end_Y = np.max(ValidPixel_Y) + 1

    #print(ValidP_beg_X, ValidP_beg_Y, ValidP_bound_W, ValidP_bound_H)
    #exit(1)

    ValidDictIds = [i for i in range(numValidPixels)]
    ValidDictMap = np.ones((imgSize, imgSize), dtype=int) * -1
    ValidDictMap[ValidPixel_Y, ValidPixel_X] = ValidDictIds

    ValidMask = np.where(ValidDictMap > -1, 255, 0)

    # cv2.imwrite('./test/geoMask.png', ValidMask)
    # refImg = cv2.imread('./test/geoMask.png', cv2.IMREAD_COLOR)
    # cv2.rectangle(refImg, (ValidP_beg_X, ValidP_beg_Y), (ValidP_end_X, ValidP_end_Y), (0, 0, 255), 1)
    # cv2.imwrite('./test/bbMask.png', refImg)

    if if_overlap:
        ValidPatchCenters = gen_half_overlap_patch_center_array(patchsize, imgSize, imgSize, ValidMask,
                                                                ValidP_beg_X, ValidP_beg_Y,
                                                                ValidP_end_X - ValidP_beg_X,
                                                                ValidP_end_Y - ValidP_beg_Y, patchAreaThr)
    else:
        ValidPatchCenters = gen_no_overlap_patch_center_array(patchsize, imgSize, imgSize, ValidMask,
                                                              ValidP_beg_X, ValidP_beg_Y,
                                                              ValidP_end_X - ValidP_beg_X,
                                                              ValidP_end_Y - ValidP_beg_Y, patchAreaThr)

    return ValidPixel_X, ValidPixel_Y, pixelInFace, pixelInABC, ValidDictMap, ValidPatchCenters


class UVImg_PatchPool(object):
    def __init__(self, file_prifix, imgSize, patchsize, patchAreaThr, GeoFaces, device,
                 if_prepare_patch=True, if_overlap=False):
        self.ValidPixel_X, self.ValidPixel_Y, \
            self.pixelInFace, self.pixelInABC, \
            self.ValidDictMap, self.ValidPatchCenters = \
            load_training_target_UVImgSampling_info(file_prifix, imgSize, patchsize, patchAreaThr,
                                                    if_overlap=if_overlap)

        self.pixelInABC = torch.from_numpy(self.pixelInABC).type(torch.FloatTensor).to(device)

        self.pixelInFace = torch.from_numpy(self.pixelInFace).type(torch.int).to(device)
        uniqueFace = torch.unique(self.pixelInFace)
        print("face shape in training:", uniqueFace.shape)

        self.GeoFaces = GeoFaces
        self.PATCHSIZE = patchsize
        self.ImgSize = imgSize
        self.device = device

        self.Render_OP = NeuralFeatMap(device).to(device)

        self.patchP_X_array = []
        self.patchP_Y_array = []
        self.patchP_UniqueVIDs_array = []
        self.patchP_Local_FVIDs_array = []
        self.patchP_HMatrix_array = []
        self.patchP_uniqueFIDs = []

        if if_prepare_patch:
            self.prepare_patches()

    def get_patch_info_from_array(self, id):
        return self.patchP_X_array[id], self.patchP_Y_array[id], self.patchP_UniqueVIDs_array[id], \
            self.patchP_Local_FVIDs_array[id], self.patchP_HMatrix_array[id]

    def get_patch_UniqueVIDs(self, id):
        return self.patchP_UniqueVIDs_array[id]

    def get_patch_LocalfVIDs(self, id):
        return self.patchP_Local_FVIDs_array[id]

    def get_patch_UniqueFIDs(self, id):
        return self.patchP_uniqueFIDs[id]

    def prepare_patches(self):
        nump = self.ValidPatchCenters.shape[0]
        # patchP_X_array = []
        # patchP_Y_array = []
        # patchP_UniqueVIDs_array = []
        # patchP_Local_FVIDs_array = []
        # patchP_HMatrix_array = []
        for p in range(nump):
            patchP_X, patchP_Y, patchInFVIDs, patchInABC, uniqueFIDs, uniqueFVIDs = self.gen_patch_sample_info(p)
            patchUniqueVIDs, patchLocal_FVIDs, patchLocal_rendering_FVIDs = \
                self.Patch_Local_FaceIDs(uniqueFVIDs, patchInFVIDs)

            num_pixels = patchP_X.shape[0]
            num_feats = patchUniqueVIDs.shape[0]
            patch_H = torch.zeros(num_feats, num_pixels).to(self.device)

            for i in range(num_pixels):
                patch_H[patchLocal_rendering_FVIDs[i, :], i] = patchInABC[i, :]

            self.patchP_X_array.append(patchP_X)
            self.patchP_Y_array.append(patchP_Y)

            self.patchP_UniqueVIDs_array.append(patchUniqueVIDs)
            self.patchP_Local_FVIDs_array.append(patchLocal_FVIDs)
            self.patchP_HMatrix_array.append(patch_H)
            self.patchP_uniqueFIDs.append(uniqueFIDs)

        #return patchP_X_array, patchP_Y_array, patchP_UniqueVIDs_array, patchP_Local_FVIDs_array, patchP_HMatrix_array

    def gen_patch_sample_info(self, center_id):
        patchCenter = self.ValidPatchCenters[center_id, :]
        minX, minY, maxX, maxY = BBox_PatchCenter(patchCenter, self.PATCHSIZE, self.ImgSize, self.ImgSize)
        indexPatch = self.ValidDictMap[minY:maxY, minX:maxX]
        #patchMask = np.where(indexPatch > -1, 1, 0)
        patchVPixels = np.argwhere(indexPatch > -1)
        patchP_Y = patchVPixels[:, 0]
        patchP_X = patchVPixels[:, 1]

        DictID = indexPatch[patchP_Y, patchP_X]

        patchInFaceIDs = self.pixelInFace[DictID]
        patchInFVIDs = self.GeoFaces[patchInFaceIDs, :]
        patchInABC = self.pixelInABC[DictID, :]

        uniqueFIDs = torch.unique(patchInFaceIDs)
        uniqueFVIDs = self.GeoFaces[uniqueFIDs, :]

        return patchP_X, patchP_Y, patchInFVIDs, patchInABC, uniqueFIDs, uniqueFVIDs

    def get_localIDs_of_verts_for_faces(self, vIDs, fVIDs, numfaces):
        ffvs = rearrange(fVIDs, 'f d -> (f d)')
        eqq = ffvs.unsqueeze(-1) == vIDs.unsqueeze(0)
        idx = eqq.nonzero()[:, -1]
        local_fvs = rearrange(idx, '(f d) -> f d', f=numfaces)
        return local_fvs

    def Patch_Local_FaceIDs(self, patch_unique_fVIDs, patch_rendering_fVIDs):
        patchUniqueVIDs = torch.unique(patch_unique_fVIDs.flatten())

        patchLocal_FVIDs = self.get_localIDs_of_verts_for_faces(patchUniqueVIDs, patch_unique_fVIDs,
                                                                patch_unique_fVIDs.shape[0])

        patchLocal_rendering_FVIDs = self.get_localIDs_of_verts_for_faces(patchUniqueVIDs, patch_rendering_fVIDs,
                                                                          patch_rendering_fVIDs.shape[0])

        return patchUniqueVIDs, patchLocal_FVIDs, patchLocal_rendering_FVIDs

    def draw_patch_feats(self, id, feats):
        img = self.Render_OP(self.PATCHSIZE, self.PATCHSIZE,
                             self.patchP_X_array[id], self.patchP_Y_array[id], self.patchP_HMatrix_array[id],
                             torch.transpose(feats, 0, 1))
        return img

    def num_valid_patches(self):
        return self.ValidPatchCenters.shape[0]


if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device("cuda" if USE_CUDA else "cpu")

    DataFolderRoot = './Data/'
    garment_pref = '/multi-layered'
    file_prifix = DataFolderRoot + garment_pref + '/Canonical/weld/test/PD10'

    UV_PatchSize = 80
    UVImgSize = 1024
    targe_garment_PD = 10
    target_pref = '/PD10/PD10_'

    from Data_IO import readObj_faces, readObj_vert_feats
    from geometry import compute_mesh_vert_normals
    from torchvision.utils import save_image

    target_faces_np = readObj_faces(
        DataFolderRoot + garment_pref + '/Canonical/Weld/PD' + str(targe_garment_PD) + '_C.obj')
    target_faces_tensor = torch.from_numpy(target_faces_np).type(torch.int).to(device)

    train_UVPatchSampling = UVImg_PatchPool(DataFolderRoot + garment_pref + '/Canonical/Weld/test/PD' +
                                            str(targe_garment_PD), UVImgSize, UV_PatchSize, 0.1,
                                            target_faces_tensor, device, if_prepare_patch=True, if_overlap=False)
    print("# of training patches = ", train_UVPatchSampling.num_valid_patches())

    refImg = cv2.imread(DataFolderRoot + garment_pref + '/Canonical/Weld/test/PD10_1024_uvmask.png', cv2.IMREAD_COLOR)
    for cc in train_UVPatchSampling.ValidPatchCenters:
        minX, minY, maxX, maxY = BBox_PatchCenter(cc, UV_PatchSize, 1024, 1024)
        cv2.rectangle(refImg, (minX, minY), (maxX, maxY), (0, 0, 255), 1)

    cv2.imwrite('./test/recSample.png', refImg)

    target_fname = DataFolderRoot + garment_pref + '/7_Swing_Dancing/' + target_pref + str(10).zfill(7) + '.obj'
    detail_ground_truth = readObj_vert_feats(target_fname, device)

    for i in range(train_UVPatchSampling.num_valid_patches()):
        target_vids = train_UVPatchSampling.get_patch_UniqueVIDs(i)
        patch_localfvids = train_UVPatchSampling.get_patch_LocalfVIDs(i)

        patch_gt = detail_ground_truth[target_vids, :]
        gt_normals = compute_mesh_vert_normals(patch_gt, patch_localfvids)
        gt_nmap = train_UVPatchSampling.draw_patch_feats(i, 0.5 * gt_normals + 0.5)
        save_image(gt_nmap, './test/patches/'+str(i)+'.png')


    # patchSize = 80
    # ValidPixel_X, ValidPixel_Y, pixelInFace, pixelInABC, ValidDictMap, ValidPatchCenters = \
    #     load_training_target_UVImgSampling_info(file_prifix, 1024, patchSize, 0.1, if_overlap=False)
    # print(len(ValidPatchCenters))
    #

