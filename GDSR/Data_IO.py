import os
import numpy as np
import torch


def float_numpy_to_tensor(x, device):
    return torch.from_numpy(x).type(torch.FloatTensor).to(device)


def float_tensor_to_numpy(x):
    return x.detach().cpu().numpy()


def int_numpy_to_tensor(x, device):
    return torch.from_numpy(x).type(torch.int).to(device)


def int_tensor_to_numpy(x):
    return x.detach().cpu().numpy()


def readObj_faces(fname, tagID=0):
    if not (os.path.exists(fname)):
        return None
    faceArray = []
    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'f':
            f = [int(x.split('/')[tagID]) - 1 for x in values[1:4]]
            faceArray.append(f)

    faceArray = np.array(faceArray)
    return faceArray


def readObj_vert_feats(fname, device=None, flag='v'):
    if not (os.path.exists(fname)):
        return None
    posArray = []

    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == flag:
            if flag == 'vt':
                v = [float(x) for x in values[1:3]]
                posArray.append([v[0], v[1]])
            else:
                v = [float(x) for x in values[1:4]]
                posArray.append([v[0], v[1], v[2]])

    posArray = np.array(posArray)

    if device is not None:
        posArray = float_numpy_to_tensor(posArray, device)

    return posArray


def readFaceSampleFile(fname):
    if not (os.path.exists(fname)):
        return None, None

    FID = []
    ABC = []
    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        fid = int(values[0])
        ab = [float(values[1]), float(values[2])]

        FID.append(fid)
        ABC.append([1 - ab[0] - ab[1], ab[0], ab[1]])

    FID = np.array(FID)
    ABC = np.array(ABC)

    return FID, ABC


def readVertMapFile(fname):
    if not (os.path.exists(fname)):
        return None

    vertID = []
    file = open(fname, "r")
    for line in file:
        values = line.split()
        cc = int(values[0])
        cvert = []
        for ci in range(cc):
            cvert.append(int(values[1 + ci]))
        vertID.append(cvert)

    return vertID


def readValidPixel(fname):
    if not (os.path.exists(fname)):
        return None

    PixelID = []
    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        pixel = [int(values[0]), int(values[1])]
        PixelID.append(pixel)
    PixelID = np.array(PixelID)

    return PixelID


def load_vertex_uv_feat(prefix, numVerts, device):
    verts_uvs = readObj_vert_feats(prefix + '_C.obj', device=device, flag='vt')
    ccid = readVertMapFile(prefix + '_g_to_u.txt')
    g_uvid = [ccid[i][0] for i in range(numVerts)]
    uv_feats = verts_uvs[g_uvid, :]
    return uv_feats


def load_Geo_Different_Resolution_Sampling(file_prefix, source_PD, target_PD):
    uv_in_coarse_faces, uv_in_coarse_abc = readFaceSampleFile(file_prefix + '/test/' + str(target_PD) + '_from_'
                                                              + str(source_PD) + '_Sampling.txt')
    geo_to_uv_vmap = readVertMapFile(file_prefix + '/PD' + str(target_PD) + '_g_to_u.txt')
    vmap = []
    for i in range(len(geo_to_uv_vmap)):
        vmap.append(geo_to_uv_vmap[i][0])
    geo_in_coarse_faces = uv_in_coarse_faces[vmap]
    geo_in_coarse_abc = uv_in_coarse_abc[vmap, :]
    return geo_in_coarse_faces, geo_in_coarse_abc


def save_obj(filename, vertices, faces=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        if faces is not None:
            for f in (faces + 1):  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

        fp.close()


def savePly(pDir, verts, colors, faces):
    numVerts = verts.shape[0]
    numFace = faces.shape[0]
    with open(pDir, 'w') as f:
        f.write("ply\n" + "format ascii 1.0\n")
        f.write("element vertex " + str(numVerts) + "\n")
        f.write("property float x\n" + "property float y\n" + "property float z\n")
        f.write("property uchar red\n" + "property uchar green\n"
                + "property uchar blue\n" + "property uchar alpha\n")
        f.write("element face " + str(numFace) + "\n")
        f.write("property list uchar int vertex_indices\n" + "end_header\n")
        for p in range(numVerts):
            v = verts[p]
            c = colors[p]
            f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + " "
                    + str(int(c[0])) + " " + str(int(c[1])) + " " + str(int(c[2])) + " " + "255\n")
        for p in range(numFace):
            fds = faces[p]
            f.write("3 " + str(fds[0]) + " " + str(fds[1])
                    + " " + str(fds[2]) + "\n")
        f.close()