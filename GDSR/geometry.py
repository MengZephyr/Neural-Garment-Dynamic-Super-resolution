import numpy as np
from sklearn import neighbors
import torch


def re_topo_verts(verts, to_map):
    numV = len(to_map)
    newVerts = torch.zeros(numV, verts.shape[1]).to(verts)
    for i in range(numV):
        cc = to_map[i]
        newVerts[i] = torch.sum(verts[cc], dim=0)
        newVerts[i] = newVerts[i] / float(len(cc))

    return newVerts


def sample_from_faces(vertarray, sampleInFaceVID, sampleInFaceABC):
    Vert0 = vertarray[sampleInFaceVID[:, 0], :].unsqueeze(1)  # Coarse_Verts_N * 1 * 3
    Vert1 = vertarray[sampleInFaceVID[:, 1], :].unsqueeze(1)  # Coarse_Verts_N * 1 * 3
    Vert2 = vertarray[sampleInFaceVID[:, 2], :].unsqueeze(1)  # Coarse_Verts_N * 1 * 3
    FaceVerts = torch.cat([Vert0, Vert1, Vert2], dim=1)  # Coarse_Verts_N * 3 * 3
    SampleVerts = torch.matmul(sampleInFaceABC.unsqueeze(1), FaceVerts).squeeze(1)  # Coarse_Verts_N * 3
    return SampleVerts


def get_shape_matrix(verts, faces):
    o1 = verts[faces[:, 1], :] - verts[faces[:, 0], :]
    o2 = verts[faces[:, 2], :] - verts[faces[:, 0], :]
    vom = torch.cat([o1.unsqueeze(-1), o2.unsqueeze(-1)], dim=-1)
    return vom


def deformation_gradient(verts, face, Dm_inv):
    Ds = get_shape_matrix(verts, face)
    F = torch.matmul(Ds, Dm_inv)
    return F


def decomposed_strain_elements(verts, face, Dm_inv):
    F = deformation_gradient(verts, face, Dm_inv)
    Fu = F[:, :, 0]
    Fv = F[:, :, 1]
    norm_Fu = torch.norm(Fu, dim=1)
    norm_Fv = torch.norm(Fv, dim=1)
    hat_Fu = torch.nn.functional.normalize(Fu, eps=1e-6, dim=1)
    hat_Fv = torch.nn.functional.normalize(Fv, eps=1e-6, dim=1)
    shearUV = torch.einsum('nd, nd -> n', hat_Fu, hat_Fv)
    return norm_Fu, norm_Fv, shearUV


def smooth_mesh_centercal(verts, vfaceIDs, vweights):
    face_centers = (verts[vfaceIDs[:, 0], :] + verts[vfaceIDs[:, 1], :] + verts[vfaceIDs[:, 2], :]) / 3.

    verts_csum = torch.zeros_like(verts).to(verts)
    verts_csum.index_add_(0, vfaceIDs[:, 0], face_centers)
    verts_csum.index_add_(0, vfaceIDs[:, 1], face_centers)
    verts_csum.index_add_(0, vfaceIDs[:, 2], face_centers)
    verts_csum = vweights * verts_csum
    cv_vec = verts - verts_csum
    return cv_vec


def compute_mesh_surface(verts, vfaceIDs):
    fv12 = verts[vfaceIDs[:, 2], :] - verts[vfaceIDs[:, 1], :]
    fv10 = verts[vfaceIDs[:, 0], :] - verts[vfaceIDs[:, 1], :]
    faces_normals = torch.cross(fv12, fv10, dim=1)

    verts_normals = torch.zeros_like(verts).to(verts)

    verts_normals.index_add_(0, vfaceIDs[:, 0], faces_normals)
    verts_normals.index_add_(0, vfaceIDs[:, 1], faces_normals)
    verts_normals.index_add_(0, vfaceIDs[:, 2], faces_normals)

    verts_normals = torch.nn.functional.normalize(verts_normals, eps=1e-6, dim=1)
    faces_normals = torch.nn.functional.normalize(faces_normals, eps=1e-6, dim=1)
    faces_tangents = torch.nn.functional.normalize(fv12, eps=1e-6, dim=1)
    faces_bases = torch.nn.functional.normalize(torch.cross(faces_tangents, faces_normals, dim=1), eps=1e-6, dim=1)
    return verts_normals, faces_bases, faces_tangents, faces_normals  # v_normals, f_x, f_y, f_z

# def compute_mesh_face_space(verts, vfaceIDs):
#     fv12 = verts[vfaceIDs[:, 2], :] - verts[vfaceIDs[:, 1], :]
#     fv10 = verts[vfaceIDs[:, 0], :] - verts[vfaceIDs[:, 1], :]
#     faces_normals = torch.cross(fv12, fv10, dim=1)
#     faces_normals = torch.nn.functional.normalize(faces_normals, eps=1e-6, dim=1)
#     faces_tangents = torch.nn.functional.normalize(fv12, eps=1e-6, dim=1)
#     faces_bases = torch.nn.functional.normalize(torch.cross(faces_tangents, faces_normals, dim=1), eps=1e-6, dim=1)
#     return faces_normals, faces_bases, faces_tangents


def compute_mesh_vert_normals(verts, vfaceIDs, if_face=False):
    fv12 = verts[vfaceIDs[:, 2], :] - verts[vfaceIDs[:, 1], :]
    fv10 = verts[vfaceIDs[:, 0], :] - verts[vfaceIDs[:, 1], :]
    faces_normals = torch.cross(fv12, fv10, dim=1)

    verts_normals = torch.zeros_like(verts).to(verts)

    verts_normals.index_add_(0, vfaceIDs[:, 0], faces_normals)
    verts_normals.index_add_(0, vfaceIDs[:, 1], faces_normals)
    verts_normals.index_add_(0, vfaceIDs[:, 2], faces_normals)

    verts_normals = torch.nn.functional.normalize(verts_normals, eps=1e-6, dim=1)
    if if_face:
        return verts_normals, torch.nn.functional.normalize(faces_normals, eps=1e-6, dim=1)
    else:
        return verts_normals


def position_sdf(verts, ref_verts, ref_normals, epsilon=0):
    vec = verts - ref_verts
    sdf = np.einsum('nd, nd -> n', vec, ref_normals)
    m = epsilon - sdf
    m_id = list(np.where(m < 0.)[0])
    m[m_id] = 0.
    p = np.expand_dims(m, axis=-1) * ref_normals
    return m, p


def get_edges(faceIDs):
    edges = np.concatenate([faceIDs[:, 0:2], faceIDs[:, 1:3], np.stack([faceIDs[:, 2], faceIDs[:, 0]], axis=1)], axis=0)
    edges = np.sort(edges, axis=-1)
    edges = np.unique(edges, axis=0)
    return edges


def create_laplacian_matrix(edges, numVerts):
    lapMatrix = torch.zeros((numVerts, numVerts), dtype=torch.float, requires_grad=False)
    lapMatrix[edges[:, 0], edges[:, 1]] = 1.
    lapMatrix[edges[:, 1], edges[:, 0]] = 1.
    sumL = torch.sum(lapMatrix, dim=1)
    sumL = torch.tensor(-1.) / sumL
    esumL = torch.diag(sumL)
    lapMatrix = torch.matmul(esumL, lapMatrix) + torch.eye(numVerts)

    return lapMatrix

def get_next_layer_edges(edgeID, numVs):
    ccMatrix = np.zeros((numVs, numVs), dtype=int)
    ccMatrix[edgeID[:, 0], edgeID[:, 1]] = 1

    next_layer_edges = []
    for i in range(numVs):
        ccids = list(np.where(ccMatrix[i, :] > 0)[0])
        ccids.append(i)
        i_next_edges = []
        for j in ccids:
            next_ccid = list(np.where(ccMatrix[j, :] > 0)[0])
            if len(next_ccid) > 0:
                next_ccid = next_ccid + ccids + ccids
                dup = [x for x in next_ccid if next_ccid.count(x) > 1]
                for x in dup:
                    next_ccid.remove(x)

                if len(next_ccid) > 0:
                    next_layerss = np.zeros((len(next_ccid), 2), dtype=int)
                    next_layerss[:, 0] = i
                    next_layerss[:, 1] = np.array(next_ccid)
                    i_next_edges.append(next_layerss)

        if len(i_next_edges) > 0:
            i_next_edges = np.concatenate(i_next_edges, axis=0)
            i_next_edges = np.unique(i_next_edges, axis=0)
            next_layer_edges.append(i_next_edges)

    next_layer_edges = np.concatenate(next_layer_edges, axis=0)
    next_layer_edges = np.sort(next_layer_edges, axis=-1)
    next_layer_edges = np.unique(next_layer_edges, axis=0)

    return next_layer_edges


def create_layer_mesh_edges(edges, numLayers, device):
    numV = np.unique(edges.flatten()).shape[0]

    layer_edges = []

    bi_edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    bi_edges = torch.from_numpy(bi_edges).type(torch.int).to(device)

    layer_edges.append(bi_edges)
    prelayer_edges = edges.copy()

    for i in range(numLayers-1):
        cedges = get_next_layer_edges(prelayer_edges, numV)
        prelayer_edges = cedges.copy()

        bi_edges = np.concatenate([cedges, cedges[:, ::-1]], axis=0)
        bi_edges = torch.from_numpy(bi_edges).type(torch.int).to(device)
        layer_edges.append(bi_edges)

    return layer_edges, numV


def dihedral_angle_adjacent_faces(normals, adjacency):
    normals0 = normals[adjacency[:, 0]]
    normals1 = normals[adjacency[:, 1]]
    cos = torch.einsum("ab,ab->a", normals0, normals1)
    sin = torch.norm(torch.cross(normals0, normals1, dim=-1), dim=-1)
    theta = torch.arctan2(sin, cos)
    return theta


def faces_to_edges_and_adjacency(faces, ifAll=True):
    edges = dict()
    for fidx, face in enumerate(faces):
        for i, v in enumerate(face):
            nv = face[(i + 1) % len(face)]
            edge = tuple(sorted([v, nv]))
            if not edge in edges:
                edges[edge] = []
            edges[edge] += [fidx]
    face_adjacency = []
    face_adjacency_edges = []
    for edge, face_list in edges.items():
        for i in range(len(face_list) - 1):
            for j in range(i + 1, len(face_list)):
                face_adjacency += [[face_list[i], face_list[j]]]
                face_adjacency_edges += [edge]
    edges = np.array([list(edge) for edge in edges.keys()], np.int32)
    face_adjacency = np.array(face_adjacency, np.int32)
    face_adjacency_edges = np.array(face_adjacency_edges, np.int32)
    if ifAll:
        return edges, face_adjacency, face_adjacency_edges
    else:
        return face_adjacency


def create_vertex_averay_weights(faces, numVerts):
    numAdjFaces = [0. for _ in range(numVerts)]
    numFaces = faces.shape[0]
    for f in range(numFaces):
        fvid = faces[f, :]
        numAdjFaces[fvid[0]] += 1.
        numAdjFaces[fvid[1]] += 1.
        numAdjFaces[fvid[2]] += 1.

    numAdjFaces = np.array(numAdjFaces)
    averageWeights = 1./numAdjFaces

    return averageWeights


def layered_collision_detection_with_kdTree(verts, split_id, distThred):
    verts_1 = verts[0:split_id, :]
    verts_2 = verts[split_id:, :]
    L1_tree = neighbors.KDTree(verts_1)
    dist, ccID = L1_tree.query(verts_2, k=1)
    numV_2 = verts_2.shape[0]
    self_collision_edges = []
    for i in range(numV_2):
        if dist[i] <= distThred:
            self_collision_edges.append([ccID[i][0], i+split_id])
    self_collision_edges = np.array(self_collision_edges)
    #print(self_collision_edges.shape)
    return self_collision_edges

