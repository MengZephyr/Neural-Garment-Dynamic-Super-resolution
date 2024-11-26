import open3d as o3d
import numpy as np
from sklearn import neighbors


def kdTree_nearest_neighbor(gverts, bverts):
    vtrees = neighbors.KDTree(bverts)
    _, vInd = vtrees.query(gverts, k=1)
    neiList = [i[0] for i in vInd]
    return neiList


class obj_collision_handling(object):
    def __init__(self, obj_fileName):
        objmesh = o3d.io.read_triangle_mesh(obj_fileName)
        objmesh.compute_vertex_normals()
        objmesh = o3d.t.geometry.TriangleMesh.from_legacy(objmesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        _ = self.scene.add_triangles(objmesh)

    def querry_nearest_points(self, qverts):
        query_point = o3d.core.Tensor(qverts, dtype=o3d.core.Dtype.Float32)
        ans = self.scene.compute_closest_points(query_point)
        # distance = self.scene.compute_signed_distance(query_point).numpy()
        # return ans['points'].numpy(), ans['primitive_normals'].numpy(), \
        #     ans['primitive_ids'].numpy(), ans['primitive_uvs'].numpy(), distance
        return ans['points'].numpy(), ans['primitive_normals'].numpy()


def creat_o3d_mesh(vertarray, facearray):
    objmesh = o3d.geometry.TriangleMesh()
    objmesh.vertices = o3d.utility.Vector3dVector(vertarray)
    objmesh.triangles = o3d.utility.Vector3iVector(facearray)
    objmesh.compute_vertex_normals()

    return objmesh


class mesh_collision_handling(object):
    def __init__(self, vertarray, facearray):
        objmesh = creat_o3d_mesh(vertarray, facearray)
        objmesh.compute_vertex_normals()

        objmesh = o3d.t.geometry.TriangleMesh.from_legacy(objmesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        _ = self.scene.add_triangles(objmesh)

    def querry_nearest_points(self, qverts):
        query_point = o3d.core.Tensor(qverts, dtype=o3d.core.Dtype.Float32)
        ans = self.scene.compute_closest_points(query_point)
        # distance = self.scene.compute_signed_distance(query_point).numpy()
        # return ans['points'].numpy(), ans['primitive_normals'].numpy(), \
        #     ans['primitive_ids'].numpy(), ans['primitive_uvs'].numpy(), distance
        return ans['points'].numpy(), ans['primitive_normals'].numpy()







