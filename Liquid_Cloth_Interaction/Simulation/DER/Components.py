import taichi as ti
import taichi.math as tm

from copy import copy

@ti.data_oriented
class DER_Vertices:
    def __init__(self, num_vertices: int):
        self.position = ti.ndarray(dtype=tm.vec3, shape=num_vertices)
        self.velocity = ti.ndarray(dtype=tm.vec3, shape=num_vertices)
        self.theta = ti.ndarray(dtype=float, shape=num_vertices)
        self.fixed = ti.ndarray(dtype=int, shape=num_vertices)


@ti.data_oriented
class DER_Edges:
    def __init__(self, vertices: DER_Vertices):
        num_edges = vertices.position.shape - 1
        self.dir = ti.ndarray(dtype=tm.vec3, shape=num_edges)
        self.length = ti.ndarray(dtype=float, shape=num_edges)
        self.vertices = copy(vertices)


@ti.data_oriented
class DER_CurvatureBinormal:
    def __init__(self, edges: DER_Edges):
        self.dir = ti.ndarray(dtype=tm.vec3, shape=edges.shape)
        self.edges = copy(edges)


# @ti.dataclass
# class HairReferenceFrame:
#     ref_frame1: tm.vec3
#     ref_frame2: tm.vec3
#
#
# @ti.dataclass
# class HairReferenceTwist:
#     theta: float
#
#
# @ti.dataclass
# class HairMaterialFrame:
#     mat_frame1: tm.vec3
#     mat_frame2: tm.vec3
#
#
# @ti.dataclass
# class HairCurvature:
#     kappa: tm.vec4
#
#
# @ti.dataclass
# class HairGradCurvature:
#     gradKappa: tm.mat4
