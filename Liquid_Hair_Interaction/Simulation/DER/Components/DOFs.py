import taichi as ti
import taichi.math as tm

import Liquid_Hair_Interaction.Simulation.DER.Utils as Utils

@ti.data_oriented
class DOFs:
    def __init__(self, vertices, degrees):
        """
        vertices is a field of tm.vec3 to indicate position
        degrees is a field of float to indicate twist angle
        """
        self.vertices = vertices
        self.degrees = degrees
        self.num_vertices = self.vertices.shape[0]
        self.num_edges = self.num_vertices - 1

    @ti.func
    def t_getVertexSize(self) -> int:
        return self.vertices.shape[0]

    def getVertexSize(self) -> int:
        return self.vertices.shape[0]


@ti.data_oriented
class Edges:
    def __init__(self, dofs: DOFs):
        self.dofs = dofs
        self.start_index = 0
        self.end_index = self.dofs.num_edges
        self.size = self.dofs.num_edges
        self.data = ti.Vector.field(3, dtype=float, shape=self.size)
    
    @ti.kernel
    def compute(self):
        for idx in range(self.start_index, self.end_index):
            self.data[idx] = self.dofs.vertices[idx + 1] - self.dofs.vertices[idx]


@ti.data_oriented
class Lengths:
    def __init__(self, edges: Edges):
        self.edges = edges
        self.start_index = 0
        self.end_index = self.edges.size
        self.size = self.edges.size
        self.data = ti.field(dtype=float, shape=self.size)

    @ti.kernel
    def compute(self):
        for idx in range(self.start_index, self.end_index):
            self.data[idx] = tm.length(self.edges.data[idx])


@ti.data_oriented
class Tangents:
    def __init__(self, edges: Edges, lengths: Lengths):
        self.edges = edges
        self.lengths = lengths

        self.start_index = 0
        self.end_index = self.edges.size
        self.size = self.edges.size
        self.data = ti.Vector.field(3, dtype=float, shape=self.size)

    @ti.kernel
    def compute(self):
        for idx in range(self.start_index, self.end_index):
            self.data[idx] = self.edges.data[idx] / self.lengths.data[idx]


@ti.data_oriented
class CurvatureBinormals:
    def __init__(self, tangents: Tangents):
        self.tangents = tangents

        self.start_index = 1
        self.end_index = self.tangents.size
        self.size = self.tangents.size
        self.data = ti.Vector.field(3, dtype=float, shape=self.size)

    @ti.kernel
    def compute(self):
        for idx in range(self.start_index, self.end_index):
            t1 = self.tangents.data[idx - 1]
            t2 = self.tangents.data[idx]
            denominator = 1.0 + tm.dot(t1, t2)

            # this is not valid
            if denominator < 0 or Utils.t_isSmall(denominator):
                assert(False)

            self.data[idx] = 2.0 * tm.cross(t1, t2) / denominator


@ti.data_oriented
class TrigThetas:
    def __init__(self, dofs: DOFs):
        self.dofs = dofs
        self.start_index = 0
        self.end_index = self.dofs.num_edges
        self.size = self.dofs.num_edges
        self.sin_data = ti.field(dtype=float, shape=self.size)
        self.cos_data = ti.field(dtype=float, shape=self.size)

    @ti.kernel
    def compute(self):
        for idx in range(self.start_index, self.end_index):
            self.sin_data[idx] = tm.sin(self.dofs.degrees[idx])
            self.cos_data[idx] = tm.cos(self.dofs.degrees[idx])
