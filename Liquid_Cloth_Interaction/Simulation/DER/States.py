import taichi as ti
import taichi.math as tm

from Liquid_Cloth_Interaction.Simulation.ElasticParameters import ElasticParameters
from Liquid_Cloth_Interaction.Simulation.DER.Components import DER_Vertices


@ti.data_oriented
class DER_State:
    def __init__(self, init_vertices: DER_Vertices, elastic: ElasticParameters):
        self.elastic = elastic
        self.vertices = init_vertices

    def get_num_vertices(self) -> int:
        return self.vertices.position.shape
