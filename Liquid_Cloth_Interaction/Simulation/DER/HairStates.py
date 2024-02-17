import taichi as ti
import taichi.math as tm

import Liquid_Cloth_Interaction.HairConstant
from Components import (HairVertex,
                        HairEdge,
                        HairReferenceTwist,
                        HairReferenceFrame,
                        HairCurvatureBinormal,
                        HairCurvature,
                        HairMaterialFrame)

@ti.dataclass
class HairInitState:
    num_strands: int
    num_vertices: int
    vertex_mass: float
    total_length: float
    youngs_modulus: int
    radius:float


@ti.data_oriented
class HairState:
    def __init__(self):
        self.vertices = HairVertex.field(shape=Cons)
