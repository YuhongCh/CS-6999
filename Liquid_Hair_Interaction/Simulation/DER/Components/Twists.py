import taichi as ti
import taichi.math as tm

import Liquid_Hair_Interaction.Simulation.DER.Utils as Utils
from Liquid_Hair_Interaction.Simulation.DER.Components.DOFs import *
from Liquid_Hair_Interaction.Simulation.DER.Components.ReferenceFrames import *


@ti.data_oriented
class Twists:
    def __init__(self, dofs: DOFs, ref_twists: ReferenceTwists):
        self.dofs = dofs
        self.ref_twists = ref_twists

        self.start_index = 1
        self.end_index = self.dofs.num_edges
        self.size = self.dofs.num_edges
        self.data = ti.field(dtype=float, shape=self.size)

    @ti.kernel
    def compute(self):
        for idx in range(self.start_index, self.end_index):
            self.data[idx] = self.data[idx] + self.dofs.degrees[idx-1] + self.dofs.degrees[idx]


@ti.data_oriented
class GradTwists:
    def __init__(self, lengths: Lengths, curvature_binormals: CurvatureBinormals):
        self.lengths = lengths
        self.curvature_binormals = curvature_binormals

        self.start_index = 1
        self.end_index = self.lengths.size
        self.size = self.lengths.size
        self.data = ti.Vector.field(11, dtype=float, shape=self.size)

    @ti.kernel
    def compute(self):
        for idx in range(self.start_index, self.end_index):
            curr_kb = self.curvature_binormals.data[idx]
            self.data[idx][0:3] = -0.5 / self.lengths.data[idx - 1] * curr_kb
            self.data[idx][8:11]= 0.5 / self.lengths.data[idx] * curr_kb
            self.data[idx][4:7] = -(self.data[idx][0:3] + self.data[idx][8:11])
            self.data[idx][3] = -1
            self.data[idx][7] = 1


@ti.data_oriented
class GradTwistsSquared:
    def __init__(self, grad_twists: GradTwists):
        self.grad_twists = grad_twists

        self.start_index = 1
        self.end_index = self.grad_twists.size
        self.size = self.grad_twists.size
        self.data = ti.Matrix.field(11, 11, dtype=float, shape=self.size)

    @ti.kernel
    def compute(self):
        for idx in range(self.start_index, self.end_index):
            self.data[idx] = self.grad_twists[idx] @ self.grad_twists[idx].transpose()
    

@ti.data_oriented
class HessTwists:
    def __init__(self, tangents: Tangents, lengths: Lengths, curvature_binormals: CurvatureBinormals):
        self.tangents = tangents
        self.lengths = lengths
        self.curvature_binormals = curvature_binormals

        self.start_index = 1
        self.end_index = self.lengths.size
        self.size = self.lengths.size
        self.data = ti.Matrix.field(11, 11, dtype=float, shape=self.size)

    @ti.kernel
    def compute(self):
        for idx in range(self.start_index, self.end_index):
            prev_tangent = self.tangents.data[idx - 1]
            curr_tangent = self.tangnents.data[idx]
            prev_length = self.lengths.data[idx - 1]
            curr_length = self.lengths.data[idx]
            kb = self.curvature_binormals.data[idx]
            
            chi = 1 + tm.dot(prev_tangent, curr_tangent)
            chi = max(chi, Utils.EPSILON)

            tilde_t = 1.0 / chi * (prev_tangent + curr_tangent)
            temp = prev_tangent + tilde_t
            D2mDe2 = -0.25 / (prev_length * prev_length) * (kb.outer_product(temp) + temp.outer_product(kb))
            temp = curr_tangent + tilde_t
            D2mDf2 = -0.25 / (curr_length * curr_length) * (kb.outer_product(temp) + temp.outer_product(kb))
            temp = curr_tangent + tilde_t
            D2mDeDf = 0.5 / (prev_length * curr_length) * (2.0 / chi * Utils.t_getCrossMatrix(prev_tangent) - kb.outer_product(tilde_t))
            D2mDfDe = D2mDeDf.transpose()

            self.data[idx][0:3, 0:3] = D2mDe2
            self.data[idx][0:3, 4:7] = -D2mDe2 + D2mDeDf
            self.data[idx][4:7, 0:3] = -D2mDe2 + D2mDfDe
            self.data[idx][4:7, 4:7] = D2mDe2 - (D2mDeDf + D2mDfDe) + D2mDf2
            self.data[idx][0:3, 8:11] = -D2mDeDf
            self.data[idx][8:11, 0:3] = -D2mDfDe
            self.data[idx][0:3, 0:3] = D2mDe2
            self.data[idx][8:11, 4:7] = D2mDfDe - D2mDf2
            self.data[idx][4:7, 8:11] = D2mDeDf - D2mDf2
            self.data[idx][8:11, 8:11] = D2mDf2
        