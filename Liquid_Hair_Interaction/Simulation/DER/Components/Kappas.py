import taichi as ti
import taichi.math as tm

import Liquid_Hair_Interaction.Simulation.DER.Utils as Utils

from Liquid_Hair_Interaction.Simulation.DER.Components.DOFs import *
from Liquid_Hair_Interaction.Simulation.DER.Components.MaterialFrames import *
from Liquid_Hair_Interaction.Simulation.DER.Components.ReferenceFrames import *

@ti.data_oriented
class Kappas:
    def __init__(self, curvature_binormals: CurvatureBinormals, 
                 mat_frame1: MaterialFrames1, mat_frame2: MaterialFrames2):
        self.curvature_binormals = curvature_binormals
        self.mat_frame1 = mat_frame1
        self.mat_frame2 = mat_frame2

        self.start_index = 1
        self.end_index = self.curvature_binormals.size
        self.size = self.curvature_binormals.size
        self.data = ti.Vector.field(2, dtype=float, shape=self.size)

    @ti.kernel
    def compute(self):
       for idx in range(self.start_index, self.end_index):
           kb = self.curvature_binormals.data[idx]
           prev_mf1 = self.mat_frame1.data[idx - 1]
           curr_mf1 = self.mat_frame1.data[idx]
           prev_mf2 = self.mat_frame2.data[idx - 1]
           curr_mf2 = self.mat_frame2.data[idx]

           d1 = 0.5 * tm.dot(kb, prev_mf2 + curr_mf2)
           d2 = -0.5 * tm.dot(kb, prev_mf1 + curr_mf1)
           self.data[idx] = tm.vec2(d1, d2)
    

@ti.data_oriented
class GradKappas:
    def __init__(self, lengths: Lengths, tangents: Tangents, curvature_binormals: CurvatureBinormals,
                 mat_frame1: MaterialFrames1, mat_frame2: MaterialFrames2, kappas: Kappas):
        self.lengths = lengths
        self.tangents = tangents
        self.curvature_binormals = curvature_binormals
        self.mat_frame1 = mat_frame1
        self.mat_frame2 = mat_frame2
        self.kappas = kappas

        self.start_index = 1
        self.end_index = self.curvature_binormals.size
        self.size = self.curvature_binormals.size
        self.data = ti.Matrix.field(11, 2, dtype=float, shape=self.size)

    @ti.kernel
    def compute(self):
        for idx in range(self.start_index, self.end_index):
            prev_length = self.lengths.data[idx - 1]
            curr_length = self.lengths.data[idx]
            prev_tangent = self.tangents.data[idx - 1]
            curr_tangent = self.tangents.data[idx]
            prev_mf1 = self.mat_frame1.data[idx - 1]
            curr_mf1 = self.mat_frame1.data[idx]
            prev_mf2 = self.mat_frame2.data[idx - 1]
            curr_mf2 = self.mat_frame2.data[idx]
            kb = self.curvature_binormals.data[idx]
            kappa = self.kappas.data[idx]

            chi = 1.0 + tm.dot(prev_tangent, curr_tangent)
            chi = max(chi, Utils.EPSILON)

            tilde_t = (prev_tangent + curr_tangent) / chi
            tilde_d1 = (prev_mf1 + curr_mf1) / chi
            tilde_d2 = (prev_mf2 + curr_mf2) / chi

            Dkappa0De = 1.0 / prev_length * (-kappa[0] * tilde_t + tm.cross(curr_tangent, tilde_d2))
            Dkappa0Df = 1.0 / curr_length * (-kappa[0] * tilde_t - tm.cross(prev_tangent, tilde_d2))
            Dkappa1De = 1.0 / prev_length * (-kappa[1] * tilde_t - tm.cross(curr_tangent, tilde_d1))
            Dkappa1Df = 1.0 / curr_length * (-kappa[1] * tilde_t - tm.cross(prev_tangent, tilde_d1))
            
            self.data[idx][0:3, 0] = Dkappa0De
            self.data[idx][4:7, 0] = Dkappa0De - Dkappa0Df
            self.data[idx][8:11, 0] = Dkappa0Df
            self.data[idx][0:3, 1] = -Dkappa1De
            self.data[idx][4:7, 1] = Dkappa1De - Dkappa1Df
            self.data[idx][8:11, 1] = Dkappa1Df

            self.data[idx][3, 0] = -0.5 * tm.dot(kb, prev_mf1)
            self.data[idx][7, 0] = -0.5 * tm.dot(kb, curr_mf1)
            self.data[idx][3, 1] = -0.5 * tm.dot(kb, prev_mf2)
            self.data[idx][7, 1] = -0.5 * tm.dot(kb, curr_mf2)


@ti.data_oriented
class HessKappas:
    def __init__(self, lengths: Lengths, tangents: Tangents, curvature_binormals: CurvatureBinormals,
                 mat_frame1: MaterialFrames1, mat_frame2: MaterialFrames2, kappas: Kappas):
        self.lengths = lengths
        self.tangents = tangents
        self.curvature_binormals = curvature_binormals
        self.mat_frame1 = mat_frame1
        self.mat_frame2 = mat_frame2
        self.kappas = kappas

        self.start_index = 1
        self.end_index = self.curvature_binormals.size
        self.size = self.curvature_binormals.size
        self.data = ti.Matrix.field(11, 11, dtype=float, shape=(self.size, 2))

    @ti.kernel
    def compute(self):
        for idx in range(self.start_index, self.end_index):
            prev_length = self.lengths.data[idx - 1]
            curr_length = self.lengths.data[idx]
            prev_length2 = prev_length * prev_length
            curr_length2 = curr_length * curr_length

            prev_tangent = self.tangents.data[idx - 1]
            curr_tangent = self.tangents.data[idx]
            prev_mf1 = self.mat_frame1.data[idx - 1]
            curr_mf1 = self.mat_frame1.data[idx]
            prev_mf2 = self.mat_frame2.data[idx - 1]
            curr_mf2 = self.mat_frame2.data[idx]

            kappa = self.kappas.data[idx]
            kb = self.curvature_binormals.data[idx]

            chi = max(1.0 + tm.dot(prev_tangent, curr_tangent), Utils.EPSILON)

            tilde_t = (prev_tangent + curr_tangent) / chi
            tilde_d1 = (prev_mf1 + curr_mf1) / chi
            tilde_d2 = (prev_mf2 + curr_mf2) / chi

            tt_o_tt = tilde_t.outer_product(tilde_t)
            tf_c_d2t_o_tt = tm.cross(curr_tangent, tilde_d2).outer_product(tilde_t)
            tt_o_tf_c_d2t = tf_c_d2t_o_tt.transpose()
            kb_o_d2e = kb.outer_product(prev_mf2)
            d2e_o_kb = kb_o_d2e.transpose()

            Id3 = tm.mat3(1, 0, 0, 0, 1, 0, 0, 0, 1)

            # Below start compute value of data[idx, 1]
            D2kappa1De2 = 1.0 / prev_length2 * \
                        (2 * kappa[0] * tt_o_tt - (tf_c_d2t_o_tt + tt_o_tf_c_d2t)) - kappa[0] / (chi * prev_length2) * (Id3 - prev_tangent.outer_product(prev_tangent)) + \
                        1.0 / (4.0 * prev_length2) * (kb_o_d2e + d2e_o_kb)
            te_c_d2t_o_tt = tm.cross(prev_tangent, tilde_d2).outer_product(tilde_t)
            tt_o_te_c_d2t = te_c_d2t_o_tt.transpose()
            kb_o_d2f = kb.outer_product(curr_mf2)
            d2f_o_kb = kb_o_d2f.transpose()

            D2kappa1Df2 = 1.0 / curr_length2 * \
                        (2 * kappa[0] * tt_o_tt + (te_c_d2t_o_tt + tt_o_te_c_d2t)) - kappa[0] / (chi * curr_length2) * (Id3 - curr_tangent.outer_product(curr_tangent)) + \
                        1.0 / (4.0 * curr_length2) * (kb_o_d2f + d2f_o_kb)
            
            D2kappa1DeDf = -kappa[0] / (chi * prev_length * curr_length) * (Id3 + prev_tangent.outer_product(curr_tangent)) + \
                        1.0 / (prev_length * curr_length) * (2 * kappa[0] * tt_o_tt - tf_c_d2t_o_tt + tt_o_te_c_d2t - Utils.t_getCrossMatrix(tilde_d2))
            D2kappa1DfDe = D2kappa1DeDf.transpose()

            D2kappa1Dthetae2 = -0.5 * tm.dot(kb, prev_mf2)
            D2kappa1Dthetaf2 = -0.5 * tm.dot(kb, curr_mf2)
            D2kappa1DeDthetae = 1.0 / prev_length * (0.5 * tm.dot(kb, prev_mf1) * tilde_t - 1.0 / chi * tm.cross(curr_tangent, prev_mf1))
            D2kappa1DeDthetaf = 1.0 / prev_length * (0.5 * tm.dot(kb, curr_mf1) * tilde_t - 1.0 / chi * tm.cross(curr_tangent, curr_mf1))
            D2kappa1DfDthetae = 1.0 / curr_length * (0.5 * tm.dot(kb, prev_mf1) * tilde_t - 1.0 / chi * tm.cross(curr_tangent, prev_mf1))
            D2kappa1DfDthetaf = 1.0 / curr_length * (0.5 * tm.dot(kb, curr_mf1) * tilde_t - 1.0 / chi * tm.cross(curr_tangent, curr_mf1))

            self.data[idx, 0][0:3, 0:3] = D2kappa1De2
            self.data[idx, 0][0:3, 4:7] = -D2kappa1De2 + D2kappa1DeDf
            self.data[idx, 0][4:7, 0:3] = -D2kappa1De2 + D2kappa1DfDe
            self.data[idx, 0][4:7, 4:7] = D2kappa1De2 - (D2kappa1DeDf + D2kappa1DfDe) + D2kappa1Df2
            self.data[idx, 0][0:3, 8:11] = -D2kappa1DeDf
            self.data[idx, 0][8:11, 0:3] = -D2kappa1DfDe
            self.data[idx, 0][4:7, 8:11] = D2kappa1DeDf - D2kappa1Df2
            self.data[idx, 0][8:11, 4:7] = D2kappa1DfDe - D2kappa1Df2
            self.data[idx, 0][8:11, 8:11] = D2kappa1Df2

            self.data[idx, 0][3, 3] = D2kappa1Dthetae2
            self.data[idx, 0][7, 7] = D2kappa1Dthetaf2
            self.data[idx, 0][3, 7] = 0
            self.data[idx, 0][7, 3] = 0

            self.data[idx, 0][0:3, 3] = -D2kappa1DeDthetae
            self.data[idx, 0][3, 0:3] = self.data[idx, 0][0:3, 3].transpose()

            self.data[idx, 0][4:7, 3] = D2kappa1DeDthetae - D2kappa1DfDthetae
            self.data[idx, 0][3, 4:7] = self.data[idx, 0][4:7, 3].transpose()
            self.data[idx, 0][8:11, 3] = D2kappa1DfDthetae
            self.data[idx, 0][3, 8:11] = self.data[idx, 0][8:11, 3].transpose()
            self.data[idx, 0][0:3, 7] = D2kappa1DeDthetaf
            self.data[idx, 0][7, 0:3] = self.data[idx, 0][0:3, 7].transpose()
            self.data[idx, 0][4:7, 7] = D2kappa1DeDthetaf - D2kappa1DfDthetaf
            self.data[idx, 0][7, 4:7] = self.data[idx, 0][4:7, 7].transpose()
            self.data[idx, 0][8:11, 7] = D2kappa1DfDthetaf
            self.data[idx, 0][7, 8:11] = self.data[idx, 0][8:11, 7].transpose()

            # Below start compute value of data[idx, 1]
            tf_c_d1t_o_tt = tm.cross(curr_tangent, tilde_d1).outer_product(tilde_t)
            tt_o_tf_c_d1t = tf_c_d1t_o_tt.transpose()
            kb_o_d1e = kb.outer_product(prev_mf1)
            d1e_o_kb = kb_o_d1e.transpose()

            D2kappa2De2 = 1.0 / prev_length2 * (2 * kappa[1] * tt_o_tt + (tf_c_d1t_o_tt + tt_o_tf_c_d1t)) - \
                            kappa[1] / (chi * prev_length2) * (Id3 - prev_tangent.outer_product(prev_tangent)) - \
                            1.0 / (4.0 * prev_length2) * (kb_o_d1e + d1e_o_kb)
            te_c_d1t_o_tt = tm.cross(prev_tangent, tilde_d1).outer_product(tilde_t)
            tt_o_te_c_d1t = te_c_d1t_o_tt.transpose()
            kb_o_d1f = kb.outer_product(curr_mf1)
            d1f_o_kb = kb_o_d1f.transpose()

            D2kappa2Df2 = 1.0 / curr_length2 * (2 * kappa[1] * tt_o_tt + (te_c_d1t_o_tt + tt_o_te_c_d1t)) - \
                            kappa[1] / (chi * curr_length2) * (Id3 - curr_tangent.outer_product(curr_tangent)) - \
                            1.0 / (4.0 * curr_length2) * (kb_o_d1f + d1f_o_kb)

            D2kappa2DeDf = -kappa[1] / (chi * prev_length * curr_length) * (Id3 + prev_tangent.outer_product(curr_tangent)) + \
                            1.0 / (prev_length * curr_length) * (2 * kappa[1] * tt_o_tt + tf_c_d1t_o_tt - tt_o_te_c_d1t + Utils.t_getCrossMatrix(tilde_d1))
            D2kappa2DfDe = D2kappa2DeDf.transpose()

            D2kappa2Dthetae2 = 0.5 * tm.dot(kb, prev_mf1)
            D2kappa2Dthetaf2 = 0.5 * tm.dot(kb, curr_mf1)

            D2kappa2DeDthetae = 1.0 / prev_length * (0.5 * tm.dot(kb, prev_mf2) * tilde_t - 1.0 / chi * tm.cross(curr_tangent, prev_mf2))
            D2kappa2DeDthetaf = 1.0 / prev_length * (0.5 * tm.dot(kb, curr_mf2) * tilde_t - 1.0 / chi * tm.cross(curr_tangent, curr_mf2))
            D2kappa2DfDthetae = 1.0 / curr_length * (0.5 * tm.dot(kb, prev_mf2) * tilde_t + 1.0 / chi * tm.cross(prev_tangent, prev_mf2))
            D2kappa2DfDthetaf = 1.0 / curr_length * (0.5 * tm.dot(kb, curr_mf2) * tilde_t + 1.0 / chi * tm.cross(prev_tangent, curr_mf2))

            self.data[idx, 1][0:3, 0:3] = D2kappa2De2
            self.data[idx, 1][0:3, 4:7] = -D2kappa2De2 + D2kappa2DeDf
            self.data[idx, 1][4:7, 0:3] = -D2kappa2De2 + D2kappa2DfDe
            self.data[idx, 1][4:7, 4:7] = D2kappa2De2 - (D2kappa2DeDf + D2kappa2DfDe) + D2kappa2Df2
            self.data[idx, 1][0:3, 8:11] = -D2kappa2DeDf
            self.data[idx, 1][8:11, 0:3] = -D2kappa2DfDe
            self.data[idx, 1][4:7, 8:11] = D2kappa2DeDf - D2kappa2Df2
            self.data[idx, 1][8:11, 4:7] = D2kappa2DfDe - D2kappa2Df2
            self.data[idx, 1][8:11, 8:11] = D2kappa2Df2

            self.data[idx, 1][3, 3] = D2kappa2Dthetae2
            self.data[idx, 1][7, 7] = D2kappa2Dthetaf2
            self.data[idx, 1][3, 7] = 0
            self.data[idx, 1][7, 3] = 0

            self.data[idx, 1][0:3, 3] = -D2kappa2DeDthetae
            self.data[idx, 1][3, 0:3] = self.data[idx, 1][0:3, 3].transpose()

            self.data[idx, 1][4:7, 3] = D2kappa2DeDthetae - D2kappa2DfDthetae
            self.data[idx, 1][3, 4:7] = self.data[idx, 1][4:7, 3].transpose()
            self.data[idx, 1][8:11, 3] = D2kappa2DfDthetae
            self.data[idx, 1][3, 8:11] = self.data[idx, 1][8:11, 3].transpose()
            self.data[idx, 1][0:3, 7] = D2kappa2DeDthetaf
            self.data[idx, 1][7, 0:3] = self.data[idx, 1][0:3, 7].transpose()
            self.data[idx, 1][4:7, 7] = D2kappa2DeDthetaf - D2kappa2DfDthetaf
            self.data[idx, 1][7, 4:7] = self.data[idx, 1][4:7, 7].transpose()
            self.data[idx, 1][8:11, 7] = D2kappa2DfDthetaf
            self.data[idx, 1][7, 8:11] = self.data[idx, 1][8:11, 7].transpose()
