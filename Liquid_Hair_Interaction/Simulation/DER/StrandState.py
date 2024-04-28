import taichi as ti

from Liquid_Hair_Interaction.Simulation.DER.Components.BendingProducts import *
from Liquid_Hair_Interaction.Simulation.DER.Components.DOFs import *
from Liquid_Hair_Interaction.Simulation.DER.Components.ElasticParameters import *
from Liquid_Hair_Interaction.Simulation.DER.Components.Kappas import *
from Liquid_Hair_Interaction.Simulation.DER.Components.MaterialFrames import *
from Liquid_Hair_Interaction.Simulation.DER.Components.ReferenceFrames import *
from Liquid_Hair_Interaction.Simulation.DER.Components.Twists import *


@ti.data_oriented
class StrandState:
    def __init__(self, vertices, degrees, base_bending_matrix: BaseBendingMatrix):
        self.dofs = DOFs(vertices, degrees)
        self.edges = Edges(self.dofs)
        self.lengths = Lengths(self.edges)
        self.tangents = Tangents(self.edges, self.lengths)
        self.ref_frame1 = ReferenceFrames1(self.tangents)
        self.ref_frame2 = ReferenceFrames2(self.tangents, self.ref_frame1)
        self.ref_twists = ReferenceTwists(self.tangents, self.ref_frame1)
        self.twists = Twists(self.dofs, self.ref_twists)
        self.curvature_binormals = CurvatureBinormals(self.tangents)
        self.trig_thetas = TrigThetas(self.dofs)
        self.mat_frame1 = MaterialFrames1(self.trig_thetas, self.ref_frame1, self.ref_frame2)
        self.mat_frame2 = MaterialFrames2(self.trig_thetas, self.ref_frame1, self.ref_frame2)
        self.kappas = Kappas(self.curvature_binormals, self.mat_frame1, self.mat_frame2)
        self.grad_kappas = GradKappas(self.lengths, self.tangents, self.curvature_binormals, self.mat_frame1, self.mat_frame2, self.kappas)
        self.grad_twists = GradTwists(self.lengths, self.curvature_binormals)
        self.grad_twists_squared = GradTwistsSquared(self.grad_twists)
        self.hess_kappas = HessKappas(self.lengths, self.tangents, self.curvature_binormals, self.mat_frame1, self.mat_frame2, self.kappas)
        self.hess_twists = HessTwists(self.tangents, self.lengths, self.curvature_binormals)
        self.bending_products = BendingProducts(base_bending_matrix, self.grad_kappas)


@ti.dataclass
class StartState:
    def __init__(self, vertices, degrees):
        self.dofs = DOFs(vertices, degrees)
        self.edges = Edges(self.dofs)
        self.lengths = Lengths(self.edges)
        self.tangents = Tangents(self.edges, self.lengths)
        self.ref_frame1 = ReferenceFrames1(self.tangents)
        self.ref_frame2 = ReferenceFrames2(self.tangents, self.ref_frame1)
        self.ref_twists = ReferenceTwists(self.tangents, self.ref_frame1)
        self.twists = Twists(self.dofs, self.ref_twists)
        self.curvature_binormals = CurvatureBinormals(self.tangents)
        self.trig_thetas = TrigThetas(self.dofs)
        self.mat_frame1 = MaterialFrames1(self.trig_thetas, self.ref_frame1, self.ref_frame2)
        self.mat_frame2 = MaterialFrames2(self.trig_thetas, self.ref_frame1, self.ref_frame2)
        self.kappas = Kappas(self.curvature_binormals, self.mat_frame1, self.mat_frame2)
