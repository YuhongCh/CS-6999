import taichi as ti
import taichi.math as tm

import Liquid_Hair_Interaction.Simulation.MathUtils as MathUtils
import Liquid_Hair_Interaction.Simulation.DER.Utils as Utils

from Liquid_Hair_Interaction.Simulation.DER.Parameters import *
from Liquid_Hair_Interaction.Simulation.DER.Components.BendingProducts import *
from Liquid_Hair_Interaction.Simulation.DER.Components.DOFs import *
from Liquid_Hair_Interaction.Simulation.DER.Components.ElasticParameters import *
from Liquid_Hair_Interaction.Simulation.DER.Components.Kappas import *
from Liquid_Hair_Interaction.Simulation.DER.Components.MaterialFrames import *
from Liquid_Hair_Interaction.Simulation.DER.Components.ReferenceFrames import *
from Liquid_Hair_Interaction.Simulation.DER.Components.Twists import *
from Liquid_Hair_Interaction.Simulation.DER.StrandState import StrandState, StartState

from Liquid_Hair_Interaction.Simulation.Forces.GravityForce import GravityForce
from Liquid_Hair_Interaction.Simulation.DER.Forces.TwistForce import TwistForce
from Liquid_Hair_Interaction.Simulation.DER.Forces.StretchForce import StretchForce
from Liquid_Hair_Interaction.Simulation.DER.Forces.BendingForce import BendingForce


@ti.data_oriented
class StrandForce:
    def __init__(self, vertices, degrees, parameters: StrandParameters):
        base_bending_matrix = BaseBendingMatrix(parameters.radius, 0)
        self.start_state = StartState(vertices, degrees,base_bending_matrix)
        self.curr_state = StrandState(vertices, degrees)
        self.parameters = parameters

        """ Strand rest state """
        vertex_size = vertices.shape[0]
        edge_size = vertex_size - 1
        self.total_rest_length = 0.0
        self.rest_lengths = ti.field(dtype=float, shape=edge_size)
        self.voronoi_lengths = ti.field(dtype=float, shape=vertex_size)
        self.inv_voronoi_lengths = ti.field(dtype=float, shape=vertex_size)
        self.vertex_masses = ti.field(dtype=float, shape=vertex_size)
        self.rest_kappas = ti.Vector.field(2, dtype=float, shape=vertex_size)
        self.rest_twists = ti.field(dtype=float, shape=vertex_size)
        self.updateRestState(vertices, degrees, 0.0)
        """ -------- End --------- """

        """ DER Hair Forces """
        self.force_types = [StretchForce, BendingForce, TwistForce, GravityForce]
        self.force = ti.Vector.field(4, shape=vertices.shape[0])
        self.velocity = ti.Vector.field(4, shape=vertices.shape[0])
        """ --------- End ----------"""

    def getVertexSize(self) -> int:
        return self.curr_state.dofs.getVertexSize()

    @ti.func
    def t_getVertexSize(self) -> int:
        return self.curr_state.dofs.t_getVertexSize()

    @ti.func
    def t_getEdgeSize(self) -> int:
        return self.t_getVertexSize() - 1

    @ti.func
    def t_computeRestLengthDependency(self):
        self.total_rest_length = 0.0
        for i in self.rest_lengths:
            self.total_rest_length += self.rest_lengths[i]

        edge_size = self.t_getEdgeSize()
        self.voronoi_lengths[0] = 0.5 * self.rest_lengths[0]
        for i in range(1, edge_size):
            self.voronoi_lengths[i] = 0.5 * (self.rest_lengths[i-1] + self.rest_lengths[i])
        self.voronoi_lengths[edge_size] = 0.5 * self.rest_lengths[edge_size - 1]

        vertex_size = edge_size + 1
        for i in range(vertex_size):
            r2 = self.parameters.radius * self.parameters.radius
            self.vertex_masses[i] = self.parameters.density * self.voronoi_lengths[i] * MathUtils.PI * r2
            self.inv_voronoi_lengths[i] = 1.0 / self.voronoi_lengths[i]

    @ti.kernel
    def updateRestState(self, vertices, degrees, damping: float):
        rest_state = StartState(vertices, degrees)
        edge_size = self.t_getEdgeSize()
        for i in range(edge_size):
            self.rest_lengths[i] = (1 - damping) * rest_state.lengths.data[i] + damping * self.rest_lengths[i]
        self.t_computeRestLengthDependency()
        for i in range(edge_size):
            self.rest_kappas[i] = (1 - damping) * rest_state.kappas.data[i] + damping * self.rest_kappas[i]
            self.rest_twists[i] = (1 - damping) * rest_state.twists.data[i] + damping * self.rest_twists[i]

    def accumulateForce(self):
        for force_type in self.force_types:
            ftype = force_type(self)
            ftype.accumulateForce()

    @ti.kernel
    def clearForce(self):
        for i in self.force:
            self.force[i] = tm.vec4(0, 0, 0, 0)

    @ti.kernel
    def updateStrandState(self, dt: float):
        """ TODO: Delete this after test """
        self.force[0] = tm.vec4(0, 0, 0, self.force[0][3])
        for i in self.force:
            self.velocity[i] += self.force[i] * dt
            self.curr_state.dofs.vertices[i] += self.velocity[i][0:3] * dt
            self.curr_state.dofs.degrees[i] += self.velocity[i][3] * dt