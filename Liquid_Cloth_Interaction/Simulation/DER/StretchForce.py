import taichi as ti
import taichi.math as tm

from math import pi
from copy import copy
from Liquid_Cloth_Interaction.Simulation.Parameters import ElasticParameters
from Liquid_Cloth_Interaction.Simulation.DER.States import DER_StrandState, DER_RestState

@ti.data_oriented
class StretchForce:
    """ Apply gravity Force to wanted objects """
    def __init__(self, curr_state: DER_StrandState, rest_state: DER_RestState, elastic: ElasticParameters):
        self.curr_state = copy(curr_state)
        self.rest_state = copy(rest_state)
        self.elastic = copy(elastic)

    @ti.kernel
    def t_accumulate_force(self, global_force: ti.types.ndarray(dtype=tm.vec4)):
        num_vertices = global_force.shape[0]
        coef = self.elastic.radius * self.elastic.radius * pi * self.elastic.youngs_modulus
        for i in ti.ndrange(num_vertices-1):
            length = self.curr_state.edges.length[i]
            rest_length = self.rest_state.rest_length[i]
            dir = self.curr_state.edges.dir[i]
            force = coef * (length / rest_length - 1.0) * dir
            force4 = tm.vec4(force.x, force.y, force.z, 0)
            global_force[i] += force4
            global_force[i+1] -= force4

    # TODO: Complete the Computation of Force Jacobi Matrix
    # @ti.kernel
    # def t_accumulate_jacobian(self, global_jacobian: ):


