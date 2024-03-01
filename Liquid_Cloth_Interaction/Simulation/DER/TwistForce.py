import taichi as ti
import taichi.math as tm

from math import pi
from copy import copy
from Liquid_Cloth_Interaction.Simulation.ElasticParameters import ElasticParameters
from Liquid_Cloth_Interaction.Simulation.DER.States import DER_StrandState, DER_RestState

@ti.data_oriented
class TwistForce:
    """ Apply Bending Force to wanted objects """
    def __init__(self, curr_state: DER_StrandState, rest_state: DER_RestState, elastic: ElasticParameters):
        self.curr_state = copy(curr_state)
        self.rest_state = copy(rest_state)
        self.elastic = copy(elastic)

    @ti.kernel
    def t_accumulate_force(self, global_force: ti.types.ndarray(dtype=tm.vec4)):
        num_vertices = global_force.shape[0]
        coef = self.elastic.radius**4 * pi * self.elastic.shear_modulus * 0.25
        for i in ti.ndrange((1, num_vertices - 1)):
            rest_twist = self.rest_state.rest_twist[i-1]
            inv_length = self.rest_state.inv_voronoi_length[i-1]
            twist = self.curr_state.twist.twist[i-1]
            grad_twist = self.curr_state.grad_twist.grad_twist[i-1]

            tot_force = coef * inv_length * (twist - rest_twist) * grad_twist
            global_force[i - 1] += tot_force[0:4]
            global_force[i] += tot_force[4:8]
            global_force[i + 1] += tm.vec4(tot_force[8], tot_force[9], tot_force[10], 0)


