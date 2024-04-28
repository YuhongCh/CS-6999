import taichi as ti
import taichi.math as tm

import Liquid_Hair_Interaction.Simulation.MathUtils as MathUtils
from Liquid_Hair_Interaction.Simulation.DER.StrandForce import StrandForce

@ti.data_oriented
class BendingForce:
    """ Apply Bending Force to wanted objects """
    def __init__(self, strand_force: StrandForce):
        self.strand_force = strand_force
        self.start_index = 1
        self.end_index = strand_force.getVertexSize() - 1

    @ti.kernel
    def accumulateForce(self):
        vertex_size = self.strand_force.t_getVertexSize()
        for i in range(self.start_index, self.end_index):
            bending_matrix = self.strand_force.parameters.t_getViscousBendingMatrix(i, vertex_size)
            start_kappa = self.strand_force.start_state.kappas.data[i]
            inv_length = self.strand_force.inv_voronoi_lengths[i]
            kappa = self.strand_force.curr_state.kappas.data[i]
            grad_kappa = self.strand_force.curr_state.grad_kappas.data[i]
            force = -inv_length * grad_kappa @ bending_matrix @ (kappa - start_kappa)

            self.strand_force.force[i-1] += force[0:4]
            self.strand_force.force[i] += force[4:8]
            self.strand_force.force[i+1] += tm.vec4(force[8], force[9], force[10], 0)
