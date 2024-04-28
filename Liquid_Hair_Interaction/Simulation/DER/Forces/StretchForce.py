import taichi as ti
import taichi.math as tm

from Liquid_Hair_Interaction.Simulation.DER.StrandForce import StrandForce

@ti.data_oriented
class StretchForce:
    """ Apply gravity Force to wanted objects """
    def __init__(self, strand_force: StrandForce):
        self.strand_force = strand_force
        self.start_index = 0
        self.end_index = strand_force.getVertexSize() - 1

    @ti.kernel
    def accumulateFrce(self):
        vertex_size = self.strand_force.curr_state.dofs.t_getVertexSize()
        for i in range(self.start_index, self.end_index):
            ks = self.strand_force.parameters.t_getViscousKs(i, vertex_size)

            rest_length = self.strand_force.rest_lengths[i]
            length = self.strand_force.curr_state.lengths.data[i]
            dir = self.strand_force.curr_state.tangents.data[i]

            force = ks * (length / rest_length - 1.0) * dir
            self.strand_force.force[i] += force
            self.strand_force.force[i+1] -= force
