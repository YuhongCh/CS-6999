import taichi as ti
import taichi.math as tm

from Liquid_Hair_Interaction.Simulation.DER.StrandForce import StrandForce

@ti.data_oriented
class TwistForce:
    """ Apply Bending Force to wanted objects """
    def __init__(self, strand_force: StrandForce):
        self.strand_force = strand_force
        self.start_index = 1
        self.end_index = strand_force.getVertexSize() - 1

    @ti.kernel
    def accumulateForce(self):
        vertex_size = self.strand_force.t_getVertexSize()
        for i in range(self.start_index, self.end_index):
            kt = self.strand_force.parameters.t_getViscousKt()
            start_twist = self.strand_force.start_state.twists.data[i]
            inv_length = self.strand_force.inv_voronoi_lengths[i]
            twist = self.strand_force.curr_state.twists.data[i]
            grad_twist = self.strand_force.curr_state.grad_twists.data[i]
            force = -kt * inv_length * (twist - start_twist) * grad_twist

            self.strand_force.force[i-1] += force[0:4]
            self.strand_force.force[i] += force[4:8]
            self.strand_force.force[i+1] += tm.vec4(force[8], force[9], force[10], 0)
