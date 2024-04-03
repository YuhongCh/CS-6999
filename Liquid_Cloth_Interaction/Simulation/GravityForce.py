import taichi as ti
import taichi.math as tm

@ti.data_oriented
class GravityForce:
    """ Apply gravity Force to wanted objects """
    def __init__(self):
        pass

    @ti.kernel
    def t_accumulate_force(self, global_force: ti.types.ndarray(dtype=tm.vec4)):
        num_vertices = global_force.shape[0]
        for i in ti.ndrange(num_vertices):
            global_force[i] += tm.vec4(0, -9.81, 0, 0)

    # TODO: Complete the Computation of Force Jacobi Matrix
