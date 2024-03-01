import taichi as ti
import taichi.math as tm

@ti.data_oriented
class Integrator:
    def __init__(self, dt: float, type: str, criterion: float):
        self.dt = dt
        self.type = type
        self.criterion = criterion

    @ti.func
    def integrate(self, force: ti.types.ndarray(dtype=tm.vec4),
                        velocity: ti.types.ndarray(dtype=tm.vec4)):
        num_vertices = force.shape[0]
        for i in ti.ndrange(num_vertices):
            velocity[i] += force[i][0:3] * self.dt


