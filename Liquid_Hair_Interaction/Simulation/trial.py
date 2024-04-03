import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.cpu, debug=True)

n = 10


# @ti.dataclass
# class Particle:
#     x: tm.vec3


# @ti.data_oriented
# class ParticleSystem:
#     def __init__(self):
#         self.particles = Particle.field(shape=n)
#         self.part2cell = np.zeros(n)
#         self.cell2part = np.zeros(n)

#         self.grid_size = 0.1
#         self.start_domain = tm.vec3(0, 0, 0)
#         self.end_domain = tm.vec3(1, 1, 1)

#     @ti.kernel
#     def t_init(self, part2cell: ti.types.ndarray()):
#         for i in range(n):
#             self.particles[i].x.x = ti.random()
#             self.particles[i].x.y = ti.random()
#             self.particles[i].x.z = ti.random()

#         for i in range(n):
#             idx = self.get_index(self.particles[i])
#             part2cell[i] = idx

#     def p_init(self):
#         self.t_init(self.part2cell)
#         self.cell2part = np.argsort(self.part2cell)


#     @ti.func
#     def get_index(self, p: Particle) -> int:
#         x_idx = int(p.x.x / self.grid_size - 0.5)
#         y_idx = int(p.x.y / self.grid_size - 0.5)
#         z_idx = int(p.x.z / self.grid_size - 0.5)
#         return y_idx * 10 * 10 + z_idx * 10 + x_idx




ps = ParticleSystem()
ps.p_init()

print(ps.particles)
print(ps.part2cell)
print(ps.cell2part)

