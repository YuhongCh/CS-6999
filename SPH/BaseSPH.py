import taichi as ti
import taichi.math as tm

from ParticleSystem import ParticleSystem, ParticleType
from Config import SPHConfig
from NeighborSearch import NeighborSearch
from Kernel import CubicSplineKernel as Kernel


@ti.data_oriented
class BaseSPH:
    def __init__(self, config: SPHConfig, particle_system: ParticleSystem, neighbor_search: NeighborSearch):
        self.viscosity_mu = config.viscosity_mu
        self.particle_mass = config.particle_mass
        self.dt = config.dt
        self.particle_system = particle_system
        self.neighbor_search = neighbor_search
        self.kernel = Kernel()

        # shallow copy needed variables from particle system
        self.num_particle = self.particle_system.get_particle_num(ParticleType.NonEmpty)

        # additional containers
        self.fluid_kernel_sum = ti.field(dtype=float, shape=self.num_particle)
        self.boundary_kernel_sum = ti.field(dtype=float, shape=self.num_particle)
        self.viscosity_force = ti.Vector.field(3, dtype=float, shape=self.num_particle)
        self.external_force = ti.Vector.field(3, dtype=float, shape=self.num_particle)

    @ti.func
    def _compute_density_task(self, pi: int, pj: int, ret: ti.template()):
        x_ij = self.particle_system.position[pi] - self.particle_system.position[pj]
        weight = self.kernel.get_weight(x_ij, self.particle_system.support_radius)
        correcting_coefficient = 1.0
        if self.particle_system.type[pj] == ParticleType.Boundary:
            correcting_coefficient = (1 / self.particle_system.particle_volume - self.fluid_kernel_sum[pj]) / self.boundary_kernel_sum[pj]
        ret += self.particle_system.particle_mass * correcting_coefficient * weight

    @ti.kernel
    def _compute_density(self):
        for pid in range(self.num_particle):
            if self.particle_system.type[pid] == ParticleType.Boundary:
                continue
            self.neighbor_search.for_each_neighbors(pid, self._compute_density_task, self.particle_system.density[pid])
            self.particle_system.density[pid] = max(self.particle_system.density[pid], self.particle_system.ref_density)

    @ti.func
    def _compute_kernel_sum_task(self, pi: int, pj: int, ret: ti.template()):
        x_ij = self.particle_system.position[pi] - self.particle_system.position[pj]
        if self.particle_system.type[pj] == ParticleType.Fluid:
            self.fluid_kernel_sum[pi] += self.kernel.get_weight(x_ij, self.particle_system.support_radius) * self.particle_mass
        elif self.particle_system.type[pj] == ParticleType.Boundary:
            self.boundary_kernel_sum[pi] += self.kernel.get_weight(x_ij, self.particle_system.support_radius) * self.particle_mass

    @ti.kernel
    def _compute_kernel_sum(self):
        for pid in range(self.num_particle):
            dummy = 0
            self.neighbor_search.for_each_neighbors(pid, self._compute_kernel_sum_task, dummy)

    def compute_density(self):
        self._compute_kernel_sum()
        self._compute_density()

    @ti.func
    def _compute_viscosity_force_task(self, pi: int, pj: int, ret: ti.template()):
        coef = self.particle_mass / self.particle_system.density[pj]
        v_ij = self.particle_system.velocity[pi] - self.particle_system.velocity[pj]
        x_ij = self.particle_system.position[pi] - self.particle_system.position[pj]
        h = self.particle_system.support_radius
        grad_weight = self.kernel.get_gradient_weight(x_ij, h)
        ret += coef * tm.dot(v_ij, x_ij) / (tm.dot(x_ij, x_ij) + 0.01 * h * h) * grad_weight

    @ti.kernel
    def _compute_viscosity_force(self):
        # TODO: checkout https://peridyno.com/zh/topics/particlesystem/artificalviscosity/ for artificial viscosity
        for pid in range(self.num_particle):
            self.neighbor_search.for_each_neighbors(pid, self._compute_viscosity_force_task, self.viscosity_force[pid])
            self.viscosity_force[pid] *= self.viscosity_mu * 10 # 10 stands for 2 * (2 + dim)

    @ti.kernel
    def _compute_external_force(self):
        for pid in range(self.num_particle):
            # external gravity force
            self.external_force[pid] = tm.vec3(0, -9.81, 0) * self.particle_system.density[pid]

    @ti.kernel
    def _reset_parameters(self):
        for pid in range(self.num_particle):
            self.viscosity_force[pid] = tm.vec3(0, 0, 0)
            self.external_force[pid] = tm.vec3(0, 0, 0)
            self.fluid_kernel_sum[pid] = 0
            self.boundary_kernel_sum[pid] = 0

    def reset_derived_parameters(self):
        raise NotImplementedError

    def reset_parameters(self):
        self._reset_parameters()