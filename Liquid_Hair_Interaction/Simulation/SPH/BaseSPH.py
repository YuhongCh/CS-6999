import taichi as ti
import taichi.math as tm
import math

from Liquid_Hair_Interaction.Simulation.SPH.ParticleSystem import ParticleSystem
from Liquid_Hair_Interaction.Simulation.SPH.SPHConfig import SPHConfig
from Liquid_Hair_Interaction.Simulation.SPH.NeighborSearch import UniformGridNS
from Liquid_Hair_Interaction.Simulation.SPH.Kernel import KernelManager

@ti.data_oriented
class BaseSPH:
    def __init__(self, config: SPHConfig, particle_system: ParticleSystem):
        # copy over the needed SPH configuration
        self.viscosity = config.viscosity

        # copy over parameters
        self.particle_system = particle_system
        self.config = config

        # construct local needed variables
        self.kernel = KernelManager.p_GetKernel(self.config.kernel_type)
        self.neighbor_search = UniformGridNS(self.config.grid_radius, 
                                             self.config.max_part_in_grid,
                                             self.config.max_neighbor_count,
                                             self.particle_system)
        
    """
    Below are Helper functions
    """
    @ti.func
    def t_ComputeDensityTask(self, pi: int, pj: int, ret: ti.template()):
        pi_pos = self.ps.particles[pi].x
        pj_pos = self.ps.particles[pj].x
        weight = self.kernel.t_GetWeight(pi_pos - pj_pos, self.ps.support_radius)
        ret += self.ps.particles[pj].mass * weight

    @ti.func
    def t_ComputeViscosityTask(self, pi: int, pj: int, ret: ti.template()):
        r = self.ps.particles[pi].x - self.ps.particles[pj].x
        vi = self.ps.particles[pi].v
        vj = self.ps.particles[pj].v
        mass = self.ps.particles[pi].mass
        density = self.ps.particles[pi].density
        # TODO: Check if this grad_weight computation is correct
        grad_weight = self.kernel.t_GetGrad2Weight(r, self.ps.support_radius)
        ret += mass * mass * self.viscosity * (vj - vi) / density * grad_weight

    @ti.func
    def t_ComputeParticleCollisionTask(self, pi: int, pj: int, ret: ti.template()):
        r = self.ps.particles[pi].x - self.ps.particles[pj].x
        r_norm = tm.length(r)
        r_norm2 = r_norm * r_norm
        if r_norm2 < self.ps.particle_radius * self.ps.particle_radius:
            coef = (1 - self.ps.particle_radius / r_norm) / r_norm2
            ret += coef * r

    """
    Below are main functions
    """

    @ti.kernel
    def t_ComputeDensity(self):
        epsilon = 1e-5
        for part_idx in self.ps.particles:
            density = 0.0
            self.neighbor_search.t_ForEachNeighbors(part_idx, self.t_ComputeDensityTask, density)
            density = max(density, epsilon)
            self.ps.particles[part_idx].density = density

    @ti.kernel
    def t_ComputeViscosityForce(self):
        for part_idx in self.ps.particles:
            self.neighbor_search.t_ForEachNeighbors(part_idx,
                                                    self.t_ComputeViscosityTask,
                                                    self.ps.viscosity_force[part_idx])

    @ti.kernel
    def t_ComputeParticleCollision(self):
        for part_idx in self.ps.particles:
            self.neighbor_search.t_ForEachNeighbors(part_idx,
                                                    self.t_ComputeParticleCollisionTask,
                                                    self.ps.collision_force[part_idx])


    def p_Update(self):
        self.ps.t_Clear()
        self.neighbor_search.p_Update()
        self.t_ComputeDensity()
        self.t_ComputeViscosityForce()
        # self.t_ComputeParticleCollision()
