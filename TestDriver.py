import taichi as ti
import taichi.math as tm

from Liquid_Hair_Interaction.Simulation.SPH.ParticleSystem import ParticleSystem
from Liquid_Hair_Interaction.Simulation.SPH.BaseSPH import BaseSPH
from Liquid_Hair_Interaction.Simulation.SPH.WCSPH import WCSPH
from Liquid_Hair_Interaction.Simulation.SPH.SPHConfig import SPHConfig
from Liquid_Hair_Interaction.Simulation.SPH.TestRenderer import SceneRenderer
from Liquid_Hair_Interaction.Simulation.SPH.NeighborSearch import UniformGridNeighborSearch


DEBUG = False

if DEBUG == True:
    ti.init(arch=ti.cpu, debug=True, cpu_max_num_threads=1, advanced_optimization=False)
else:
    ti.init(arch=ti.gpu)


if __name__ == "__main__":
    config = SPHConfig(
        num_particle = 100,
        particle_radius = 0.05,
        particle_mass=1,
        support_radius=0.2,
        viscosity=0.1,
        stiffness=1000,
        ref_density=1,
        resti_coeff=0.9,

        # scene configuration
        start_domain=tm.vec3(-1,-1,-1),
        end_domain=tm.vec3(1, 1, 1),
        grid_size=0.06,
        max_neighbor_count=100,

        # user choice methods
        kernel_type=2
    )

    ps = ParticleSystem(config)
    sph = WCSPH(config, ps)
    ps.t_Initialize(tm.vec3(0, 0, 0))

    sceneRenderer = SceneRenderer(ps)
    while sceneRenderer.window.running:

        for i in range(1):
            sph.p_Update()
            # dt = ps.t_ComputeCFL()
            # print(dt)
            ps.t_AdvanceTimeStep(0.001)
            ps.t_ComputeBoundaryCollision()

        sceneRenderer.p_render()
