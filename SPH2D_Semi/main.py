import taichi as ti
import taichi.math as tm

from Config import SPHConfig
from Renderer import SceneRenderer
from ParticleSystem import ParticleSystem, ParticleType
from NeighborSearch import NeighborSearch
from WCSPH import WCSPH

DEBUG = False
if DEBUG:
    ti.init(arch=ti.cpu, debug=DEBUG)
else:
    ti.init(arch=ti.gpu, debug=DEBUG)

config = SPHConfig()
config.default_init()
config.dump()

particle_system = ParticleSystem(config)

particle_system.generate_particles(start_domain=tm.vec2(0.2, 0.5),
                                   end_domain=tm.vec2(0.8, 0.9),
                                   particle_type=ParticleType.Fluid)

print(f"generated total of {particle_system.get_particle_num(-1)} particles")

neighbor_search = NeighborSearch(config, particle_system)
model = WCSPH(config, particle_system, neighbor_search)

renderer = SceneRenderer(particle_system)

while renderer.window.running:
    neighbor_search.compute_update()
    model.simulate()
    renderer.render()