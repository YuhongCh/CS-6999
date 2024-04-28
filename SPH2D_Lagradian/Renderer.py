import taichi as ti
from ParticleSystem import ParticleSystem, ParticleType

@ti.data_oriented
class SceneRenderer:
    def __init__(self, particle_system: ParticleSystem):
        # below are simulation scene variables
        self.ps = particle_system
        self.num_particle = self.ps.get_particle_num(ParticleType.NonEmpty)
        self.position_count = ti.field(int, shape=())
        self.position = ti.Vector.field(2, dtype=float, shape=self.ps.get_particle_num(ParticleType.Fluid))

        # below are elements to create scene on screen
        self.window = ti.ui.Window(name="SPH Simulation DEMO", res=(680, 680), vsync=True)
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((1, 1, 1))
        self.render_scene = self.window.get_scene()

        self.camera = ti.ui.Camera()
        self.camera.position(0.0, 0.0, 5)
        self.camera.lookat(0.0, 0.0, 0)
        self.camera.fov(60)

        self.render_scene.set_camera(self.camera)
        self.render_scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        self.render_scene.ambient_light((0.5, 0.5, 0.5))

    @ti.kernel
    def __update_position(self):
        for pid in range(self.num_particle):
            if self.ps.type[pid] == ParticleType.Fluid:
                idx = ti.atomic_add(self.position_count[None], 1)
                self.position[idx] = self.ps.position[pid]

    def update_position(self):
        self.position_count[None] = 0
        self.__update_position()
        assert(self.position_count[None] == self.ps.num_fluid_particle)

    def render(self):
        self.update_position()
        self.canvas.circles(self.position, radius=self.ps.particle_radius, color=(0.5, 0.42, 0.8))

        self.canvas.scene(self.render_scene)
        self.window.show()
