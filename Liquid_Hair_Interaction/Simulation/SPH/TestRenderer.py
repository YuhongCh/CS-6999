import taichi as ti
import taichi.math as tm

from Liquid_Hair_Interaction.Simulation.SPH.ParticleSystem import ParticleSystem


@ti.data_oriented
class SceneRenderer:
    def __init__(self, ps: ParticleSystem):
        # below are simulation scene variables
        self.sph_renderer = SPH_Renderer(ps)

        # below are elements to create scene on screen
        self.window = ti.ui.Window("SPH Simulation DEMO", (1024, 1024), vsync=True)
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

    def p_render(self):
        self.sph_renderer.p_render(self.render_scene)

        self.camera.track_user_inputs(self.window, movement_speed=0.1, hold_key=ti.ui.LMB)
        self.render_scene.set_camera(self.camera)
        self.canvas.scene(self.render_scene)
        self.window.show()


@ti.data_oriented
class SPH_Renderer:
    """ This is a temporary class to display the DER model on screen """
    def __init__(self, ps: ParticleSystem):
        self.ps = ps
        self.position = ti.field(dtype=ti.math.vec3, shape=self.ps.num_particle)

    @ti.kernel
    def t_update_position(self):
        for part_idx in self.ps.particles:
            self.position[part_idx] = self.ps.particles[part_idx].x

    def p_render(self, render_scene: ti.ui.Scene):
        self.t_update_position()
        render_scene.particles(self.position, radius=0.01, color=(0.5, 0.42, 0.8))