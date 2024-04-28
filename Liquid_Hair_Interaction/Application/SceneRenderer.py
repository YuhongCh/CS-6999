import taichi as ti

from copy import copy

from Liquid_Hair_Interaction.Simulation.Scene import Scene
from Liquid_Hair_Interaction.Simulation.DER.StrandForce import StrandState

@ti.data_oriented
class SceneRenderer:
    def __init__(self, sim_scene: Scene):
        # below are simulation scene variables
        self.sim_scene = copy(sim_scene)
        self.der_renderer = DER_Renderer(sim_scene.DER_model)

        # below are elements to create scene on screen
        self.window = ti.ui.Window("DER Hair Simulation DEMO", (1024, 1024), vsync=True)
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((1, 1, 1))
        self.render_scene = self.window.get_scene()

        self.camera = ti.ui.Camera()
        self.camera.position(0.0, 0.0, 20)
        self.camera.lookat(0.0, 0.0, 0)

        self.render_scene.set_camera(self.camera)
        self.render_scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        self.render_scene.ambient_light((0.5, 0.5, 0.5))

    def p_render(self):
        self.der_renderer.p_render(self.render_scene)
        self.canvas.scene(self.render_scene)
        self.window.show()


@ti.data_oriented
class DER_Renderer:
    """ This is a temporary class to display the DER model on screen """
    def __init__(self, strand_state: StrandState):
        self.strand_state = copy(strand_state)

    def p_render(self, render_scene: ti.ui.Scene):
        render_scene.lines(vertices=self.strand_state.dofs.vertices, width=1)