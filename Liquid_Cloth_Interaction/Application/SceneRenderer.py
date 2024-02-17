import taichi as ti

from copy import copy

from Liquid_Cloth_Interaction.Simulation.Scene import Scene
from Liquid_Cloth_Interaction.Simulation.DER.States import DER_State

@ti.data_oriented
class SceneRenderer:
    def __init__(self, sim_scene: Scene):
        # below are simulation scene variables
        self.sim_scene = sim_scene
        self.der_renderer = DER_Renderer(sim_scene.DER_model)

        # below are elements to create scene on screen
        self.window = ti.ui.Window("DER Hair Simulation DEMO", (1024, 1024), vsync=True)
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((1, 1, 1))
        self.render_scene = self.window.get_scene()

        self.camera = ti.ui.Camera()
        self.camera.position(0.0, 0.0, 3)
        self.camera.lookat(0.0, 0.0, 0)

        self.render_scene.set_camera(self.camera)
        self.render_scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        self.render_scene.ambient_light((0.5, 0.5, 0.5))

    def render(self):
        self.der_renderer.render(self.render_scene)
        self.canvas.scene(self.render_scene)
        self.window.show()


@ti.data_oriented
class DER_Renderer:
    """ This is a temporary class to display the DER model on screen """
    def __init__(self, state: DER_State):
        num_vertices = state.get_num_vertices()
        self.state = copy(state)
        self.hair_position = ti.field(dtype=ti.math.vec3, shape=num_vertices)
        self.hair_indices = ti.field(dtype=int, shape=2 * (num_vertices - 1))
        self.load_hair_indices()

    @ti.kernel
    def load_hair_position(self):
        for i in self.state.vertices.position:
            self.hair_position[i] = self.state.vertices.position[i]

    def load_hair_indices(self):
        num_node = self.hair_indices.shape[0] // 2 + 1
        self.hair_indices[0] = 0
        self.hair_indices[2 * num_node - 3] = num_node - 1
        for i in ti.ndrange((1, num_node - 1)):
            self.hair_indices[2 * i] = i
            self.hair_indices[2 * i - 1] = i

    def render(self, scene_renderer: SceneRenderer):
        self.load_hair_position()
        scene_renderer.scene.lines(vertices=self.hair_position,
                                   width=self.hair_radius,
                                   indices=self.hair_indices)