import taichi as ti
import taichi.math as tm

from copy import copy

from Liquid_Cloth_Interaction.Simulation.Scene import Scene
from Liquid_Cloth_Interaction.Simulation.DER.Strand import DER_Strand

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
    def __init__(self, strand: DER_Strand):
        num_vertices = strand.state.p_get_num_vertices()
        self.state = copy(strand.state)
        self.hair_position = ti.field(dtype=ti.math.vec3, shape=num_vertices)
        self.hair_indices = ti.field(dtype=int, shape=2 * (num_vertices - 1))
        self.p_load_hair_indices()

    @ti.kernel
    def t_load_hair_position(self, ):
        for i in self.state.vertices.position:
            self.hair_position[i] = self.state.vertices.position[i]

    def p_load_hair_indices(self):
        num_node = self.hair_indices.shape[0] // 2 + 1
        self.hair_indices[0] = 0
        self.hair_indices[2 * num_node - 3] = num_node - 1
        for idx in ti.ndrange((1, num_node - 1)):
            i = idx[0]
            self.hair_indices[2 * i] = i
            self.hair_indices[2 * i - 1] = i

    def p_render(self, render_scene: ti.ui.Scene):
        self.t_load_hair_position()
        render_scene.lines(vertices=self.hair_position,
                                   width=1,
                                   indices=self.hair_indices)