import taichi as ti

from Liquid_Cloth_Interaction.Application.SceneRenderer import SceneRenderer
from Liquid_Cloth_Interaction.Simulation.Scene import Scene

ti.init(arch=ti.cpu, debug=True)


if __name__ == "__main__":
    sim_scene = Scene("../Test/simple_yarn.xml")
    sim_scene.p_init()

    sceneRenderer = SceneRenderer(sim_scene)
    while sceneRenderer.window.running:

        sceneRenderer.p_render()
