import taichi as ti

from Liquid_Hair_Interaction.Application.SceneRenderer import SceneRenderer
from Liquid_Hair_Interaction.Simulation.Scene import Scene

ti.init(arch=ti.cpu, debug=True)


if __name__ == "__main__":
    sim_scene = Scene("../Test/simple_yarn.xml")
    sim_scene.init()

    sceneRenderer = SceneRenderer(sim_scene)
    while sceneRenderer.window.running:
        sim_scene.p_next_step()
        sceneRenderer.p_render()