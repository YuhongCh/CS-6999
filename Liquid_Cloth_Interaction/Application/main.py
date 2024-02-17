import taichi as ti

from Liquid_Cloth_Interaction.Application.SceneRenderer import SceneRenderer
from Liquid_Cloth_Interaction.Simulation.Scene import Scene

ti.init(arch=ti.cpu, debug=True)


if __name__ == "__main__":
    scene = Scene("../Test/simple_yarn.xml")
    sceneRenderer = SceneRenderer(scene)
    while sceneRenderer.window.running:

        sceneRenderer.render()
