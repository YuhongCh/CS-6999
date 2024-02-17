from Liquid_Cloth_Interaction.Simulation.XMLParser import XMLParser
from Liquid_Cloth_Interaction.Simulation.DER.States import DER_State


class Scene:
    def __init__(self, filename: str):
        self.parser = XMLParser(filename)
        self.elastic = self.parser.load_elastic_parameters()
        self.integrator = self.parser.load_integrator()
        self.DER_model = DER_State(init_vertices=self.parser.load_der_vertices(),
                                   elastic=self.elastic)

    def next_step(self):
        pass

