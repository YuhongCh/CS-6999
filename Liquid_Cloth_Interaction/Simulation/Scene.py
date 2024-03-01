from Liquid_Cloth_Interaction.Simulation.XMLParser import XMLParser
from Liquid_Cloth_Interaction.Simulation.DER.Strand import DER_Strand


class Scene:
    def __init__(self, filename: str):
        self.parser = XMLParser(filename)
        self.elastic = self.parser.p_load_elastic_parameters()
        self.integrator = self.parser.p_load_integrator()
        self.DER_model = DER_Strand(init_vertices=self.parser.p_load_der_vertices(),
                                    elastic=self.elastic,
                                    integrator=self.integrator)

    def p_init(self):
        """ Do everything that needs be precomputed for the scene here """
        self.DER_model.p_init()

    def p_next_step(self):
        self.DER_model.p_accumulate_force()
        self.DER_model.p_update_state()

