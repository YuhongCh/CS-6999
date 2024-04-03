"""
This file should ONLY contain the Scene related parameter and functions
"""

from Liquid_Cloth_Interaction.Simulation.XMLParser import XMLParser
from Liquid_Cloth_Interaction.Simulation.Parameters import LiquidParameter
from Liquid_Cloth_Interaction.Simulation.DER.Strand import DER_Strand


"""
Scene contains variables:
int step_count;
  VectorXs m_x;       // particle pos
  VectorXs m_rest_x;  // particle rest pos

  VectorXs m_v;  // particle velocity
  VectorXs m_saved_v;
  VectorXs m_dv;
  VectorXs m_fluid_v;  // fluid velocity
  VectorXs m_m;        // particle mass
  VectorXs m_fluid_m;
  VectorXs m_radius;  // particle radius
  VectorXs m_vol;     // particle volume
  VectorXs m_rest_vol;
  VectorXs m_shape_factor;
  VectorXs m_fluid_vol;
  VectorXuc m_inside;
  VectorXs m_volume_fraction;  // elastic particle fraction
  VectorXs m_rest_volume_fraction;
  VectorXs m_orientation;
  std::vector<VectorXs> m_div;
  std::vector<VectorXs> m_sphere_pattern;
  std::vector<ParticleClassifier> m_classifier;
"""
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

