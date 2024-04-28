"""
This file should ONLY contain the Scene related parameter and functions
"""

import taichi as ti

from Liquid_Hair_Interaction.Application.XMLParser import XMLParser
from Liquid_Hair_Interaction.Simulation.Parameters import LiquidParameter
from Liquid_Hair_Interaction.Simulation.DER.StrandForce import StrandState, StrandForce


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
        self.elastic = self.parser.load_elastic_parameters()
        self.integrator = self.parser.load_integrator()

        vertices = self.parser.load_der_vertices()
        degrees = ti.field(dtype=float, shape=vertices.shape[0])
        self.Hair = StrandForce(vertices, degrees)

    def init(self):
        """ Do everything that needs be precomputed for the scene here """


    def next_step(self):
        self.Hair.accumulateForce()
        self.Hair.updateStrandState()
        self.Hair.clearForce()

