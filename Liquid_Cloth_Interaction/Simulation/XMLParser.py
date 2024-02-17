import taichi.math as tm

import xml.etree.ElementTree as ET
from ast import literal_eval

from Liquid_Cloth_Interaction.Simulation.ElasticParameters import ElasticParameters
from Liquid_Cloth_Interaction.Simulation.Integrator import Integrator
from Liquid_Cloth_Interaction.Simulation.DER.Components import DER_Vertices


@ti.data_oriented
class XMLParser:
    def __init__(self, filename: str):
        self.root = ET.parse(filename).getroot()

    def load_elastic_parameters(self) -> ElasticParameters:
        """ Read and parse the Elastic Parameter tag from the XML file """
        parameters = self.root.find('ElasticParameters')
        if parameters is None:
            print("Error: Failed to find elastic parameters")
            return None
        radius = float(parameters.find("radius").get("value"))
        youngs_modulus = literal_eval(parameters.find("youngsModulus").get("value"))
        poisson_ratio = float(parameters.find("poissonRatio").get("value"))
        collision_multiplier = float(parameters.find("collisionMultiplier").get("value"))
        attach_multiplier = float(parameters.find("attachMultiplier").get("value"))
        density = float(parameters.find("density").get("value"))
        viscosity = literal_eval(parameters.find("viscosity").get("value"))
        base_rotation = float(parameters.find("baseRotation").get("value"))
        return ElasticParameters(radius=radius,
                                 youngs_modulus=youngs_modulus,
                                 poisson_ratio=poisson_ratio,
                                 collision_multiplier=collision_multiplier,
                                 attach_multiplier=attach_multiplier,
                                 density=density,
                                 viscosity=viscosity,
                                 base_rotation=base_rotation)

    def load_integrator(self) -> Integrator:
        parameters = self.root.find('integrator')
        if parameters is None:
            print("Error: Failed to find integrator parameter")
            return None
        dt = float(parameters.get("dt"))
        type = parameters.get("type")
        criterion = float(parameters.get("criterion"))
        return Integrator(dt=dt, type=type, criterion=criterion)

    def load_der_vertices(self) -> DER_Vertices:
        particles = self.root.findall('particle')
        vertices = DER_Vertices(len(particles))
        for i in range(len(particles)):
            particle = particles[i]
            position = particle.get('x')
            velocity = particle.get('v')
            fixed = particle.get('fixed')

            position = position.split(' ')
            position = tm.vec3(float(position[0]), float(position[1]), float(position[2]))
            velocity = velocity.split(' ')
            velocity = tm.vec3(float(velocity[0]), float(velocity[1]), float(velocity[2]))
            fixed = int(fixed)

            vertices.position[i] = position
            vertices.velocity[i] = velocity
            vertices.fixed[i] = fixed
        return vertices

