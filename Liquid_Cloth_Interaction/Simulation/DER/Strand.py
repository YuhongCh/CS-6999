import taichi as ti
import taichi.math as tm

from copy import copy

from Liquid_Cloth_Interaction.Simulation.Integrator import Integrator
from Liquid_Cloth_Interaction.Simulation.ElasticParameters import ElasticParameters

from Liquid_Cloth_Interaction.Simulation.DER.Components import DER_Vertices
from Liquid_Cloth_Interaction.Simulation.DER.States import DER_StrandState, DER_RestState

from Liquid_Cloth_Interaction.Simulation.GravityForce import GravityForce
from Liquid_Cloth_Interaction.Simulation.DER.StretchForce import StretchForce
from Liquid_Cloth_Interaction.Simulation.DER.BendingForce import BendingForce
from Liquid_Cloth_Interaction.Simulation.DER.TwistForce import TwistForce


@ti.data_oriented
class DER_Strand:
    def __init__(self, init_vertices: DER_Vertices, elastic: ElasticParameters, integrator: Integrator):
        # set up initial DER parameters
        self.elastic = elastic
        self.integrator = copy(integrator)
        self.state = DER_StrandState(init_vertices)
        self.rest_state = DER_RestState(self.state)

        # set up force related variables
        num_vertices = self.state.p_get_num_vertices()
        self.force = ti.ndarray(dtype=tm.vec4, shape=num_vertices)
        self.velocity = ti.ndarray(dtype=tm.vec4, shape=num_vertices)

        self.force_types = [GravityForce(),
                            StretchForce(self.state, self.rest_state, self.elastic),
                            BendingForce(self.state, self.rest_state, self.elastic),
                            TwistForce(self.state, self.rest_state, self.elastic)]

    """
    Below are Python-Scope Methods
    """
    def p_init(self):
        """ Initializes the parameters based on declared value from __init__ method """
        self.state.p_init_state()
        self.rest_state.p_init_state()

    def p_accumulate_force(self):
        self.t_clear_force(self.force)
        for force_type in self.force_types:
            force_type.t_accumulate_force(self.force)

    def p_update_state(self):
        self.t_update_state(self.force, self.velocity)
        self.state.t_update_state()


    """
    Below are Taichi-Scope Methods
    """
    @ti.kernel
    def t_update_state(self, force: ti.types.ndarray(dtype=tm.vec4),
                       velocity: ti.types.ndarray(dtype=tm.vec3)):
        """
        Update DER State based on previous computed forces
        ONLY call this method after all force computation completed
        """
        num_vertices = self.state.t_get_num_vertices()
        for i in ti.ndrange(num_vertices):
            if self.state.vertices.fixed[i]:
                force[i] = tm.vec4(0, 0, 0, 0)

        # TODO: Update this method to use semi-implicit integrator
        self.integrator.integrate(force, velocity)

        num_vertices = self.state.t_get_num_vertices()
        for i in ti.ndrange(num_vertices):
            self.state.vertices.position[i] += velocity[i].xyz * self.integrator.dt
            self.state.vertices.theta[i] += velocity[i].w * self.integrator.dt

    @ti.kernel
    def t_clear_force(self, force: ti.types.ndarray(dtype=tm.vec4)):
        num_vertices = self.state.t_get_num_vertices()
        for i in ti.ndrange(num_vertices):
            force[i] = tm.vec4(0, 0, 0, 0)
