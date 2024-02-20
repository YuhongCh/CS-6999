import taichi as ti
import taichi.math as tm

from Liquid_Cloth_Interaction.Simulation.ElasticParameters import ElasticParameters
from Liquid_Cloth_Interaction.Simulation.DER.Components import DER_Vertices
from Liquid_Cloth_Interaction.Simulation.DER.States import DER_StrandState, DER_RestState

from Liquid_Cloth_Interaction.Simulation.GravityForce import GravityForce


@ti.data_oriented
class DER_Strand:
    def __init__(self, init_vertices: DER_Vertices, elastic: ElasticParameters):
        # set up initial DER parameters
        self.elastic = elastic
        self.state = DER_StrandState(init_vertices)
        self.rest_state = DER_RestState(self.state)

        # set up force related variables
        num_vertices = self.state.p_get_num_vertices()
        self.force = ti.ndarray(dtype=tm.vec3, shape=num_vertices)
        self.force_types = [GravityForce()]

    """
    Below are Python-Scope Methods
    """
    def p_init(self):
        """ Initializes the parameters based on declared value from __init__ method """
        self.state.p_init_state()
        self.rest_state.p_init_state()

    def p_accumulate_force(self):
        self.t_clear_force()
        for force_type in self.force_types:
            force_type.t_accumulate_force(self.force)

    def p_update_state(self, dt: float):
        self.t_update_state(dt)


    """
    Below are Taichi-Scope Methods
    """
    @ti.kernel
    def t_update_state(self, dt: float):
        """
        Update DER State based on previous computed forces
        ONLY call this method after all force computation completed
        """
        # TODO: Update this method to use semi-implicit integrator
        num_vertices = self.state.t_get_num_vertices()
        for i in ti.ndrange(num_vertices):
            self.state.vertices.velocity[i] += self.force[i] * dt
            self.state.vertices.position[i] += self.state.vertices.velocity[i] * dt

    @ti.kernel
    def t_clear_force(self):
        num_vertices = self.state.p_get_num_vertices()
        for i in ti.ndrange(num_vertices):
            self.force[i] = tm.vec3(0, 0, 0)
