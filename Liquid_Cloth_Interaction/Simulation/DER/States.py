import taichi as ti
import taichi.math as tm

from copy import copy

from Liquid_Cloth_Interaction.Simulation.ElasticParameters import ElasticParameters
from Liquid_Cloth_Interaction.Simulation.DER.Components import (DER_Vertices,
                                                                DER_Edges,
                                                                DER_CurvatureBinormal,
                                                                DER_ReferenceFrame,
                                                                DER_MaterialFrame,
                                                                DER_Curvature)


@ti.data_oriented
class DER_StrandState:
    """ Represent the state of a single DER strand """
    def __init__(self, init_vertices: DER_Vertices):
        self.vertices = init_vertices
        self.edges = DER_Edges(self.vertices)
        self.kb = DER_CurvatureBinormal(self.edges)
        self.ref_frame = DER_ReferenceFrame(self.edges)
        self.mat_frame = DER_MaterialFrame(self.ref_frame)
        self.curvature = DER_Curvature(self.kb, self.mat_frame)

    """
    Below are Python-Scope Methods
    """
    def p_get_num_vertices(self) -> int:
        return self.vertices.p_get_num_vertices()

    def p_get_num_edges(self) -> int:
        return self.edges.p_get_num_edges()

    def p_init_state(self):
        self.t_init_state()
        self.t_update_state()

    """
    Below are Taichi-Scope Methods
    """
    @ti.func
    def t_get_num_vertices(self) -> int:
        return self.vertices.t_get_num_vertices()

    @ti.func
    def t_get_num_edges(self) -> int:
        return self.edges.t_get_num_edges()

    @ti.kernel
    def t_init_state(self):
        self.ref_frame.t_set_start_frame(tm.vec3(0, 0, 0))

    @ti.kernel
    def t_update_state(self):
        """ Update strand state based on updated DER vertices """
        self.edges.t_update()
        self.kb.t_update()
        self.ref_frame.t_update_frame()
        self.ref_frame.t_update_twist()
        self.mat_frame.t_update()
        self.curvature.t_update()


@ti.data_oriented
class DER_RestState:
    """ Represent the rest state of a given DER Strand """
    def __init__(self, start_state: DER_StrandState, damping: float = 0.0):
        self.state = copy(start_state)
        self.damping = damping
        self.tot_rest_length = 0.0

        num_edges = start_state.p_get_num_edges()
        num_vertices = start_state.p_get_num_vertices()
        self.rest_length = ti.ndarray(dtype=float, shape=num_edges)
        self.rest_kappa = ti.ndarray(dtype=tm.vec4, shape=num_edges)
        self.rest_twist = ti.ndarray(dtype=float, shape=num_edges)
        self.voronoi_length = ti.ndarray(dtype=float, shape=num_vertices)
        self.vertex_mass = ti.ndarray(dtype=float, shape=num_vertices)

    def p_init_state(self):
        self.t_update_state()

    @ti.kernel
    def t_update_state(self):
        """
        Compute the rest state variable value based on given start_state
        ONLY call this method at start of simulation
        """
        num_edges = self.state.t_get_num_edges()
        num_vertices = self.state.t_get_num_vertices()

        # update the rest edge length
        self.tot_rest_length = 0.0
        for i in ti.ndrange(num_edges):
            self.rest_length[i] = ((1.0 - self.damping) * self.state.edges.length[i] +
                                   self.damping * self.rest_length[i])
            self.tot_edge_length += self.rest_length[i]

        # update the voronoi length and vertex mass
        self.voronoi_length[0] = 0.5 * self.rest_length[0]
        for idx in ti.ndrange((1, num_edges)):  # (0, num_vertices-1)
            i = idx[0]
            self.voronoi_length[i] = 0.5 * (self.rest_length[i-1] + self.rest_length[i])
        self.voronoi_length[num_edges] = 0.5 * self.rest_length[num_edges - 1]

            # TODO: Change this mass calculation to more physic based approach
        for i in ti.ndrange(num_vertices):
            self.vertex_mass[i] = 1

        #update the rest curvature and twist
        # for i in ti.ndrange(num_edges):
        #     self.rest_kappa[i] = ((1.0 - self.damping) * self.state.curvature.kappa[i] +
        #                           self.damping * self.rest_kappa[i])
        #     self.rest_twist[i] = ((1.0 - self.damping) * self.state.curvature.kappa[i] +
        #                           self.damping * self.rest_kappa[i])




