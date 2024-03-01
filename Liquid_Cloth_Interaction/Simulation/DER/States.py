import taichi as ti
import taichi.math as tm

from copy import copy

from Liquid_Cloth_Interaction.Simulation.DER.Components import (DER_Vertices,
                                                                DER_Edges,
                                                                DER_CurvatureBinormal,
                                                                DER_ReferenceFrame,
                                                                DER_Twist,
                                                                DER_MaterialFrame,
                                                                DER_Curvature,
                                                                DER_GradTwist,
                                                                DER_GradCurvature)


@ti.data_oriented
class DER_StrandState:
    """ Represent the state of a single DER strand """
    def __init__(self, init_vertices: DER_Vertices):
        self.vertices = init_vertices
        self.edges = DER_Edges(self.vertices)
        self.kb = DER_CurvatureBinormal(self.edges)
        self.ref_frame = DER_ReferenceFrame(self.edges)
        self.twist = DER_Twist(self.vertices, self.ref_frame)
        self.mat_frame = DER_MaterialFrame(self.ref_frame)
        self.curvature = DER_Curvature(self.kb, self.mat_frame)
        self.grad_twist = DER_GradTwist(self.edges, self.kb)
        self.grad_curvature = DER_GradCurvature(self.edges, self.kb, self.mat_frame, self.curvature)

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
        self.edges.t_update()
        self.ref_frame.t_set_start_frame()

    @ti.kernel
    def t_update_state(self):
        """ Update strand state based on updated DER vertices """
        self.edges.t_update()
        self.kb.t_update()
        self.ref_frame.t_update_frame()
        self.ref_frame.t_update_twist()
        self.twist.t_update()
        self.mat_frame.t_update()
        self.curvature.t_update()
        self.grad_twist.t_update()
        self.grad_curvature.t_update()


@ti.data_oriented
class DER_RestState:
    """ Represent the rest state of a given DER Strand """
    def __init__(self, start_state: DER_StrandState, damping: float = 0.0):
        self.state = copy(start_state)
        self.damping = damping
        self.tot_rest_length = 0.0

        num_edges = start_state.p_get_num_edges()
        num_vertices = start_state.p_get_num_vertices()
        self.rest_length = ti.field(dtype=float, shape=num_edges)
        self.rest_kappa = ti.field(dtype=tm.vec4, shape=num_edges-1)
        self.rest_twist = ti.field(dtype=float, shape=num_edges)
        self.voronoi_length = ti.field(dtype=float, shape=num_vertices)
        self.inv_voronoi_length = ti.field(dtype=float, shape=num_vertices)
        self.vertex_mass = ti.field(dtype=float, shape=num_vertices)

    def p_init_state(self):
        self.p_update_state()

    def p_update_state(self):
        """
        Compute the rest state variable value based on given start_state
        ONLY call this method at start of simulation
        """
        self.tot_rest_length = self.t_update_edge_length()
        self.t_update_state()

    @ti.kernel
    def t_update_state(self):
        """
        Compute the rest state variable value based on given start_state
        ONLY call this method at start of simulation
        """
        num_vertices = self.state.t_get_num_vertices()

        # TODO: Change this mass calculation to more physic based approach
        for i in ti.ndrange(num_vertices):
            self.vertex_mass[i] = 1

        # update the rest curvature and twist
        num_edges = num_vertices - 1
        for i in ti.ndrange(num_edges):
            self.rest_twist[i] = ((1.0 - self.damping) * self.state.twist.twist[i] +
                                  self.damping * self.rest_twist[i])
        for i in ti.ndrange(num_edges - 1):
            self.rest_kappa[i] = ((1.0 - self.damping) * self.state.curvature.kappa[i] +
                                  self.damping * self.rest_kappa[i])

    @ti.kernel
    def t_update_edge_length(self) -> float:
        """
        Compute edge length related variable value based on given start state
        @return the total edge length
        """
        num_edges = self.state.t_get_num_edges()
        # traditional definition of edge length
        tot_edge_length = 0.0
        for i in ti.ndrange(num_edges):
            self.rest_length[i] = ((1.0 - self.damping) * self.state.edges.length[i] +
                                   self.damping * self.rest_length[i])
            tot_edge_length += self.rest_length[i]

        # voronoi edge length
        self.voronoi_length[0] = 0.5 * self.rest_length[0]
        for i in ti.ndrange((1, num_edges)):  # (0, num_vertices-1)
            self.voronoi_length[i] = 0.5 * (self.rest_length[i - 1] + self.rest_length[i])
            self.inv_voronoi_length[i] = 1.0 / self.voronoi_length[i]
        self.voronoi_length[num_edges] = 0.5 * self.rest_length[num_edges - 1]
        self.inv_voronoi_length[num_edges] = 1.0 / self.voronoi_length[num_edges]

        return tot_edge_length




