import taichi as ti
import taichi.math as tm

from copy import copy
from Liquid_Cloth_Interaction.Simulation.SimulationUtils import t_find_normal
from Liquid_Cloth_Interaction.Simulation.SimulationUtils import Epsilon

"""
Here is the idea:
Assume there are n vertices, then there are n - 1 edges
Then there are n - 2 curvature and grad_curvature since they are defined between edges
"""


@ti.data_oriented
class DER_Vertices:
    """ Basic DER Vertices class """
    def __init__(self, num_vertices: int):
        self.position = ti.field(dtype=tm.vec3, shape=num_vertices)
        self.theta = ti.field(dtype=float, shape=num_vertices)
        self.fixed = ti.field(dtype=bool, shape=num_vertices)

    def p_get_num_vertices(self) -> int:
        return self.position.shape[0]

    @ti.func
    def t_get_num_vertices(self) -> int:
        return self.position.shape[0]


@ti.data_oriented
class DER_Edges:
    """ DER edges connecting DER vertices """
    def __init__(self, vertices: DER_Vertices):
        num_edges = vertices.p_get_num_vertices() - 1
        self.dir = ti.field(dtype=tm.vec3, shape=num_edges)
        self.length = ti.field(dtype=float, shape=num_edges)
        self.vertices = copy(vertices)

    def p_get_num_edges(self) -> int:
        return self.dir.shape[0]

    @ti.func
    def t_get_num_edges(self) -> int:
        return self.dir.shape[0]

    @ti.func
    def t_update(self):
        """ Update DER edges direction and length based on given vertices """
        num_edges = self.t_get_num_edges()
        for i in ti.ndrange(num_edges):
            edge = self.vertices.position[i+1] - self.vertices.position[i]
            self.length[i] = edge.norm()
            self.dir[i] = edge / self.length[i]


@ti.data_oriented
class DER_CurvatureBinormal:
    """
    DER Curvature Binormal, defined between two DER edges ONLY
    Hence, It ONLY exists at non-boundary DER vertices
    """
    def __init__(self, edges: DER_Edges):
        num_edges = edges.p_get_num_edges()
        self.kb = ti.field(dtype=tm.vec3, shape=num_edges)
        self.edges = copy(edges)

    @ti.func
    def t_update(self):
        num_edges = self.edges.t_get_num_edges()
        for i in ti.ndrange((1, num_edges)):
            prev_dir = self.edges.dir[i-1]
            curr_dir = self.edges.dir[i]
            denominator = 1 + tm.dot(prev_dir, curr_dir)
            if ti.abs(denominator) < Epsilon:
                print("INVALID denominator")
            self.kb[i] = 2.0 * tm.cross(prev_dir, curr_dir) / denominator


@ti.data_oriented
class DER_ReferenceFrame:
    """
    This class contains reference frame and reference twist angle
    DER Reference Frame is defined on every DER Edges to indicate its twist condition
    DER Reference Twist is defined on every reference frame to indicate how reference frame changes over vertices
    """
    def __init__(self, edges: DER_Edges):
        num_edges = edges.p_get_num_edges()
        self.ref_frame1 = ti.field(dtype=tm.vec3, shape=num_edges)
        self.ref_frame2 = ti.field(dtype=tm.vec3, shape=num_edges)
        self.ref_twist = ti.field(dtype=float, shape=num_edges)
        self.edges = copy(edges)

    @ti.func
    def t_set_start_frame(self):
        """ Set up the initial reference frame if there is any """
        # determine the initial reference frame
        num_edges = self.edges.t_get_num_edges()
        start_edge_dir = self.edges.dir[0]
        rf1 = t_find_normal(start_edge_dir)
        self.ref_frame1[0] = rf1
        self.ref_frame2[0] = tm.cross(start_edge_dir, rf1)

        # spatial parallel transport the initial reference frame
        for j in ti.ndrange(1):
            for i in ti.ndrange((1, num_edges)):
                prev_ref_frame1 = self.ref_frame1[i - 1]
                prev_dir = self.edges.dir[i - 1]
                curr_dir = self.edges.dir[i]

                b = tm.cross(prev_dir, curr_dir)
                res_dir = prev_ref_frame1
                if b.norm() > Epsilon:
                    b.normalized()
                    n0 = tm.cross(prev_dir, b)
                    n1 = tm.cross(curr_dir, b)
                    res_dir = tm.dot(prev_ref_frame1, n0) * n1 + tm.dot(prev_ref_frame1, b) * b

                self.ref_frame1[i] = res_dir
                self.ref_frame2[i] = tm.cross(curr_dir, res_dir)

    @ti.func
    def t_update_frame(self):
        """ update reference frame via time-parallel transport """
        num_edges = self.edges.t_get_num_edges()
        for i in ti.ndrange((1, num_edges)):
            prev_dir = self.edges.dir[i - 1]
            curr_dir = self.edges.dir[i]
            b = tm.cross(prev_dir, curr_dir)

            if b.norm() < Epsilon:
                continue

            # here we construct coordinate system
            # previous edge has axis (n0, b, prev_dir), curr edge has axis (n1, b, curr_dir)
            b.normalized()
            n0 = tm.cross(prev_dir, b)
            n1 = tm.cross(curr_dir, b)

            # parallel transport the reference frame from last time step to current time step
            ref_frame1 = tm.dot(self.ref_frame1[i], n0) * n1 + tm.dot(self.ref_frame1[i], b) * b
            ref_frame1 -= tm.dot(ref_frame1, curr_dir) * curr_dir
            ref_frame1.normalized()
            ref_frame2 = tm.cross(curr_dir, ref_frame1)
            ref_frame2.normalized()

            # assign back the result
            self.ref_frame1[i] = ref_frame1
            self.ref_frame2[i] = ref_frame2

    @ti.func
    def t_update_twist(self):
        """ Based on previous reference frame and current reference frame to calculate the twist angle between them """
        num_edges = self.edges.t_get_num_edges()

        # outside loop to prevent parallel
        for j in ti.ndrange(1):
            for i in ti.ndrange((1, num_edges)):
                prev_ref_frame1 = self.ref_frame1[i-1]
                curr_ref_frame1 = self.ref_frame1[i]
                prev_dir = self.edges.dir[i-1]
                curr_dir = self.edges.dir[i]

                b = tm.cross(prev_dir, curr_dir)
                res_dir = prev_ref_frame1
                if b.norm() > Epsilon:
                    b.normalized()
                    n0 = tm.cross(prev_dir, b)
                    n1 = tm.cross(curr_dir, b)
                    res_dir = tm.dot(prev_ref_frame1, n0) * n1 + tm.dot(prev_ref_frame1, b) * b

                cos_theta = tm.cos(self.ref_twist[i])
                sin_theta = tm.sin(self.ref_twist[i])
                res_dir = (cos_theta * res_dir + sin_theta * tm.cross(curr_dir, res_dir) +
                           (1.0 - cos_theta) * tm.dot(curr_dir, res_dir) * curr_dir)
                w = tm.cross(res_dir, curr_ref_frame1)
                angle = tm.atan2(w.norm(), tm.dot(res_dir, curr_ref_frame1))
                if tm.dot(curr_dir, w) < 0:
                    angle = -angle
                self.ref_twist[i] += angle


@ti.data_oriented
class DER_Twist:
    def __init__(self, vertices: DER_Vertices, ref_frame: DER_ReferenceFrame):
        num_edges = ref_frame.edges.p_get_num_edges()
        self.twist = ti.field(dtype=float, shape=num_edges)
        self.ref_frame = copy(ref_frame)
        self.vertices = copy(vertices)

    @ti.func
    def t_update(self):
        num_edges = self.ref_frame.edges.t_get_num_edges()
        for i in ti.ndrange(num_edges):
            self.twist[i] = self.ref_frame.ref_twist[i] + self.vertices.theta[i+1] - self.vertices.theta[i]


@ti.data_oriented
class DER_MaterialFrame:
    """ This class defines material frame of the edges, one-to-one relationship to reference frames """
    def __init__(self, ref_frame: DER_ReferenceFrame):
        num_edges = ref_frame.edges.p_get_num_edges()
        self.mat_frame1 = ti.field(dtype=tm.vec3, shape=num_edges)
        self.mat_frame2 = ti.field(dtype=tm.vec3, shape=num_edges)
        self.ref_frame = copy(ref_frame)

    @ti.func
    def t_update(self):
        num_edges = self.ref_frame.edges.t_get_num_edges()
        for i in ti.ndrange(num_edges):
            ref_frame1 = self.ref_frame.ref_frame1[i]
            ref_frame2 = self.ref_frame.ref_frame2[i]
            theta = self.ref_frame.edges.vertices.theta[i]
            sin_theta = tm.sin(theta)
            cos_theta = tm.cos(theta)
            self.mat_frame1[i] = cos_theta * ref_frame1 + sin_theta * ref_frame2
            self.mat_frame2[i] = -sin_theta * ref_frame1 + cos_theta * ref_frame2


@ti.data_oriented
class DER_Curvature:
    """
    This class defines curvature of hair vertices.
    Similar as Curvature Binormal, this class is defined between DER vertices
    """
    def __init__(self, kb: DER_CurvatureBinormal, mat_frame: DER_MaterialFrame):
        num_edges = kb.edges.p_get_num_edges()
        self.kappa = ti.field(dtype=tm.vec4, shape=num_edges-1)
        self.kb = copy(kb)
        self.mat_frame = copy(mat_frame)

    @ti.func
    def t_update(self):
        num_edges = self.kb.edges.t_get_num_edges()
        for i in ti.ndrange((1, num_edges-1)):
            curr_kb = self.kb.kb[i]
            prev_mat_frame1 = self.mat_frame.mat_frame1[i-1]
            prev_mat_frame2 = self.mat_frame.mat_frame2[i-1]
            curr_mat_frame1 = self.mat_frame.mat_frame1[i]
            curr_mat_frame2 = self.mat_frame.mat_frame2[i]

            kappa1 = tm.dot(curr_kb, prev_mat_frame2)
            kappa2 = -tm.dot(curr_kb, prev_mat_frame1)
            kappa3 = tm.dot(curr_kb, curr_mat_frame2)
            kappa4 = -tm.dot(curr_kb, curr_mat_frame1)
            self.kappa[i] = tm.vec4(kappa1, kappa2, kappa3, kappa4)


@ti.data_oriented
class DER_GradTwist:
    def __init__(self, edges: DER_Edges, kb: DER_CurvatureBinormal):
        num_edges = edges.p_get_num_edges()
        self.grad_twist = ti.Vector.field(11, dtype=float, shape=num_edges-1)
        self.edges = copy(edges)
        self.kb = copy(kb)

    @ti.func
    def t_update(self):
        num_edges = self.edges.t_get_num_edges()
        for i in ti.ndrange(num_edges-1):
            curr_kb = self.kb.kb[i+1]
            self.grad_twist[i][0:3] = -0.5 / self.edges.length[i] * curr_kb
            self.grad_twist[i][8:] = 0.5 / self.edges.length[i + 1] * curr_kb
            self.grad_twist[i][4:7] = -self.grad_twist[i][0:3] - self.grad_twist[i][8:]
            self.grad_twist[i][4] = -1
            self.grad_twist[i][7] = 1


@ti.data_oriented
class DER_GradCurvature:
    """ This class contains gradient of DER Curvature """
    def __init__(self, edges: DER_Edges, kb: DER_CurvatureBinormal,
                       mat_frame: DER_MaterialFrame, curvature: DER_Curvature):
        num_edges = edges.p_get_num_edges()
        self.grad_kappa = ti.Matrix.field(11, 4, dtype=float, shape=num_edges-1)
        self.edges = copy(edges)
        self.kb = copy(kb)
        self.mat_frame = copy(mat_frame)
        self.curvature = copy(curvature)

    @ti.func
    def t_update(self):
        # TODO: Check if the following indexing is correct
        # TODO: Complete the assignment of gradient kappa
        num_edges = self.edges.t_get_num_edges()
        for i in ti.ndrange((1, num_edges - 1)):
            prev_edge_dir = self.edges.dir[i-1]
            curr_edge_dir = self.edges.dir[i]
            prev_edge_length = self.edges.length[i-1]
            curr_edge_length = self.edges.length[i]
            prev_mat_frame1 = self.mat_frame.mat_frame1[i - 1]
            prev_mat_frame2 = self.mat_frame.mat_frame2[i - 1]
            curr_mat_frame1 = self.mat_frame.mat_frame1[i]
            curr_mat_frame2 = self.mat_frame.mat_frame2[i]
            kappa = self.curvature.kappa[i-1]

            chi = 1.0 + tm.dot(prev_edge_dir, curr_edge_dir)
            chi = max(chi, Epsilon)

            tilde_t = (prev_edge_dir + curr_edge_dir) / chi
            tilde_pmf1 = (2 * prev_mat_frame1) / chi
            tilde_pmf2 = (2 * prev_mat_frame2) / chi
            tilde_cmf1 = (2 * curr_mat_frame1) / chi
            tilde_cmf2 = (2 * curr_mat_frame2) / chi

            Dk0pDep = 1.0 / prev_edge_length * (-kappa[0] * tilde_t + tm.cross(curr_edge_dir, tilde_pmf2))
            Dk0pDec = 1.0 / curr_edge_length * (-kappa[0] * tilde_t - tm.cross(prev_edge_dir, tilde_pmf2))
            Dk1pDep = 1.0 / prev_edge_length * (-kappa[1] * tilde_t - tm.cross(curr_edge_dir, tilde_pmf1))
            Dk1pDec = 1.0 / curr_edge_length * (-kappa[1] * tilde_t + tm.cross(prev_edge_dir, tilde_pmf1))

            Dk0cDep = 1.0 / prev_edge_length * (-kappa[2] * tilde_t + tm.cross(curr_edge_dir, tilde_cmf2))
            Dk0cDec = 1.0 / curr_edge_length * (-kappa[2] * tilde_t - tm.cross(prev_edge_dir, tilde_cmf2))
            Dk1cDep = 1.0 / prev_edge_length * (-kappa[3] * tilde_t - tm.cross(curr_edge_dir, tilde_cmf1))
            Dk1cDec = 1.0 / curr_edge_length * (-kappa[3] * tilde_t + tm.cross(prev_edge_dir, tilde_cmf1))

            self.grad_kappa[i][0:3, 0] = -Dk0pDep
            self.grad_kappa[i][4:7, 0] = Dk0pDep - Dk0pDec
            self.grad_kappa[i][8:11,0] = Dk0pDec
            self.grad_kappa[i][0:3, 1] = -Dk1pDep
            self.grad_kappa[i][4:7, 1] = Dk1pDep - Dk1pDec
            self.grad_kappa[i][8:11,1] = Dk1pDec

            self.grad_kappa[i][0:3, 2] = -Dk0cDep
            self.grad_kappa[i][4:7, 2] = Dk0cDep - Dk0cDec
            self.grad_kappa[i][8:11,2] = Dk0cDec
            self.grad_kappa[i][0:3, 3] = -Dk1cDep
            self.grad_kappa[i][4:7, 3] = Dk1cDep - Dk1cDec
            self.grad_kappa[i][8:11,3] = Dk1cDec

            kb = self.kb.kb[i]
            self.grad_kappa[i][3, 0] = tm.dot(-kb, prev_mat_frame1)
            self.grad_kappa[i][7, 0] = 0
            self.grad_kappa[i][3, 1] = tm.dot(-kb, prev_mat_frame2)
            self.grad_kappa[i][7, 1] = 0
            self.grad_kappa[i][3, 2] = tm.dot(-kb, curr_mat_frame1)
            self.grad_kappa[i][7, 2] = 0
            self.grad_kappa[i][3, 3] = tm.dot(-kb, curr_mat_frame2)
            self.grad_kappa[i][7, 3] = 0
