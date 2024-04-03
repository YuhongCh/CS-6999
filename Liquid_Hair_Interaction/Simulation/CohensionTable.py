import taichi as ti
import taichi.math as tm

import MathUtils

@ti.func
def t_getSpringFunc(x, dmin, k0, k1, x1, h) -> float:
   t2 = 1.0 / h
   t3 = x * x
   t4 = dmin + h
   t5 = h * h
   t6 = k1 * t5
   t7 = dmin * h * k1
   return -t2 - t2 * t3 * 1.0 / (t4 * t4) * (t6 + t7 - h * x1 * 3.0 - 3.0) + \
          t2 * t3 * 1.0 / (t4 * t4 * t4) * x * (t6 + t7 - h * x1 * 2.0 - 2.0)

@ti.func
def t_getGradSpringFunc(x, dmin, k0, k1, x1, h) -> float:
   t2 = 1.0 / h
   t3 = dmin + h
   t4 = h * h
   t5 = k1 * t4
   t6 = dmin * h * k1
   return t2 * 1.0 / (t3 * t3 * t3) * (x * x) * (t5 + t6 - h * x1 * 2.0 - 2.0) * 3.0 - \
          t2 * 1.0 / (t3 * t3) * x * (t5 + t6 - h * x1 * 3.0 - 3.0) * 2.0


@ti.data_oriented
class CohensionTable:

    ANGLE_EPSILON = 0.008

    def __init__(self, radius_multiplier, collision_stiffness, 
                       radius_multiplier_planar, collision_stiffness_planar, ):
        self.sigma = 72.0
        self.theta = MathUtils.PI
        self.radii = 0.004
        self.max_alpha = MathUtils.PI - self.theta
        self.max_d0 = 0.2
        self.min_d0 = 2 * self.radii
        self.min_d0_planar = self.radii
        self.radius_multiplier = radius_multiplier
        self.collision_stiffness = collision_stiffness
        self.radius_multiplier_planar = radius_multiplier_planar
        self.collision_stiffness_planar = collision_stiffness_planar
        self.discretization = 256

        n = 100
        self.alpha_table = ti.Matrix.field(n, n, dtype=float, shape=(self.discretization, self.discretization))
        self.A_table = ti.Matrix.field(n, n, dtype=float, shape=(self.discretization, self.discretization))
        self.dEdd_table = ti.Matrix.field(n, n, dtype=float, shape=(self.discretization, self.discretization))
        self.max_As = ti.Vector.field(n, dtype=float, shape=self.discretization)
        self.min_As = ti.Vector.field(n, dtype=float, shape=self.discretization)

        self.alpha_planar_table = ti.Matrix.field(n, n, dtype=float, shape=(self.discretization, self.discretization))
        self.A_planar_table = ti.Matrix.field(n, n, dtype=float, shape=(self.discretization, self.discretization))
        self.dEdd_planar_table = ti.Matrix.field(n, n, dtype=float, shape=(self.discretization, self.discretization))

    @ti.func
    def t_getInterpolateTable(self, A: float, d0: float, mat, dmin: float) -> float:
      """ mat is a matrix of dynamic size """
      d_inc = self.max_d0 / self.discretization
      p = max(0, d0 - dmin) / d_inc
      fp = min(p - tm.floor(p), 1.0)
      ip0 = max(0, min(self.discretization - 2, tm.floor(p)))
      ip1 = ip0 + 1

      a_inc0 = (self.max_As[ip0] - self.min_As[ip0]) / self.discretization
      a_inc1 = (self.max_As[ip1] - self.min_As[ip1]) / self.discretization

      result = 0.0
      if a_inc0 != 0.0 and a_inc1 != 0.0:
        q0 = max(self.min_As[ip0], min(self.max_As[ip0], A) - self.min_As[ip0]) / a_inc0
        q1 = max(self.min_As[ip1], min(self.max_As[ip1], A) - self.min_As[ip1]) / a_inc1

        fq0 = min(q0 - tm.floor(q0), 1.0)
        fq1 = min(q1 - tm.floor(q1), 1.0)

        iq00 = max(0, min(self.discretization - 2, q0))
        iq10 = max(0, min(self.discretization - 2, q1))
        iq01 = iq00 + 1
        iq11 = iq10 + 1
      
        v00 = mat[iq00, ip0]
        v01 = mat[iq01, ip0]
        v10 = mat[iq10, ip1]
        v11 = mat[iq11, ip1]

        dEdd0 = MathUtils.t_lerp(v00, v01, fq0)
        dEdd1 = MathUtils.t_lerp(v10, v11, fq1)
        assert(not tm.isnan(dEdd0) and not tm.isnan(dEdd1))

        result = MathUtils.t_lerp(dEdd0, dEdd1, fp)
      return result

    @ti.func
    def t_getInterpolateTableGrad(self, A: float, d0: float, mat, dmin: float) -> float:
      d_inc = self.max_d0 / self.discretization
      p = max(0.0, d0 - dmin) / d_inc
      ip0 = max(0, min(self.discretization - 2, tm.floor(p)))
      ip1 = ip0 + 1

      a_inc0 = (self.max_As[ip0] - self.min_As[ip0]) / self.discretization
      a_inc1 = (self.max_As[ip1] - self.min_As[ip1]) / self.discretization

      result = 0.0
      if a_inc0 != 0.0 and a_inc1 != 0.0:
        q0 = max(self.min_As[ip0], min(self.max_As[ip0], A) - self.min_As[ip0]) / a_inc0
        q1 = max(self.min_As[ip1], min(self.max_As[ip1], A) - self.min_As[ip1]) / a_inc1

        fq0 = min(q0 - tm.floor(q0), 1.0)
        fq1 = min(q1 - tm.floor(q1), 1.0)

        iq00 = max(0, min(self.discretization - 2, q0))
        iq10 = max(0, min(self.discretization - 2, q1))
        iq01 = iq00 + 1
        iq11 = iq10 + 1
      
        v00 = mat[iq00, ip0]
        v01 = mat[iq01, ip0]
        v10 = mat[iq10, ip1]
        v11 = mat[iq11, ip1]

        dEdd0 = MathUtils.t_lerp(v00, v01, fq0)
        dEdd1 = MathUtils.t_lerp(v10, v11, fq1)
        assert(not tm.isnan(dEdd0) and not tm.isnan(dEdd1))

        result = (dEdd1 - dEdd0) / d_inc
      return result

    @ti.func
    def t_getInterpolate_dEdd(self, A: float, d0: float) -> float:
      return self.t_getInterpolateTable(A, d0, self.dEdd_table, 0.0)
    
    @ti.func
    def t_getInterpolate_d2Edd2(self, A: float, d0: float) -> float:
      return self.t_getInterpolateTableGrad(A, d0, self.dEdd_table, 0.0)
    
    @ti.func
    def t_getInterpolate_alpha(self, A: float, d0: float) -> float:
      return self.t_getInterpolateTable(A, d0, self.alpha_table, 0.0)
    
    @ti.func
    def t_getInterpolate_dEdd_planar(self, A: float, d0: float) -> float:
      return self.t_getInterpolateTable(A, d0, self.dEdd_planar_table, 0.0)
    
    @ti.func
    def t_getInterpolate_d2Edd2_planar(self, A: float, d0: float) -> float:
      return self.t_getInterpolateTableGrad(A, d0, self.dEdd_planar_table, 0.0)
    
    @ti.func
    def t_getInterpolate_alpha_planar(self, A: float, d0: float) -> float:
      return self.t_getInterpolateTable(A, d0, self.alpha_planar_table, 0.0)

    @ti.func
    def t_setParameter(self, sigma: float, theta: float, radii: float, max_d0: float, disc: int):
      self.sigma = sigma
      self.theta = theta
      self.radii = radii
      self.max_d0 = max_d0
      self.discretization = disc
      self.min_d0 = radii * 2.0
      self.max_alpha = MathUtils.PI - self.theta

    @ti.func
    def t_computeH(self, R: float, alpha: float) -> float:
      return self.radii  * tm.sin(alpha) - R * (1.0 - tm.sin(self.theta + alpha))
    
    @ti.func
    def t_computeR(self, alpha: float, d0: float) -> float:
      return (d0 - 2.0 * self.radii * tm.cos(alpha)) / (2.0 * tm.cos(self.theta + alpha))
    
    @ti.func
    def t_computeA(self, R: float, alpha: float) -> float:
      return 2.0 * R * R * (alpha + self.theta - MathUtils.PI / 2 + 0.5 * tm.sin(2.0 * (alpha + self.theta))) + \
             2.0 * self.radii * R * (tm.sin(2.0 * alpha + self.radii) - tm.sin(self.radii)) - \
             self.radii * self.radii * (2.0 * alpha - tm.sin(2.0 * alpha))
    
    @ti.func
    def t_computeApproxA(self, alpha: float, d0: float) -> float:
      gamma = alpha + self.theta
      t2 = self.radii * self.radii
      t3 = d0 * d0
      t4 = tm.cos(self.theta)
      t5 = t4 * t4
      t6 = tm.sin(self.theta)
      return -t2 * tm.sin(self.theta * 2.0) + t2 * MathUtils.PI * (1.0 / 3.0) - \
              t3 * MathUtils.PI * (1.0 / 6.0) - gamma * t2 * (8.0 / 3.0) + \
              gamma * t3 * (1.0 / 3.0) + t2 * self.theta * 2.0 - \
              t2 * t5 * MathUtils.PI * (4.0 / 3.0) + d0 * self.radii * t4 * 2.0 + \
              gamma * t2 * t5 * (8.0 / 3.0) + d0 * gamma * self.theta * t6 * (2.0 / 3.0) - \
              d0 * self.theta * t6 * MathUtils.PI * (1.0 / 3.0)


scalar CohesionTable::computeApproxdEdd(const scalar& alpha,
                                        const scalar& d0) const {
  const scalar gamma = alpha + m_theta;

  const scalar t2 = sin(m_theta);
  const scalar t3 = m_radii * m_radii;
  const scalar t4 = cos(m_theta);
  const scalar t5 = d0 * d0;
  const scalar t6 = d0 * m_radii * t2 * 2.0;
  const scalar t7 = m_theta * 2.0;
  const scalar t8 = sin(t7);
  return (m_sigma *
          (t3 * -8.0 + t5 + t6 + t3 * (t4 * t4) * 8.0 + t3 * t8 * M_PI * 2.0 -
           gamma * t3 * t8 * 4.0 - d0 * gamma * m_radii * t4 * 2.0 +
           d0 * m_radii * t4 * M_PI) *
          2.0) /
         (t5 + t6 - (t2 * t2) * t3 * 8.0);
}

scalar CohesionTable::computedEddPlanar(const scalar& R,
                                        const scalar& alpha) const {
  if (R == 0.0) {
    return 0.0;
  } else {
    const scalar t2 = m_theta * 2.0;
    const scalar t3 = sin(alpha);
    const scalar t4 = alpha + m_theta;
    const scalar t5 = sin(t4);
    const scalar t6 = alpha + t2;
    const scalar t7 = m_radii * m_radii;
    const scalar t8 = sin(t6);
    const scalar t9 = R * R;
    const scalar t10 = alpha * 2.0;
    const scalar t11 = sin(t2);
    const scalar t12 = t2 + t10;
    const scalar t13 = sin(t12);
    const scalar t14 = m_theta * 3.0;
    const scalar t15 = alpha + t14;
    const scalar t16 = cos(t10);
    const scalar t17 = cos(t12);
    const scalar t18 = cos(m_theta);
    const scalar t19 = t10 + m_theta;
    const scalar t20 = cos(t19);
    return (m_sigma *
            (t7 * M_PI * -2.0 - t9 * M_PI * 2.0 + alpha * t7 * 2.0 +
             alpha * t9 * 2.0 + t3 * t7 * 4.0 + t3 * t9 * 4.0 + t7 * t8 * 2.0 +
             t8 * t9 * 4.0 - t7 * t11 * 2.0 + t7 * t13 * 2.0 - t9 * t11 +
             t9 * t13 * 3.0 + t7 * m_theta * 4.0 + t9 * m_theta * 4.0 +
             t7 * sin(t10) * 2.0 + t7 * sin(alpha - t2) * 2.0 +
             t9 * sin(t10 + m_theta * 4.0) - R * m_radii * sin(t14) +
             R * m_radii * sin(t15) * 2.0 + R * m_radii * sin(t19) * 5.0 -
             R * m_radii * sin(m_theta) * 3.0 +
             R * m_radii * sin(alpha - m_theta) * 6.0 + t7 * t16 * M_PI * 2.0 +
             t9 * t17 * M_PI * 2.0 + R * m_radii * t5 * 8.0 -
             alpha * t7 * t16 * 2.0 - alpha * t9 * t17 * 2.0 -
             t7 * t16 * m_theta * 4.0 - t9 * t17 * m_theta * 4.0 +
             R * m_radii * sin(t10 + t14) * 3.0 +
             R * m_radii * t18 * m_theta * 8.0 -
             R * m_radii * t20 * m_theta * 8.0 -
             R * m_radii * t18 * M_PI * 4.0 + R * m_radii * t20 * M_PI * 4.0 +
             R * alpha * m_radii * t18 * 4.0 -
             R * alpha * m_radii * t20 * 4.0)) /
           (R * (m_radii * 2.0 + R * t18 * 4.0 + R * cos(t4) * 3.0 +
                 R * cos(t15) + m_radii * cos(alpha) * 2.0 +
                 m_radii * cos(t2) * 2.0 + m_radii * cos(t6) * 2.0 -
                 R * t5 * M_PI * 2.0 + R * alpha * t5 * 2.0 -
                 m_radii * t3 * M_PI * 2.0 + R * t5 * m_theta * 4.0 +
                 alpha * m_radii * t3 * 2.0 + m_radii * t3 * m_theta * 4.0));
  }
}