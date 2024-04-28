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
      p = max(0.0, d0 - dmin) / d_inc
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

    @ti.func
    def t_computeApproxdEdd(self, alpha: float, d0: float) -> float:
        gamma = alpha + self.theta
        t2 = tm.sin(self.theta)
        t3 = self.radii * self.radii
        t4 = tm.cos(self.theta)
        t5 = d0 * d0
        t6 = d0 * self.radii * t2 * 2.0
        t7 = self.theta * 2.0
        t8 = tm.sin(t7)
        return (self.sigma * (t3 * -8.0 + t5 + t6 + t3 * (t4 * t4) * 8.0 + t3 * t8 * MathUtils.PI * 2.0 -
               gamma * t3 * t8 * 4.0 - d0 * gamma * self.radii * t4 * 2.0 +
               d0 * self.radii * t4 * MathUtils.PI) * 2.0) / (t5 + t6 - (t2 * t2) * t3 * 8.0)

    @ti.func
    def t_computedEddPlanar(self, R: float, alpha: float) -> float: 
        result = R
        if R != 0.0:
            t2 = self.theta * 2.0
            t3 = tm.sin(alpha)
            t4 = alpha + self.theta
            t5 = tm.sin(t4)
            t6 = alpha + t2
            t7 = self.radii * self.radii
            t8 = tm.sin(t6)
            t9 = R * R
            t10 = alpha * 2.0
            t11 = tm.sin(t2)
            t12 = t2 + t10
            t13 = tm.sin(t12)
            t14 = self.theta * 3.0
            t15 = alpha + t14
            t16 = tm.cos(t10)
            t17 = tm.cos(t12)
            t18 = tm.cos(self.theta)
            t19 = t10 + self.theta
            t20 = tm.cos(t19)

            result = (self.sigma * (t7 * MathUtils.PI * -2.0 - t9 * MathUtils.PI * 2.0 + alpha * t7 * 2.0 +
                     alpha * t9 * 2.0 + t3 * t7 * 4.0 + t3 * t9 * 4.0 + t7 * t8 * 2.0 +
                     t8 * t9 * 4.0 - t7 * t11 * 2.0 + t7 * t13 * 2.0 - t9 * t11 +
                     t9 * t13 * 3.0 + t7 * self.theta * 4.0 + t9 * self.theta * 4.0 +
                     t7 * tm.sin(t10) * 2.0 + t7 * tm.sin(alpha - t2) * 2.0 +
                     t9 * tm.sin(t10 + self.theta * 4.0) - R * self.radii * tm.sin(t14) +
                     R * self.radii * tm.sin(t15) * 2.0 + R * self.radii * tm.sin(t19) * 5.0 -
                     R * self.radii * tm.sin(self.theta) * 3.0 +
                     R * self.radii * tm.sin(alpha - self.theta) * 6.0 + t7 * t16 * MathUtils.PI * 2.0 +
                     t9 * t17 * MathUtils.PI * 2.0 + R * self.radii * t5 * 8.0 -
                     alpha * t7 * t16 * 2.0 - alpha * t9 * t17 * 2.0 -
                     t7 * t16 * self.theta * 4.0 - t9 * t17 * self.theta * 4.0 +
                     R * self.radii * tm.sin(t10 + t14) * 3.0 +
                     R * self.radii * t18 * self.theta * 8.0 -
                     R * self.radii * t20 * self.theta * 8.0 -
                     R * self.radii * t18 * MathUtils.PI * 4.0 + R * self.radii * t20 * MathUtils.PI * 4.0 +
                     R * alpha * self.radii * t18 * 4.0 -
                     R * alpha * self.radii * t20 * 4.0) /
                    (R * (self.radii * 2.0 + R * t18 * 4.0 + R * tm.cos(t4) * 3.0 +
                     R * tm.cos(t15) + self.radii * tm.cos(alpha) * 2.0 +
                     self.radii * tm.cos(t2) * 2.0 + self.radii * tm.cos(t6) * 2.0 -
                     R * t5 * MathUtils.PI * 2.0 + R * alpha * t5 * 2.0 -
                     self.radii * t3 * MathUtils.PI * 2.0 + R * t5 * self.theta * 4.0 +
                     alpha * self.radii * t3 * 2.0 + self.radii * t3 * self.theta * 4.0)))
        return result

    @ti.func
    def t_computedEdd(self, R: float, alpha: float) -> float:
        result = 0.0
        if R != 0.0:
            t2 = tm.sin(alpha)
            t3 = alpha + self.theta
            t4 = tm.sin(t3)
            t5 = R * R
            t6 = self.radii * self.radii
            t7 = self.theta * 2.0
            t8 = alpha * 2.0
            t9 = t7 + t8
            t10 = tm.sin(t9)
            t11 = tm.cos(t8)
            t12 = tm.cos(t9)
            t13 = tm.cos(self.theta)
            t14 = t8 + self.theta
            t15 = tm.cos(t14)
            
            result = (self.sigma *
                        (-t5 * MathUtils.PI - t6 * MathUtils.PI + alpha * t5 * 2.0 + alpha * t6 * 2.0 +
                        t5 * t10 * 2.0 + t6 * t10 + t5 * self.theta * 2.0 +
                        t6 * self.theta * 2.0 - t6 * tm.sin(t7) + t6 * tm.sin(t8) +
                        R * self.radii * tm.sin(t14) * 3.0 - R * self.radii * tm.sin(self.theta) * 2.0 +
                        R * self.radii * tm.sin(t8 + self.theta * 3.0) + t5 * t12 * MathUtils.PI +
                        t6 * t11 * MathUtils.PI - alpha * t5 * t12 * 2.0 - alpha * t6 * t11 * 2.0 -
                        t5 * t12 * self.theta * 2.0 - t6 * t11 * self.theta * 2.0 +
                        R * self.radii * t13 * self.theta * 4.0 -
                        R * self.radii * t15 * self.theta * 4.0 -
                        R * self.radii * t13 * MathUtils.PI * 2.0 + R * self.radii * t15 * MathUtils.PI * 2.0 +
                        R * alpha * self.radii * t13 * 4.0 -
                        R * alpha * self.radii * t15 * 4.0)
                      ) / (R * (self.radii * tm.cos(alpha + t7) + R * tm.cos(t3) * 2.0 +
                       self.radii * tm.cos(alpha) - R * t4 * MathUtils.PI + R * alpha * t4 * 2.0 -
                       self.radii * t2 * MathUtils.PI + R * t4 * self.theta * 2.0 +
                       alpha * self.radii * t2 * 2.0 + self.radii * t2 * self.theta * 2.0))
        return result

    @ti.func
    def t_computeRPlanar(self, alpha, d0) -> float:
        return (d0 - self.radii * tm.cos(alpha)) / (tm.cos(self.theta + alpha) + tm.ccos(self.theta))

    @ti.func
    def t_computeAPlanar(self, R: float, alpha: float) -> float:
        return 2.0 * (0.5 * self.radii * self.radii * tm.sin(alpha) * tm.cos(alpha) +
                      self.radii * tm.sin(alpha) * R * tm.cos(self.theta + alpha) +
                      0.5 * R * R * tm.sin(self.theta + alpha) * tm.cos(self.theta + alpha)
                      ) + \
                     (2.0 * (R * tm.sin(self.theta + alpha) - R * tm.sin(self.theta) +
                      self.radii * tm.sin(alpha)) * R * tm.cos(self.theta) +
                      R * R * tm.sin(self.theta) * tm.cos(self.theta) -
                      (alpha * self.radii * self.radii + R * R * (MathUtils.PI - 2.0 * self.theta - alpha))
                      )

    @ti.func
    def t_computeHPlanar(self, R: float, alpha: float) -> float:
        return self.t_computeH(R, alpha)

    @ti.func
    def t_computeApproxAPlanar(self, alpha: float, d0: float) -> float:
        result = self.theta
        if self.theta != 0.0:
            gamma = alpha + self.theta * 2.0
            t2 = self.theta * 3.0
            t3 = tm.cos(t2)
            t4 = self.radii * self.radii
            t5 = d0 * d0
            t6 = tm.cos(self.theta)
            t7 = MathUtils.PI * MathUtils.PI
            t8 = self.theta * 5.0
            t9 = tm.cos(t8)
            t10 = gamma * gamma
            t11 = tm.sin(self.theta)
            t12 = tm.sin(t2)
            t13 = tm.sin(t8)
            result = 1.0 / (t11 * t11 * t11) * (1.0 / 4.8E1) * \
                        (t3 * t4 * 6.0 - t3 * t5 * 1.2E1 + t5 * t6 * 1.2E1 - t4 * t9 * 6.0 -
                        t4 * t11 * MathUtils.PI * 4.0 - t4 * t12 * MathUtils.PI * 1.6E1 -
                        t5 * t11 * MathUtils.PI * 3.2E1 + t4 * t13 * MathUtils.PI * 4.0 -
                        d0 * self.radii * t3 * 2.4E1 + d0 * self.radii * t6 * 2.4E1 -
                        gamma * t4 * t11 * 3.2E1 + gamma * t4 * t12 * 2.8E1 +
                        gamma * t5 * t11 * 3.2E1 - gamma * t4 * t13 * 4.0 -
                        t3 * t4 * t7 * 3.0 - t3 * t4 * t10 * 3.0 + t4 * t6 * t7 * 2.2E1 +
                        t5 * t6 * t7 * 2.0E1 + t4 * t6 * t10 * 2.2E1 + t4 * t7 * t9 +
                        t5 * t6 * t10 * 2.0E1 + t4 * t9 * t10 + t4 * t11 * self.theta * 7.2E1 -
                        t4 * t12 * self.theta * 2.4E1 + d0 * gamma * self.radii * t11 * 4.0E1 +
                        d0 * gamma * self.radii * t12 * 8.0 + d0 * self.radii * t6 * t7 * 4.0E1 +
                        d0 * self.radii * t6 * t10 * 4.0E1 -
                        d0 * self.radii * t11 * MathUtils.PI * 4.0E1 -
                        d0 * self.radii * t12 * MathUtils.PI * 8.0 + gamma * t3 * t4 * MathUtils.PI * 6.0 -
                        gamma * t4 * t6 * MathUtils.PI * 4.4E1 - gamma * t5 * t6 * MathUtils.PI * 4.0E1 -
                        gamma * t4 * t9 * MathUtils.PI * 2.0 -
                        d0 * gamma * self.radii * t6 * MathUtils.PI * 8.0E1)
        return result

    @ti.func
    def t_computeApproxdEddPlanar(self, alpha: float, d0: float) -> float:
        gamma = alpha + self.theta * 2.0
        result = 0.0
        if self.theta == 0.0:
            t2 = d0 * d0
            t3 = self.radii * self.radii
            result = self.sigma * 1.0 / pow(d0 + self.radii, 2.0) * (1.0 / 2.0) * \
                     (t2 * MathUtils.PI * 4.0 + t3 * MathUtils.PI * 4.0 - gamma * t2 * 4.0 -
                      gamma * t3 * 4.0 + d0 * self.radii * MathUtils.PI * 8.0 -
                      d0 * gamma * self.radii * 8.0)
        else:
            t2 = self.theta * 2.0
            t3 = tm.cos(t2)
            t4 = d0 * d0
            t5 = self.radii * self.radii
            t6 = t3 * t3
            t7 = MathUtils.PI * MathUtils.PI
            t8 = gamma * gamma
            t9 = tm.sin(t2)
            t10 = self.theta * 4.0
            t11 = tm.sin(t10)
            result = (self.sigma * 1.0 / pow(d0 + self.radii * t3, 2.0) *
                     (t4 * -2.0 + t3 * t4 * 2.0 + t4 * t7 - t5 * t6 * 2.0 + t4 * t8 -
                     t5 * t7 * 2.0 - t5 * t8 * 2.0 - gamma * t4 * MathUtils.PI * 2.0 +
                     gamma * t5 * MathUtils.PI * 4.0 - t4 * t9 * MathUtils.PI * 2.0 - t5 * t11 * MathUtils.PI -
                     d0 * self.radii * t3 * 4.0 + d0 * self.radii * t6 * 4.0 -
                     d0 * self.radii * t7 - d0 * self.radii * t8 + gamma * t4 * t9 * 2.0 +
                     gamma * t5 * t11 - t3 * t4 * t7 + t3 * t5 * t6 * 2.0 -
                     t3 * t4 * t8 + t3 * t5 * t7 + t3 * t5 * t8 + t5 * t6 * t7 +
                     t5 * t6 * t8 + d0 * gamma * self.radii * t9 * 2.0 +
                     d0 * gamma * self.radii * t11 + d0 * self.radii * t6 * t7 +
                     d0 * self.radii * t6 * t8 + d0 * gamma * self.radii * MathUtils.PI * 2.0 -
                     d0 * self.radii * t9 * MathUtils.PI * 2.0 - d0 * self.radii * t11 * MathUtils.PI +
                     gamma * t3 * t4 * MathUtils.PI * 2.0 - gamma * t3 * t5 * MathUtils.PI * 2.0 -
                     gamma * t5 * t6 * MathUtils.PI * 2.0 -
                     d0 * gamma * self.radii * t6 * MathUtils.PI * 2.0) *
                    (-1.0 / 2.0)) / tm.sin(self.theta)
        return result

    @ti.func
    def t_computedEddAreaDist(self, A_target: float, d0: float) -> float:
        result = 0.0
        if d0 < self.t_getDStar():
            result = self.t_computedEddAreaDist(A_target, self.t_getDStar())
        elif d0 < tm.sqrt(A_target / MathUtils.PI + 2.0 * self.radii * self.radii) - self.radii * 2.0:
            result = 0.0
        else:
            alpha = self.t_getInterpolate_alpha(A_target, d0)
            gamma = alpha + self.theta
            dEdd = 0.0
            if (gamma < MathUtils.PI * 0.5 + CohensionTable.ANGLE_EPSILON and
               gamma > MathUtils.PI * 0.5 + CohensionTable.ANGLE_EPSILON):
               dEdd = self.t_computeApproxdEdd(alpha, d0)
            else:
                R_target = self.t_computeR(alpha, d0)
                dEdd = self.t_computedEdd(R_target, alpha)
            result = max(0.0, dEdd)
        return result


