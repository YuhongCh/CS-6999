import taichi as ti
import taichi.math as tm

@ti.data_oriented
class Integrator:
    def __init__(self, dt: float, type: str, criterion: float):
        self.dt = dt
        self.type = type
        self.criterion = criterion

    @ti.func
    def integrate(self, force: ti.types.ndarray(dtype=tm.vec4),
                        velocity: ti.types.ndarray(dtype=tm.vec4)):
        num_vertices = force.shape[0]
        for i in ti.ndrange(num_vertices):
            velocity[i] += force[i] * self.dt

    @ti.func
    def t_semi_implicit_integrate(self, grad_force: ti.types.sparse_matrix_builder(),
                                        force: ti.types.ndarray(dtype=tm.vec4),
                                        velocity: ti.types.ndarray(dtype=tm.vec4)):
        row_length = force.shape[0]

        A_matrix = grad_force.build()
        A_matrix *= self.dt * self.dt
        for i in ti.ndrange(row_length):
            grad_force[i, i] += 1
            force[i] = force[i] * self.dt + velocity[i]

        solver = ti.linalg.SparseSolver(solver_type="LLT")
        solver.analyze_pattern(A_matrix)
        solver.factorize(A_matrix)
        velocity = solver.solve(force)      # <-- double check if this assigned correctly
        success = solver.info()
        if not success:
            exit(-2)


