import taichi as ti
import taichi.math as tm
import numpy as np

from Liquid_Hair_Interaction.Simulation.SPH.SPHConfig import SPHConfig

"""
IMPORTANT: I hereby made one assumption s.t. ONLY surround 3x3x3 grids are neighbors
"""

# @ti.data_oriented
# class HashGridNS:
#     def __init__(self, grid_radius, max_in_grid, max_neighbor, particle_system):
#         # copy over the parameter
#         self.grid_radius = grid_radius
#         self.inv_grid_radius = 1.0 / grid_radius
#         self.search_radius = 1.5 * grid_radius

#         self.max_in_grid = max_in_grid
#         self.max_neighbor = max_neighbor
#         self.particle_system = particle_system

#         # declare needed data structure
#         self.particle_count = self.particle_system.p_GetNumParticle()

#         self.grid_count = ti.field(dtype=int, shape=self.particle_count)
#         self.grid = ti.field(dtype=int, shape=(self.num_particle, self.max_in_grid))
#         self.max_curr_neighbor = ti.field(dtype=int, shape=self.particle_count)
#         self.neighbor_count = ti.field(dtype=int, shape=self.liquid_count)
#         self.neighbor = ti.field(dtype=int, shape=(self.liquid_count, self.max_neighbor))

#         # variables require computation
#         self.start_domain = ti.field(dtype=int, shape=3)
#         self.end_domain = ti.field(dtype=int, shape=3)
#         self.grid_dim = ti.field(dtype=int, shape=3)


    

#     @ti.func
#     def t_IndexInGrid(self, pos_index: tm.vec3) -> bool:
#         cond1 = pos_index.x >= 0 and pos_index.y >= 0 and pos_index.z >= 0
#         cond2 = pos_index.x < self.grid_dim.x and pos_index.y < self.grid_dim.y and pos_index.z < self.grid_dim.z
#         return cond1 and cond2

#     @ti.func
#     def t_HashIndex(self, pos_index: tm.vec3) -> int:
#         p1 = 73856093 * pos_index.x
#         p2 = 19349663 * pos_index.y
#         p3 = 83492791 * pos_index.z
#         return ((p1^p2^p3) % self.particle_count + self.particle_count) % self.particle_count 

#     @ti.kernel
#     def t_Update(self):
#         # remove the previous data
#         for i in self.grid_count:
#             self.grid_count[i] = 0
#         for i in self.neighbor_count:
#             self.neighbor_count[i] = 0

#         #insert pos
#         for i in self.particle_system.pos:
#             pos = self.particle_system.pos[i]
#             pos_index = ti.cast((pos - self.start_domain) * self.inv_grid_radius, int)
#             if self.t_IndexInGrid(pos_index):
#                 hash_index = self.t_HashIndex(pos_index)
#                 old_count = ti.atomic_add(self.grid_count[hash_index], 1)
#                 if old_count > self.max_in_grid - 1:
#                     self.grid_count[hash_index] = self.max_in_grid
#                 else:
#                     self.grid[hash_index, old_count] = i
        
#         #find neighbour
#         for i in self.neighbor_count:
#             pos = self.particle_system.pos[i]
#             pos_index = ti.cast((pos - self.start_domain) * self.inv_grid_radius, int)
#             indexV         = ti.cast((self.particle_data.pos[i] - self.min_boundary[0])*self.invGridR, ti.i32)
#             if self.check_in_box(indexV) == 1:
#                 for m in range(-2,3):
#                     for n in range(-2,3):
#                         for q in range(-2,3):
#                             self.insert_neighbor(i, ti.Vector([m, n, q]) + indexV)


@ti.data_oriented
class BaseNS:
    def __init__(self, grid_radius, max_part_in_grid, max_neighbor, particle_system):
        # copy over the configuration
        self.num_particle = particle_system.p_GetNumParticle()
        self.start_domain = particle_system.start_domain
        self.end_domain = particle_system.end_domain

        self.grid_radius = grid_radius
        self.max_neighbor = max_neighbor
        self.max_part_in_grid = max_part_in_grid

        # configure the neighbor grid setting
        self.grid_dim = ti.field(dtype=int, shape=3)
        for i in range(3):
            self.grid_dim[i] = ti.ceil((self.end_domain[i] - self.start_domain[i]) / self.grid_radius)
        self.tot_grid = self.grid_dim.x * self.grid_dim.y * self.grid_dim.z

    @ti.func
    def t_Pid2GridCoord(self, part_idx: int) -> [int, int, int]:
        p = self.ps.particles[part_idx]
        coord = p.x - self.start_domain
        x_idx = max(0, int(coord.x / self.grid_size - 0.5))
        y_idx = max(0, int(coord.y / self.grid_size - 0.5))
        z_idx = max(0, int(coord.z / self.grid_size - 0.5))
        return [x_idx, y_idx, z_idx]

    @ti.func
    def t_Pid2Gid(self, part_idx: int) -> int:
        x_idx, y_idx, z_idx = self.t_Pid2GridCoord(part_idx)
        return self.t_GridCoord2Gid(x_idx, y_idx, z_idx)

    @ti.func
    def t_GridCoord2Gid(self, x_idx: int, y_idx: int, z_idx: int) -> int:
        return y_idx * self.grid_dim.x * self.grid_dim.z + z_idx * self.grid_dim.x + x_idx
    
    def p_GridCoord2Gid(self, x_idx: int, y_idx: int, z_idx: int):
        return y_idx * self.grid_dim.x * self.grid_dim.z + z_idx * self.grid_dim.x + x_idx

    @ti.func
    def t_IsNeighbor(self, pa_idx: int, pb_idx: int) -> bool:
        dist = self.ps.particles[pa_idx].x - self.ps.particles[pb_idx].x
        is_neighbor = False
        if tm.dot(dist, dist) < self.search_radius:
            is_neighbor = True
        return is_neighbor

    def p_DebugPrint(self):
        print("===============NS Data================")
        print(f"start_domain: {self.start_domain}")
        print(f"end_domain: {self.end_domain}")
        print(f"grid_size: {self.grid_size}")
        print(f"grid_dim: {self.grid_dim}")
        print(f"total_grid: {self.tot_grid}")
        print("=============== End ================")

    @ti.func
    def t_ForEachNeighbors(self, task: ti.template(), ret: ti.template()):
        raise NotImplementedError

    def p_Update(self):
        raise NotImplementedError


@ti.data_oriented
class UniformGridNS(BaseNS):
    """ Uniform grid neighbor search, should be fast but memory intensive """
    def __init__(self, grid_radius, max_part_in_grid, max_neighbor, particle_system):
        super().__init__(grid_radius, max_part_in_grid, max_neighbor, particle_system)
        self.grid = ti.field(dtype=int, shape=(self.tot_grid, self.max_part_in_grid))
        self.grid_count = ti.field(dtype=int, shape=self.tot_grid)

    @ti.kernel
    def t_Update(self):
        # clear the grid
        for i in range(self.tot_grid):
            self.grid_count[i] = 0

        # re-compute the entire grid
        for part_idx in self.ps.particles:
            grid_idx = self.t_PartIdx2GridIndex(part_idx)
            if grid_idx >= self.max_part_in_grid:
                raise RuntimeError("gid surpass the limit")
            self.grid[grid_idx, self.grid_count[grid_idx]] = part_idx
            self.grid_count[grid_idx] += 1

    @ti.func
    def t_ForEachNeighbors(self, pi: int, task: ti.template(), ret: ti.template()):
        x_coord, y_coord, z_coord = self.t_PartIdx2GridCoord(pi)

        # for each neighbor grid
        for x_offset in ti.static([-1, 0, 1]):
            for y_offset in ti.static([-1, 0, 1]):
                for z_offset in ti.static([-1, 0, 1]):
                    neighbor_idx = self.t_GridCoord2GridIndex(x_coord + x_offset, y_coord + y_offset, z_coord + z_offset)
                    neighbor_count = 0
                    if neighbor_idx >= 0 and neighbor_idx < self.tot_grid:
                        neighbor_count = self.grid_count[neighbor_idx]

                    for i in range(neighbor_count):
                        pj = self.grid[neighbor_idx, i]
                        if pi != pj:
                            task(pi, pj, ret)

    def p_Update(self):
        self.t_Update()

