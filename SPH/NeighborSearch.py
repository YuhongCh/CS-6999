import taichi as ti
import taichi.math as tm

from Config import SPHConfig
from ParticleSystem import ParticleSystem, ParticleType


@ti.data_oriented
class NeighborSearch:
    """ A Simple neighbor search algorithm with scattered grid and prefix sum sort """
    def __init__(self, config: SPHConfig, particle_system: ParticleSystem):
        self.neighbor_radius2 = config.support_radius * config.support_radius
        self.grid_radius = config.grid_radius
        self.start_domain = config.start_domain
        self.end_domain = config.end_domain

        self.num_particle = particle_system.get_particle_num(ParticleType.NonEmpty)
        self.particle_position = particle_system.position

        self.inv_grid_radius = 1 / self.grid_radius
        self.grid_dimension = tm.ivec3(0, 0, 0)
        self.grid_dimension[0] = int((self.end_domain.x - self.start_domain.x) * self.inv_grid_radius) + 1
        self.grid_dimension[1] = int((self.end_domain.y - self.start_domain.y) * self.inv_grid_radius) + 1
        self.grid_dimension[2] = int((self.end_domain.z - self.start_domain.z) * self.inv_grid_radius) + 1

        self.grid_count = self.grid_dimension.x * self.grid_dimension.y * self.grid_dimension.z
        self.pid2gid = ti.field(int, shape=self.num_particle)
        self.grid_end_indices = ti.field(int, shape=self.grid_count)
        self.grid_sorted = ti.field(int, shape=self.grid_count)
        self.grid_atomic_buffer = ti.field(int, shape=self.grid_count)
        self.prefix_executor = ti.algorithms.PrefixSumExecutor(self.grid_count)

    @ti.func
    def get_gid_3D(self, pos: tm.vec3) -> tm.ivec3:
        gid_3D = ti.cast((pos - self.start_domain) * self.inv_grid_radius, int)
        return tm.clamp(gid_3D, tm.ivec3(0, 0, 0), self.grid_dimension - tm.ivec3(1, 1, 1))

    @ti.func
    def get_gid_3D_to_1D(self, x: int, y: int, z: int) -> int:
        return x + z * self.grid_dimension.x + y * self.grid_dimension.x * self.grid_dimension.z

    @ti.func
    def get_gid_1D(self, pos: tm.vec3) -> int:
        gid_3D = self.get_gid_3D(pos)
        return self.get_gid_3D_to_1D(gid_3D.x, gid_3D.y, gid_3D.z)

    @ti.func
    def is_in_bound(self, x: int, y: int, z: int) -> bool:
        cond1 = x >= 0 and x < self.grid_dimension.x
        cond2 = y >= 0 and y < self.grid_dimension.y
        cond3 = z >= 0 and z < self.grid_dimension.z
        return cond1 and cond2 and cond3

    @ti.func
    def is_neighbor(self, pi, pj) -> bool:
        dist = self.particle_position[pi] - self.particle_position[pj]
        return pi != pj and tm.dot(dist, dist) < self.neighbor_radius2

    @ti.kernel
    def compute_gid(self):
        for pid in self.pid2gid:
            pos = self.particle_position[pid]
            gid = self.get_gid_1D(pos)
            self.pid2gid[pid] = gid

            # += atomic
            # temporarily serve to count number of pid in a gid
            self.grid_end_indices[gid] += 1

    @ti.kernel
    def compute_grid_sort(self):
        for pid in range(self.num_particle):
            gid = self.pid2gid[pid]

            start = 0
            if gid > 0:
                start = self.grid_end_indices[gid - 1]
            target = ti.atomic_add(self.grid_atomic_buffer[gid], 1) + start
            self.grid_sorted[target] = pid

    def compute_update(self):
        self.grid_end_indices.fill(0)
        self.grid_atomic_buffer.fill(0)
        self.compute_gid()
        self.prefix_executor.run(self.grid_end_indices)
        self.compute_grid_sort()

    @ti.func
    def for_each_neighbors(self, pid: int, func: ti.template(), ret: ti.template()):
        gid_3D = self.get_gid_3D(self.particle_position[pid])
        for delta_x, delta_y, delta_z in ti.static(ti.ndrange((-1, 2), (-1, 2),(-1, 2))):
            nx = gid_3D.x + delta_x
            ny = gid_3D.y + delta_y
            nz = gid_3D.z + delta_z
            if self.is_in_bound(nx, ny, nz):
                ngid = self.get_gid_3D_to_1D(nx, ny, nz)
                start_idx = 0
                if ngid > 0:
                    start_idx = self.grid_end_indices[ngid - 1]
                for idx in range(start_idx, self.grid_end_indices[ngid]):
                    if self.is_neighbor(pid, self.grid_sorted[idx]):
                        func(pid, self.grid_sorted[idx], ret)
