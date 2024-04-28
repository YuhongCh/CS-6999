import taichi as ti
import taichi.math as tm

from Config import SPHConfig

EPSILON = 1e-6

class ParticleType:
    NonEmpty = -1
    Empty = 0
    Fluid = 1
    Solid = 2

@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SPHConfig):
        # assign parameters from config
        self.max_particle_count = config.max_particle_count
        self.particle_radius = config.particle_radius
        self.ref_mass = config.ref_mass
        self.particle_volume = config.particle_volume
        self.support_radius = config.support_radius
        self.ref_density = config.ref_density
        self.min_coord = config.start_domain
        self.max_coord = config.end_domain
        self.dt = ti.field(float, shape=())

        # attribute to Taichi limit, use a field as int here to use in @ti.kernel
        # use num_particle as public variable in other file
        self.num_particle_field = ti.field(dtype=ti.int32, shape=())
        self.num_fluid_particle = 0
        self.num_solid_particle = 0

        # declare needed array fields
        self.position = ti.Vector.field(2, dtype=float, shape=self.max_particle_count)
        self.velocity = ti.Vector.field(2, dtype=float, shape=self.max_particle_count)
        self.acceleration = ti.Vector.field(2, dtype=float, shape=self.max_particle_count)
        self.type = ti.field(int, shape=self.max_particle_count)
        self.density = ti.field(float, shape=self.max_particle_count)
        self.pressure = ti.field(float, shape=self.max_particle_count)

        # precomputed parameters
        self.density.fill(self.ref_density)

    @ti.kernel
    def __generate_particles(self, start_domain: tm.vec2, end_domain: tm.vec2, interval: float, particle_type: int) -> int:
        """
         generate particles compactly within the bounding box
         @return number of particles generated
        """
        dimension = ti.cast((end_domain - start_domain) / interval + 1, dtype=int)
        for coord in ti.grouped(ti.ndrange((0, dimension.x), (0, dimension.y))):
            position = start_domain + coord * interval
            pid = ti.atomic_add(self.num_particle_field[None], 1)
            self.position[pid] = position
            self.type[pid] = particle_type
        return dimension.x * dimension.y

    def generate_particles(self, start_domain: tm.vec2, end_domain: tm.vec2, particle_type: int):
        interval = self.particle_radius * 2
        num_particle = self.__generate_particles(start_domain, end_domain, interval, particle_type)
        if particle_type == ParticleType.Fluid:
            self.num_fluid_particle += num_particle

    def add_particle(self, pos: tm.vec2, particle_type: int):
        self.position[self.num_particle_field[None]] = pos
        self.type[self.num_particle_field[None]] = particle_type
        self.num_particle_field[None] += 1
        self.num_fluid_particle += 1

    @ti.kernel
    def reset_parameters(self):
        for pid in range(self.num_particle_field[None]):
            self.acceleration[pid] = tm.vec2(0, 0)

    @ti.func
    def compute_CFL(self):
        max_velocity = 0.1
        for pid in range(self.num_particle_field[None]):
            ti.atomic_max(max_velocity, self.velocity[pid].norm())
        self.dt[None] = 0.4 * self.particle_radius / max_velocity

    @ti.kernel
    def step(self):
        self.compute_CFL()
        for pid in range(self.num_particle_field[None]):
            if self.type[pid] == ParticleType.Fluid:
                self.velocity[pid] += self.acceleration[pid] * self.dt[None]
                self.position[pid] += self.velocity[pid] * self.dt[None]

                if self.position[pid].x < self.min_coord.x:
                    self.position[pid].x = self.min_coord.x
                    self.velocity[pid].x = -self.velocity[pid].x
                elif self.position[pid].x > self.max_coord.x:
                    self.position[pid].x = self.max_coord.x
                    self.velocity[pid].x = -self.velocity[pid].x

                if self.position[pid].y < self.min_coord.y:
                    self.position[pid].y = self.min_coord.y
                    self.velocity[pid].y = -self.velocity[pid].y
                elif self.position[pid].y > self.max_coord.y:
                    self.position[pid].y = self.max_coord.y
                    self.velocity[pid].y = -self.velocity[pid].y


    def get_particle_num(self, particle_type: int) -> int:
        """ -1 for total number of particle added """
        if particle_type == ParticleType.NonEmpty:
            return self.num_particle_field[None]
        elif particle_type == ParticleType.Empty:
            return self.max_particle_count - self.num_particle_field[None]
        elif particle_type == ParticleType.Fluid:
            return self.num_fluid_particle
        elif particle_type == ParticleType.Boundary:
            return self.num_solid_particle
        else:
            raise NotImplementedError

    def dump(self, pid: int):
        print(f"{pid}\t {self.position[pid]}\t {self.velocity[pid]}\t {self.acceleration[pid]}\t {self.density[pid]}\t {self.pressure[pid]}")

