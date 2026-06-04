import taichi as ti
import numpy as np
import math

@ti.data_oriented
class TaichiFluidSolver:
    def __init__(self, res=128, domain_size=2.0, particles=100000):
        self.domain_size = domain_size
        self.max_particles = particles
        
        # ==========================================
        # 1. SPH 물리 상수 세팅 (물에 가까운 점성과 압력)
        # ==========================================
        self.rho0 = 1000.0           # 기준 밀도
        self.stiffness = 500.0       # 압력 강성 (높을수록 압축되지 않음)
        self.viscosity = 0.05        # 점성 (끈적임 정도)
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        self.dt = 2e-4               # 타임스텝 (SPH는 작게 유지해야 터지지 않음)
        
        self.particle_radius = 0.02
        self.h = self.particle_radius * 2.5  # 스무딩 반경 (주변을 탐색할 범위)
        self.h2 = self.h * self.h
        self.p_mass = self.rho0 * (self.particle_radius * 2)**3 * 0.8
        
        # 커널 함수 상수 미리 계산 (CPU에서 한 번만 연산하여 GPU 부하 감소)
        self.poly6_factor = 315.0 / (64.0 * math.pi * self.h**9)
        self.spiky_grad_factor = -45.0 / (math.pi * self.h**6)
        self.visc_lap_factor = 45.0 / (math.pi * self.h**6)

        # ==========================================
        # 2. O(N) 초고속 이웃 탐색을 위한 공간 해싱 격자 설정
        # ==========================================
        self.grid_size = self.h
        self.grid_res = int(np.ceil(domain_size / self.grid_size))
        self.max_particles_per_cell = 200  # 한 격자 방에 최대로 들어갈 수 있는 입자 수
        
        # 공간 해싱 필드 (MPM과 달리 물리량이 아닌 '파티클 인덱스'만 저장합니다)
        self.grid_count = ti.field(dtype=ti.i32, shape=(self.grid_res, self.grid_res, self.grid_res))
        self.grid_particles = ti.field(dtype=ti.i32, shape=(self.grid_res, self.grid_res, self.grid_res, self.max_particles_per_cell))

        # ==========================================
        # 3. 파티클 속성 필드 할당 (Lagrangian)
        # ==========================================
        self.num_particles = ti.field(dtype=ti.i32, shape=())
        self.p_x = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)
        self.p_v = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)
        self.p_a = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles) # 가속도
        self.p_rho = ti.field(dtype=ti.f32, shape=self.max_particles)        # 밀도
        self.p_press = ti.field(dtype=ti.f32, shape=self.max_particles)      # 압력
        self.p_color = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)

    # ------------------------------------------------------------------
    # 초기화 및 인터페이스 함수들 (기존 구조 유지)
    # ------------------------------------------------------------------
    def setup_initial_fluid_block(self):
        half_d = self.domain_size / 2.0
        
        # 파티클이 처음부터 겹쳐서 폭발하지 않도록 간격을 넉넉히 둡니다.
        spacing = self.particle_radius * 2.0
        x_range = np.arange(-half_d + 0.2, -half_d + 0.8, spacing)
        y_range = np.arange(-half_d + 0.1, -half_d + 0.8, spacing)
        z_range = np.arange(-half_d + 0.2, -half_d + 0.8, spacing)

        X, Y, Z = np.meshgrid(x_range, y_range, z_range)
        positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1).astype(np.float32)
        
        p_count = min(len(positions), self.max_particles)
        self.num_particles[None] = p_count

        self._init_particles_kernel(positions[:p_count], p_count)
        print(f"🌊 SPH 유체 블록 초기화 완료! 파티클 수: {p_count:,}개")

    @ti.kernel
    def _init_particles_kernel(self, pos_arr: ti.types.ndarray(), p_count: ti.i32):
        for i in range(p_count):
            self.p_x[i] = ti.Vector([pos_arr[i, 0], pos_arr[i, 1], pos_arr[i, 2]])
            self.p_v[i] = ti.Vector([0.0, 0.0, 0.0])
            self.p_a[i] = ti.Vector([0.0, 0.0, 0.0])
            self.p_rho[i] = self.rho0
            self.p_press[i] = 0.0
            self.p_color[i] = ti.Vector([0.1, 0.6, 1.0])

    def get_active_particle_count(self):
        return self.num_particles[None]

    @ti.kernel
    def copy_positions_to_field(self, target_field: ti.template()):
        for i in range(self.num_particles[None]):
            target_field[i] = self.p_x[i]

    # ------------------------------------------------------------------
    # SPH 핵심 시뮬레이션 파이프라인
    # ------------------------------------------------------------------
    def step(self):
        # SPH는 명시적 적분을 사용하므로 뻗어나가는 힘을 감당하기 위해 서브스텝을 많이 줍니다.
        substeps = 25
        for _ in range(substeps):
            self.update_grid()
            self.compute_density_pressure()
            self.compute_forces()
            self.integrate()

    @ti.func
    def get_cell(self, pos):
        half_d = self.domain_size / 2.0
        return ti.cast((pos + half_d) / self.grid_size, ti.i32)

    @ti.func
    def is_valid_cell(self, c):
        return 0 <= c.x < self.grid_res and 0 <= c.y < self.grid_res and 0 <= c.z < self.grid_res

    @ti.kernel
    def update_grid(self):
        # 1. 격자 초기화
        for i, j, k in self.grid_count:
            self.grid_count[i, j, k] = 0
            
        # 2. 파티클을 해당하는 격자 방에 배정 (공간 해싱)
        for p in range(self.num_particles[None]):
            cell = self.get_cell(self.p_x[p])
            if self.is_valid_cell(cell):
                idx = ti.atomic_add(self.grid_count[cell], 1)
                if idx < self.max_particles_per_cell:
                    self.grid_particles[cell, idx] = p

    @ti.kernel
    def compute_density_pressure(self):
        for p_i in range(self.num_particles[None]):
            pos_i = self.p_x[p_i]
            cell = self.get_cell(pos_i)
            density = 0.0
            
            # 주변 3x3x3 격자의 파티클들하고만 거리를 잽니다. (O(N) 최적화의 핵심)
            for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                neighbor_cell = cell + offset
                if self.is_valid_cell(neighbor_cell):
                    count = ti.min(self.grid_count[neighbor_cell], self.max_particles_per_cell)
                    for j in range(count):
                        p_j = self.grid_particles[neighbor_cell, j]
                        r_vec = pos_i - self.p_x[p_j]
                        r2 = r_vec.norm_sqr()
                        
                        # 스무딩 반경(h) 안쪽의 이웃만 밀도에 기여 (Poly6 커널)
                        if r2 < self.h2:
                            if r2 > 1e-10:
                                h2_r2 = self.h2 - r2
                                density += self.p_mass * self.poly6_factor * (h2_r2 ** 3)
                            else:
                                density += self.p_mass * self.poly6_factor * (self.h2 ** 3)
            
            self.p_rho[p_i] = density
            # 상태 방정식(Tait Equation)으로 압력 도출
            self.p_press[p_i] = self.stiffness * ti.max(density / self.rho0 - 1.0, 0.0)

    @ti.kernel
    def compute_forces(self):
        for p_i in range(self.num_particles[None]):
            pos_i = self.p_x[p_i]
            vel_i = self.p_v[p_i]
            rho_i = self.p_rho[p_i]
            press_i = self.p_press[p_i]
            
            f_press = ti.Vector([0.0, 0.0, 0.0])
            f_visc = ti.Vector([0.0, 0.0, 0.0])
            
            cell = self.get_cell(pos_i)
            for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                neighbor_cell = cell + offset
                if self.is_valid_cell(neighbor_cell):
                    count = ti.min(self.grid_count[neighbor_cell], self.max_particles_per_cell)
                    for j in range(count):
                        p_j = self.grid_particles[neighbor_cell, j]
                        if p_i != p_j:
                            r_vec = pos_i - self.p_x[p_j]
                            r = r_vec.norm()
                            if 1e-5 < r < self.h:
                                rho_j = self.p_rho[p_j]
                                press_j = self.p_press[p_j]
                                vel_j = self.p_v[p_j]
                                
                                # 압력 반발력 (Spiky Gradient)
                                grad_w = self.spiky_grad_factor * ((self.h - r) ** 2) * (r_vec / r)
                                p_term = (press_i / (rho_i**2 + 1e-6)) + (press_j / (rho_j**2 + 1e-6))
                                f_press += -self.p_mass * p_term * grad_w
                                
                                # 끈적이는 점성력 (Viscosity Laplacian)
                                lap_w = self.visc_lap_factor * (self.h - r)
                                v_term = (vel_j - vel_i) / (rho_j + 1e-6)
                                f_visc += self.viscosity * self.p_mass * v_term * lap_w
                                
            # 총 가속도 누적
            self.p_a[p_i] = f_press + f_visc + self.gravity

    @ti.kernel
    def integrate(self):
        half_d = self.domain_size / 2.0
        boundary_min = -half_d + self.particle_radius
        boundary_max = half_d - self.particle_radius
        
        for p_i in range(self.num_particles[None]):
            # Symplectic Euler 적분
            self.p_v[p_i] += self.dt * self.p_a[p_i]
            self.p_x[p_i] += self.dt * self.p_v[p_i]
            
            # 도메인 경계 충돌 (바닥이나 벽에 닿으면 튕겨져 나오도록 반전)
            for d in ti.static(range(3)):
                if self.p_x[p_i][d] < boundary_min:
                    self.p_x[p_i][d] = boundary_min
                    self.p_v[p_i][d] *= -0.5  # 충돌 시 에너지 감쇠
                elif self.p_x[p_i][d] > boundary_max:
                    self.p_x[p_i][d] = boundary_max
                    self.p_v[p_i][d] *= -0.5

    # ------------------------------------------------------------------
    # Taichi GGUI 실시간 시각화
    # ------------------------------------------------------------------
    @ti.kernel
    def update_colors(self):
        half_d = self.domain_size / 2.0
        for i in range(self.num_particles[None]):
            h = (self.p_x[i].y + half_d) / self.domain_size
            h = ti.max(0.0, ti.min(1.0, h))
            self.p_color[i] = ti.Vector([0.1, 0.3 + 0.5 * h, 1.0])

    def run_taichi_viewer(self):
        print("\n[SPH 유체 실시간 뷰어 구동]")
        window = ti.ui.Window("SPH Fluid Simulator", (1280, 720))
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        
        camera.position(0.0, 1.0, self.domain_size * 1.5)
        camera.lookat(0.0, 0.0, 0.0)
        camera.up(0.0, 1.0, 0.0)
        
        paused = False
        
        while window.running:
            if window.get_event(ti.ui.PRESS):
                if window.event.key == ti.ui.SPACE:
                    paused = not paused
                elif window.event.key == 'r':
                    self.setup_initial_fluid_block()
            
            if not paused:
                self.step()
                self.update_colors()
            
            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.ambient_light((0.6, 0.6, 0.6))
            scene.point_light(pos=(2.0, 3.0, 2.0), color=(1.0, 1.0, 1.0))
            
            scene.particles(
                self.p_x,
                radius=self.particle_radius * 1.5,
                per_vertex_color=self.p_color,
                index_count=self.num_particles[None]
            )
            
            canvas.scene(scene)
            window.show()