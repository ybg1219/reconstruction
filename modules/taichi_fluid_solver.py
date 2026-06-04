import taichi as ti
import numpy as np
import math

@ti.data_oriented
class TaichiFluidSolver:
    def __init__(self, res=128, domain_size=2.0, particles=100000):
        self.domain_size = domain_size
        self.max_particles = particles
        
        # 물리 상수 (dt를 살짝 키워서 연산량 감소 - 최적화 포인트 6)
        self.rho0 = 1000.0
        self.stiffness = 500.0
        self.viscosity = 0.05
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        self.dt = 0.005  # 기존 2e-4에서 대폭 상향 (최적화)
        
        self.particle_radius = 0.02
        self.h = self.particle_radius * 2.5
        self.h2 = self.h * self.h
        self.p_mass = self.rho0 * (self.particle_radius * 2)**3 * 0.8
        
        self.poly6_factor = 315.0 / (64.0 * math.pi * self.h**9)
        self.spiky_grad_factor = -45.0 / (math.pi * self.h**6)
        self.visc_lap_factor = 45.0 / (math.pi * self.h**6)

        # -----------------------------------------------------------
        # 🔥 [최적화 1] SNode 메모리 구조 개선 (Flat Array & Dynamic)
        # -----------------------------------------------------------
        self.grid_size = self.h
        self.grid_res = int(np.ceil(domain_size / self.grid_size))
        self.max_p_per_cell = 200
        
        # GPU 캐시 히트율을 극대화하기 위해 다차원 배열을 1D 형태로 펴서 메모리 할당 (SoA 구조 유지)
        self.grid_count = ti.field(dtype=ti.i32, shape=(self.grid_res * self.grid_res * self.grid_res))
        self.grid_particles = ti.field(dtype=ti.i32, shape=(self.grid_res * self.grid_res * self.grid_res, self.max_p_per_cell))

        # 파티클 데이터 (SoA - Structure of Arrays)
        self.num_particles = ti.field(dtype=ti.i32, shape=())
        self.p_x = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)
        self.p_v = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)
        # self.p_a = ti.Vector.field(...) # 🔥 [최적화 3] 커널 퓨전으로 인해 가속도(a)를 메모리에 저장할 필요가 없어짐! (메모리 절약)
        self.p_rho = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.p_press = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.p_color = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)

    def setup_initial_fluid_block(self):
        half_d = self.domain_size / 2.0
        spacing = self.particle_radius * 2.0
        x_range = np.arange(-half_d + 0.2, -half_d + 0.8, spacing)
        y_range = np.arange(-half_d + 0.1, -half_d + 0.8, spacing)
        z_range = np.arange(-half_d + 0.2, -half_d + 0.8, spacing)

        X, Y, Z = np.meshgrid(x_range, y_range, z_range)
        positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1).astype(np.float32)
        
        p_count = min(len(positions), self.max_particles)
        self.num_particles[None] = p_count

        self._init_particles_kernel(positions[:p_count], p_count)
        print(f"🌊 최적화 SPH 유체 블록 초기화 완료! 파티클 수: {p_count:,}개")

    @ti.kernel
    def _init_particles_kernel(self, pos_arr: ti.types.ndarray(), p_count: ti.i32):
        ti.loop_config(block_dim=128)
        for i in range(p_count):
            self.p_x[i] = ti.Vector([pos_arr[i, 0], pos_arr[i, 1], pos_arr[i, 2]])
            self.p_v[i] = ti.Vector([0.0, 0.0, 0.0])
            self.p_rho[i] = self.rho0
            self.p_press[i] = 0.0
            self.p_color[i] = ti.Vector([0.1, 0.6, 1.0])

    def get_active_particle_count(self):
        return self.num_particles[None]

    # -----------------------------------------------------------
    # 🔥 [최적화 공통] Grid 3D 인덱스를 1D 인덱스로 변환하는 헬퍼 함수
    # -----------------------------------------------------------
    @ti.func
    def get_grid_idx(self, pos):
        half_d = self.domain_size / 2.0
        c = ti.cast((pos + half_d) / self.grid_size, ti.i32)
        return c

    @ti.func
    def flatten_grid_idx(self, c):
        return c.x * self.grid_res * self.grid_res + c.y * self.grid_res + c.z

    @ti.func
    def is_valid_cell(self, c):
        return 0 <= c.x < self.grid_res and 0 <= c.y < self.grid_res and 0 <= c.z < self.grid_res

    # ------------------------------------------------------------------
    # SPH 메인 파이프라인
    # ------------------------------------------------------------------
    def step(self):
        # dt가 커졌으므로 서브스텝 수를 확 줄입니다. (성능 폭발적 향상)
        substeps = 5 
        for _ in range(substeps):
            self.update_grid()
            self.compute_density_pressure()
            self.compute_forces_and_integrate() # 🔥 [최적화 3] 커널 병합됨!

    @ti.kernel
    def update_grid(self):
        ti.loop_config(block_dim=256)
        # 1. 1D Flat 격자 초기화
        for i in self.grid_count:
            self.grid_count[i] = 0
            
        # 2. 파티클 해싱
        for p in range(self.num_particles[None]):
            cell = self.get_grid_idx(self.p_x[p])
            if self.is_valid_cell(cell):
                flat_idx = self.flatten_grid_idx(cell)
                # 🔥 [최적화 5] atomic_add는 어쩔 수 없지만 1D 배열이라 훨씬 빠름
                idx = ti.atomic_add(self.grid_count[flat_idx], 1)
                if idx < self.max_p_per_cell:
                    self.grid_particles[flat_idx, idx] = p

    @ti.kernel
    def compute_density_pressure(self):
        ti.loop_config(block_dim=128) # 🔥 [최적화 7] CUDA Thread Block 튜닝
        for p_i in range(self.num_particles[None]):
            pos_i = self.p_x[p_i] # 🔥 [최적화 4] Local Cache (Global memory 접근 최소화)
            cell = self.get_grid_idx(pos_i)
            density = 0.0
            
            for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                neighbor_cell = cell + offset
                if self.is_valid_cell(neighbor_cell):
                    flat_idx = self.flatten_grid_idx(neighbor_cell)
                    count = ti.min(self.grid_count[flat_idx], self.max_p_per_cell)
                    for j in range(count):
                        p_j = self.grid_particles[flat_idx, j]
                        r_vec = pos_i - self.p_x[p_j]
                        
                        # 🔥 [최적화 2] sqrt 제거! norm_sqr (r^2) 만으로 범위 판별
                        r2 = r_vec.norm_sqr()
                        if r2 < self.h2 and r2 > 1e-10:
                            h2_r2 = self.h2 - r2
                            density += self.p_mass * self.poly6_factor * (h2_r2 * h2_r2 * h2_r2)
                        elif r2 <= 1e-10:
                            density += self.p_mass * self.poly6_factor * (self.h2 * self.h2 * self.h2)
            
            self.p_rho[p_i] = density
            self.p_press[p_i] = self.stiffness * ti.max(density / self.rho0 - 1.0, 0.0)

    # 🔥 [최적화 3] Force 계산과 Integrate를 하나의 커널로 합침 (Kernel Fusion)
    @ti.kernel
    def compute_forces_and_integrate(self):
        ti.loop_config(block_dim=128)
        
        half_d = self.domain_size / 2.0
        boundary_min = -half_d + self.particle_radius
        boundary_max = half_d - self.particle_radius

        for p_i in range(self.num_particles[None]):
            # 🔥 [최적화 4] Local Cache 적극 활용
            pos_i = self.p_x[p_i]
            vel_i = self.p_v[p_i]
            rho_i = self.p_rho[p_i]
            press_i = self.p_press[p_i]
            
            f_press = ti.Vector([0.0, 0.0, 0.0])
            f_visc = ti.Vector([0.0, 0.0, 0.0])
            
            cell = self.get_grid_idx(pos_i)
            for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                neighbor_cell = cell + offset
                if self.is_valid_cell(neighbor_cell):
                    flat_idx = self.flatten_grid_idx(neighbor_cell)
                    count = ti.min(self.grid_count[flat_idx], self.max_p_per_cell)
                    for j in range(count):
                        p_j = self.grid_particles[flat_idx, j]
                        if p_i != p_j:
                            r_vec = pos_i - self.p_x[p_j]
                            r2 = r_vec.norm_sqr()
                            
                            # 🔥 [최적화 2] 거리(r)가 정말 필요한 Spiky/Visc 계산 내부에서만 딱 한 번 sqrt 호출
                            if r2 < self.h2 and r2 > 1e-10:
                                r = ti.math.sqrt(r2) 
                                rho_j = self.p_rho[p_j]
                                press_j = self.p_press[p_j]
                                vel_j = self.p_v[p_j]
                                
                                h_minus_r = self.h - r
                                grad_w = self.spiky_grad_factor * (h_minus_r * h_minus_r) * (r_vec / r)
                                p_term = (press_i / (rho_i * rho_i + 1e-6)) + (press_j / (rho_j * rho_j + 1e-6))
                                f_press += -self.p_mass * p_term * grad_w
                                
                                lap_w = self.visc_lap_factor * h_minus_r
                                v_term = (vel_j - vel_i) / (rho_j + 1e-6)
                                f_visc += self.viscosity * self.p_mass * v_term * lap_w
                                
            # 1. 가속도(a) 도출
            a_i = f_press + f_visc + self.gravity
            
            # 2. 속도(v) 업데이트 (Integrate)
            vel_i += self.dt * a_i
            
            # 3. 위치(x) 업데이트 (Integrate)
            pos_i += self.dt * vel_i
            
            # 4. 경계 충돌 처리
            for d in ti.static(range(3)):
                if pos_i[d] < boundary_min:
                    pos_i[d] = boundary_min
                    vel_i[d] *= -0.5 
                elif pos_i[d] > boundary_max:
                    pos_i[d] = boundary_max
                    vel_i[d] *= -0.5
            
            # 5. 최종 값을 Global Memory에 단 한 번만 기록! (병목 킬러)
            self.p_v[p_i] = vel_i
            self.p_x[p_i] = pos_i

    @ti.kernel
    def update_colors(self):
        ti.loop_config(block_dim=256)
        half_d = self.domain_size / 2.0
        for i in range(self.num_particles[None]):
            h = (self.p_x[i].y + half_d) / self.domain_size
            h = ti.max(0.0, ti.min(1.0, h))
            self.p_color[i] = ti.Vector([0.1, 0.3 + 0.5 * h, 1.0])

    def run_taichi_viewer(self):
        print("\n[초고속 튜닝 SPH 뷰어 구동]")
        window = ti.ui.Window("Extreme SPH Fluid", (1280, 720))
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        
        camera.position(0.0, 1.0, self.domain_size * 1.5)
        camera.lookat(0.0, 0.0, 0.0)
        camera.up(0.0, 1.0, 0.0)
        
        paused = False
        frame_count = 0
        
        while window.running:
            if window.get_event(ti.ui.PRESS):
                if window.event.key == ti.ui.SPACE:
                    paused = not paused
                elif window.event.key == 'r':
                    self.setup_initial_fluid_block()
            
            if not paused:
                self.step()
                self.update_colors()
            
            # 🔥 [최적화 9] Visualization 병목 최소화
            # 매 프레임 그리지 않고 씬 업데이트는 그대로 두되 물리 연산 비중을 늘림
            # (위의 self.step() 내부 substep 개수를 조절하는 것으로 이미 해결됨)

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