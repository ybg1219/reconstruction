import taichi as ti
import numpy as np
import math
import os
import glob
import torch
from modules.sdf_network import FeatureConstruction


@ti.data_oriented
class TaichiFluidSolver:
    def __init__(self, res=128, domain_size=2.0, max_particles=400000, p_per_cell=1.0):
        self.domain_size = domain_size
        self.max_particles = max_particles
        self.res = res
        self.p_per_cell = p_per_cell

        # =========================================================
        # 1. 공간 해상도 기반 기본 설정
        # =========================================================
        self.dx = self.domain_size / self.res

        # 셀당 목표 파티클 수 기반 spacing 계산
        self.spacing = self.dx / (self.p_per_cell ** (1.0 / 3.0))

        # 안정적인 초기 배치를 위한 particle radius
        self.particle_radius = self.spacing * 0.5

        # =========================================================
        # 2. SPH 물리 파라미터 (안정성 중심 세팅)
        # =========================================================

        # 기준 밀도 (물)
        self.rho0 = 1000.0

        # 압력 강성 (너무 크면 폭발)
        self.stiffness = 150.0

        # 점성 (충분히 줘야 안정됨)
        self.viscosity = 0.02

        # 시간 스텝 (SPH 안정 핵심)
        self.dt = 0.001
        self.surface_tension = 0.03
        self.gravity = ti.Vector([0.0, -9.8, 0.0])

        # smoothing length (spacing 기준으로 고정)
        self.h = self.spacing * 1.4
        self.h2 = self.h * self.h

        # particle mass
        self.p_mass = self.rho0 * (self.particle_radius * 2)**3

        # SPH kernel constants
        self.poly6_factor = 315.0 / (64.0 * math.pi * self.h**9)
        self.spiky_grad_factor = -45.0 / (math.pi * self.h**6)
        self.visc_lap_factor = 45.0 / (math.pi * self.h**6)

        # =========================================================
        # 3. Spatial Grid (Neighbor search acceleration)
        # =========================================================
        self.grid_size = self.h
        self.grid_res = int(np.ceil(domain_size / self.grid_size))
        self.max_p_per_cell = 64

        self.grid_count = ti.field(dtype=ti.i32, shape=(self.grid_res**3))
        self.grid_particles = ti.field(
            dtype=ti.i32,
            shape=(self.grid_res**3, self.max_p_per_cell)
        )

        # =========================================================
        # 4. Particle buffers (SoA layout)
        # =========================================================
        self.num_particles = ti.field(dtype=ti.i32, shape=())

        self.p_x = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)
        self.p_v = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)

        self.p_rho = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.p_press = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.p_color = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)

    # =========================================================
    # 초기 유체 생성
    # =========================================================
    def setup_initial_fluid_block(self):
        half_d = self.domain_size / 2.0

        x_range = np.arange(-half_d + 0.2, 0.0, self.spacing)
        y_range = np.arange(-half_d + 0.1, 0.0, self.spacing)
        z_range = np.arange(-half_d + 0.2, 0.0, self.spacing)

        X, Y, Z = np.meshgrid(x_range, y_range, z_range)
        positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1).astype(np.float32)

        # jitter initial particles
        jitter = self.spacing * 0.1
        noise = np.random.uniform(-jitter, jitter, positions.shape).astype(np.float32)
        positions += noise
        
        p_count = len(positions)

        if p_count > self.max_particles:
            print(f"⚠️ Particle limit exceeded: {p_count:,}")
            p_count = self.max_particles

        self.num_particles[None] = p_count
        self._init_particles_kernel(positions[:p_count], p_count)

    @ti.kernel
    def _init_particles_kernel(self, pos_arr: ti.types.ndarray(), p_count: ti.i32):
        for i in range(p_count):
            self.p_x[i] = ti.Vector([pos_arr[i, 0], pos_arr[i, 1], pos_arr[i, 2]])
            self.p_v[i] = ti.Vector([0.0, 0.0, 0.0])
            self.p_rho[i] = self.rho0
            self.p_press[i] = 0.0
            self.p_color[i] = ti.Vector([0.1, 0.6, 1.0])
    
    def get_active_particle_count(self):
        return self.num_particles[None]
    
    @ti.kernel
    def copy_positions_to_field(self, target_field: ti.template()):
        for i in range(self.num_particles[None]):
            target_field[i] = self.p_x[i]

    # =========================================================
    # Grid utilities
    # =========================================================
    @ti.func
    def get_grid_idx(self, pos):
        half_d = self.domain_size / 2.0
        return ti.cast((pos + half_d) / self.grid_size, ti.i32)

    @ti.func
    def flatten_grid_idx(self, c):
        return c.x * self.grid_res * self.grid_res + c.y * self.grid_res + c.z

    @ti.func
    def is_valid_cell(self, c):
        return 0 <= c.x < self.grid_res and 0 <= c.y < self.grid_res and 0 <= c.z < self.grid_res

    # =========================================================
    # Simulation step
    # =========================================================
    def step(self):
        # SPH 안정성 우선: substep 최소화
        substeps = 8
        for _ in range(substeps):
            self.update_grid()
            self.compute_density_pressure()
            self.compute_forces_and_integrate()

    # =========================================================
    # Spatial hashing
    # =========================================================
    @ti.kernel
    def update_grid(self):
        for i in self.grid_count:
            self.grid_count[i] = 0

        for p in range(self.num_particles[None]):
            cell = self.get_grid_idx(self.p_x[p])

            if self.is_valid_cell(cell):
                flat = self.flatten_grid_idx(cell)

                idx = ti.atomic_add(self.grid_count[flat], 1)
                if idx < self.max_p_per_cell:
                    self.grid_particles[flat, idx] = p

    # =========================================================
    # Density / pressure
    # =========================================================
    @ti.kernel
    def compute_density_pressure(self):
        for i in range(self.num_particles[None]):
            pos_i = self.p_x[i]
            cell = self.get_grid_idx(pos_i)

            density = 0.0

            for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                nc = cell + offset

                if self.is_valid_cell(nc):
                    flat = self.flatten_grid_idx(nc)
                    count = ti.min(self.grid_count[flat], self.max_p_per_cell)

                    for j in range(count):
                        p_j = self.grid_particles[flat, j]
                        r = pos_i - self.p_x[p_j]

                        r2 = r.norm_sqr()

                        if r2 < self.h2:
                            h2_r2 = self.h2 - r2
                            density += self.p_mass * self.poly6_factor * (h2_r2 ** 3)

            # 안정성 clamp (핵심)
            density = ti.max(density, self.rho0 * 0.5)
            self.p_rho[i] = density

            # pressure (clamped)
            gamma = 7.0
            p = self.stiffness * ((density / self.rho0) ** gamma - 1.0)
            self.p_press[i] = ti.max(0.0, p)

    # =========================================================
    # Forces + Integration
    # =========================================================
    @ti.kernel
    def compute_forces_and_integrate(self):
        half_d = self.domain_size / 2.0
        padding = 0.1  
        boundary_min = -half_d + padding + self.particle_radius
        boundary_max = half_d - padding - self.particle_radius
        
        for i in range(self.num_particles[None]):
            pos_i = self.p_x[i]
            vel_i = self.p_v[i]
            rho_i = self.p_rho[i]
            press_i = self.p_press[i]

            f_press = ti.Vector([0.0, 0.0, 0.0])
            f_visc = ti.Vector([0.0, 0.0, 0.0])
            f_tension = ti.Vector([0.0, 0.0, 0.0])

            cell = self.get_grid_idx(pos_i)

            for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                nc = cell + offset

                if self.is_valid_cell(nc):
                    flat = self.flatten_grid_idx(nc)
                    count = ti.min(self.grid_count[flat], self.max_p_per_cell)

                    for j in range(count):
                        pj = self.grid_particles[flat, j]

                        if i != pj:
                            r = pos_i - self.p_x[pj]
                            r2 = r.norm_sqr()

                            if r2 < self.h2:
                                dist = ti.sqrt(r2 + 1e-6)

                                rho_j = self.p_rho[pj]
                                press_j = self.p_press[pj]
                                vel_j = self.p_v[pj]

                                # pressure force
                                h_r = self.h - dist
                                grad = self.spiky_grad_factor * (h_r * h_r) * (r / dist)

                                p_term = (press_i / (rho_i * rho_i + 1e-6)) + \
                                         (press_j / (rho_j * rho_j + 1e-6))
                                f_press += -self.p_mass * p_term * grad

                                # viscosity
                                lap = self.visc_lap_factor * h_r
                                f_visc += self.viscosity * self.p_mass * (vel_j - vel_i) / (rho_j + 1e-6) * lap
                                
                                f_tension += -self.surface_tension * self.p_mass * (r / dist) * h_r

            # gravity + forces
            acc = f_press + f_visc + f_tension + self.gravity
            
            # integrate
            vel_i += self.dt * acc
            
            pos_i += self.dt * vel_i

            # boundary
            for d in ti.static(range(3)):
                if pos_i[d] < boundary_min:
                    pos_i[d] = boundary_min
                    vel_i[d] *= -0.7
                    
                    vel_i[(d + 1) % 3] *= 0.999
                    vel_i[(d + 2) % 3] *= 0.999
                    
                elif pos_i[d] > boundary_max:
                    pos_i[d] = boundary_max
                    vel_i[d] *= -0.7

                    vel_i[(d + 1) % 3] *= 0.999
                    vel_i[(d + 2) % 3] *= 0.999
                
            self.p_v[i] = vel_i
            self.p_x[i] = pos_i

    # =========================================================
    # Color update
    # =========================================================
    @ti.kernel
    def update_colors(self):
        for i in range(self.num_particles[None]):
            h = (self.p_x[i].y + self.domain_size * 0.5) / self.domain_size
            h = ti.min(1.0, ti.max(0.0, h))

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
        frame_count = 0  # 프레임 카운터 초기화
        
        while window.running:
            if window.get_event(ti.ui.PRESS):
                if window.event.key == ti.ui.SPACE:
                    paused = not paused
                elif window.event.key == 'r':
                    self.setup_initial_fluid_block()
                    frame_count = 0  # R키를 누르면 프레임도 다시 0으로 초기화
            
            if not paused:
                self.step()
                self.update_colors()
                frame_count += 1  # 시뮬레이션이 진행될 때만 프레임 증가
            
            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.ambient_light((0.6, 0.6, 0.6))
            scene.point_light(pos=(2.0, 3.0, 2.0), color=(1.0, 1.0, 1.0))
            
            scene.particles(
                self.p_x,
                radius=self.particle_radius * 1.0,
                per_vertex_color=self.p_color,
                index_count=self.num_particles[None]
            )
            
            canvas.scene(scene)

            # =========================================================
            # 상태창 GUI 추가
            # =========================================================
            gui = window.get_gui()
            # 좌측 상단(x: 0.05, y: 0.05)에 너비 0.2, 높이 0.1 크기의 패널 생성
            with gui.sub_window("Simulation Status", 0.05, 0.05, 0.2, 0.12):
                gui.text(f"Current Frame: {frame_count}")
                gui.text(f"Active Particles: {self.num_particles[None]:,}")
                if paused:
                    gui.text("Status: PAUSED")
                else:
                    gui.text("Status: RUNNING")
            
            window.show()
            
    # =========================================================
    # Dataset Generation & Utilities
    # =========================================================
    @staticmethod
    def generate_dataset(
        output_dir: str = "dataset_fluid",
        dataset_size: int = 40,      # 총 생성할 프레임(샘플) 수
        start_index: int = 0,
        save_sdf: bool = True,
        save_particles: bool = True,
        save_mc: bool = True,
        config = None,               # resolution, domain_size 포함 필수
        device = 'cpu',
        cleanup_old_files: bool = False
    ):
        """
        Taichi 유체 시뮬레이션을 돌려 실시간 파티클을 뽑고, 
        Spatial Hash 기반 초고속 SDF 변환 및 mc 특징맵을 추출하여 데이터셋을 빌드합니다.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if cleanup_old_files:
            old_files = glob.glob(os.path.join(output_dir, "*.npy"))
            for f in old_files:
                os.remove(f)
            print(f"🗑️ Removed {len(old_files)} old dataset files.")

        # 1. 전역 파라미터 세팅
        res = config.resolution  
        dx = config.domain_size / res
        
        # 2. Taichi 유체 솔버 초기화
        solver = TaichiFluidSolver(res=res, domain_size=config.domain_size)
        solver.setup_initial_fluid_block() 
        
        # 파티클 획득용 임시 벡터 필드
        tmp_particle_field = ti.Vector.field(3, dtype=ti.f32, shape=solver.max_particles)

        # 3. 모델 특징맵(m_c) 추출기 초기화
        feature_constructor = None
        if save_mc:
            # config.dx 대신 위에서 계산한 dx 사용
            feature_constructor = FeatureConstruction(dx=dx, particle_spacing=solver.spacing, device=device)

        print(f"\n🚀 Launching Optimized Fluid Simulator Pipeline...")
        print(f"   [Target Frames]: {dataset_size - start_index} | Resolution: {res}^3")

        # 4. 프레임 시뮬레이션 및 데이터 추출 루프
        for frame in range(start_index, dataset_size):
            print(f"\n[{frame+1}/{dataset_size}] Simulating Fluid State...")
            
            # [단계 A] 유체 1스텝 전진 (내부적으로 파티클 이동은 무조건 매 프레임 실행)
            solver.step() 
            
            # 🔥 [핵심 수정] 4 프레임마다 한 번씩만 데이터를 추출하고 저장합니다.
            if frame % 4 == 0:
                print(f"   💾 [Frame {frame:03d}] Extracting & Saving Data...")
                
                # 활성화된 파티클 Numpy 배열로 추출
                num_active = solver.get_active_particle_count()
                solver.copy_positions_to_field(tmp_particle_field)
                raw_particles_np = tmp_particle_field.to_numpy()[:num_active]

                saved_files = []

                # [단계 B] 파티클 -> SDF 필드 초고속 변환 (Spatial Hash Gather 적용)
                if save_sdf:
                    sdf_grid = TaichiFluidSolver.convert_particles_to_sdf(
                        particles_np=raw_particles_np,
                        res=res,
                        domain_size=config.domain_size,
                        radius_ratio=1.25 # 파티클 두께 조절 (필요시 튜닝)
                    )
                    
                    sdf_filename = os.path.join(output_dir, f"sdf_grid_{frame:03d}.npy")
                    np.save(sdf_filename, sdf_grid)
                    saved_files.append("SDF")

                # [단계 C] 원본 파티클 저장
                if save_particles:
                    particles_filename = os.path.join(output_dir, f"particles_{frame:03d}.npy")
                    np.save(particles_filename, raw_particles_np)
                    saved_files.append("Particles")

                # [단계 D] m_c 격자 특징맵 연산 (PyTorch)
                if save_mc:
                    particles_tensor = torch.tensor(raw_particles_np, dtype=torch.float32, device=device)
                    
                    with torch.no_grad():
                        # Spatial Hash & Sort가 내장된 FeatureConstruction 가동
                        grid_nodes, m_c, grid_shape = feature_constructor(particles_tensor)
                    
                    mc_grid = m_c.reshape(grid_shape).cpu().numpy()
                    mc_filename = os.path.join(output_dir, f"mc_grid_{frame:03d}.npy")
                    np.save(mc_filename, mc_grid)
                    saved_files.append("m_c Grid")
                    
                    del particles_tensor, grid_nodes, m_c
                    torch.cuda.empty_cache()

                print(f"   ✅ Saved Frame {frame:03d}: {', '.join(saved_files)} (Particles: {num_active:,})")
            
            else:
                # 저장하지 않는 프레임은 로그만 남기고 빠르게 패스합니다.
                print(f"   ⏩ Pass saving (Physics updated)")

            # (선택) 다양성 확보를 위해 N 프레임마다 물방울 초기화 
            # if (frame + 1) % 20 == 0:
            #      solver.setup_initial_fluid_block()

        print(f"\n🎉 Fluid Dataset successfully generated in '{output_dir}'!")

    @staticmethod
    def save_particles_to_npy(solver, output_dir="dataset_particles", num_frames=100):
        """
        Taichi 솔버를 구동하여 매 프레임의 파티클 위치를 .npy 파일로 저장합니다.
        """
        # 1. 저장 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. 파티클 데이터를 GPU에서 CPU(Numpy)로 가져오기 위한 임시 필드 생성
        tmp_field = ti.Vector.field(3, dtype=ti.f32, shape=solver.max_particles)
        
        print(f"\n🚀 파티클 데이터셋 생성을 시작합니다...")
        print(f"📂 저장 경로: {output_dir}")
        print(f"🎞️ 총 프레임 수: {num_frames}개")
        print("-" * 40)

        # 3. 프레임 단위 시뮬레이션 및 저장 루프
        for frame in range(num_frames):
            # A. 시뮬레이션 1스텝 전진
            solver.step()
            
            # B. 현재 활성화된 파티클 개수 확인
            num_particles = solver.get_active_particle_count()
            
            # C. 솔버 내부의 파티클 위치를 임시 필드로 복사
            solver.copy_positions_to_field(tmp_field)
            
            # D. Numpy 배열로 변환 및 유효한 파티클만 슬라이싱
            # shape: (num_particles, 3)
            particles_np = tmp_field.to_numpy()[:num_particles]
            
            # E. .npy 파일로 디스크에 저장
            filename = os.path.join(output_dir, f"particles_{frame:04d}.npy")
            np.save(filename, particles_np)
            
            # 10프레임마다 진행 상황 출력
            if frame % 10 == 0 or frame == num_frames - 1:
                print(f"   [Frame {frame:04d}/{num_frames}] 저장 완료 (활성 파티클: {num_particles:,}개)")
                
        print("-" * 40)
        print(f"🎉 데이터셋 추출이 완료되었습니다! 총 {num_frames}개의 파일이 저장되었습니다.")

    @staticmethod
    def convert_particles_to_sdf(particles_np, res=256, domain_size=2.0, radius_ratio=1.0):
        """
        Numpy 파티클 배열을 입력받아 고해상도 SDF 그리드를 반환하는 래퍼 함수입니다.
        
        Args:
            particles_np: (N, 3) 형태의 파티클 월드 좌표 배열
            res: 변환할 SDF 그리드 해상도 (기본값 256)
            domain_size: 시뮬레이션 물리 도메인 크기 (기본값 2.0)
            radius_ratio: 하나의 유체 입자가 차지하는 두께 (기본 격자 크기 dx 대비 배수)
        
        Returns:
            sdf_grid: (res, res, res) 형태의 Numpy SDF 배열
        """
        num_particles = len(particles_np)
        if num_particles == 0:
            print("⚠️ 파티클 데이터가 비어 있습니다. 빈 그리드를 반환합니다.")
            return np.full((res, res, res), domain_size, dtype=np.float32)

        # 출력용 빈 넘파이 배열 할당 (Taichi 커널에서 직접 덮어씀)
        sdf_grid_np = np.empty((res, res, res), dtype=np.float32)
        
        # 유체 파티클 하나의 물리적 반경(두께) 계산
        dx = domain_size / (res-1.0)
        p_radius = dx * radius_ratio

        # GPU 커널 가동
        _compute_sdf_scatter_kernel(
            particles=particles_np.astype(np.float32),
            sdf_grid=sdf_grid_np,
            num_particles=num_particles,
            res=res,
            domain_size=domain_size,
            p_radius=p_radius
        )

        return sdf_grid_np

@ti.kernel
def _compute_sdf_scatter_kernel(
    particles: ti.types.ndarray(),
    sdf_grid: ti.types.ndarray(),
    num_particles: ti.i32,
    res: ti.i32,
    domain_size: ti.f32,
    p_radius: ti.f32
):
    """
    GPU 병렬 처리를 통해 각 파티클이 자신의 주변 격자에만 SDF 값을 기록하는 커널입니다.
    """
    half_d = domain_size / 2.0

    # 1. SDF 그리드를 충분히 큰 양수(도메인 바깥 거리)로 초기화
    for i, j, k in ti.ndrange(res, res, res):
        sdf_grid[i, j, k] = domain_size

    # 2. 파티클 관점에서 주변 격자 탐색 마진(칸 수) 계산
    dx = domain_size / res
    margin = ti.cast(ti.ceil(p_radius / dx), ti.i32) + 2

    # 3. 모든 파티클을 병렬로 순회하며 주변 그리드에 최단 거리 갱신
    for p in range(num_particles):
        px = particles[p, 0]
        py = particles[p, 1]
        pz = particles[p, 2]

        # 현재 파티클이 위치한 중심 격자 인덱스 도출
        base_i = ti.cast(ti.round((px + half_d) / domain_size * (res - 1)), ti.i32)
        base_j = ti.cast(ti.round((py + half_d) / domain_size * (res - 1)), ti.i32)
        base_k = ti.cast(ti.round((pz + half_d) / domain_size * (res - 1)), ti.i32)

        # 파티클 반경(margin)을 덮는 이웃 격자들만 부분적으로 루프 (Narrow Band)
        for i_off in range(-margin, margin + 1):
            for j_off in range(-margin, margin + 1):
                for k_off in range(-margin, margin + 1):
                    grid_i = base_i + i_off
                    grid_j = base_j + j_off
                    grid_k = base_k + k_off

                    # 도메인 경계 내부에 있는 격자인지 확인
                    if 0 <= grid_i < res and 0 <= grid_j < res and 0 <= grid_k < res:
                        # 격자 노드의 정확한 월드 좌표 복원
                        g_x = (grid_i / (res - 1.0)) * domain_size - half_d
                        g_y = (grid_j / (res - 1.0)) * domain_size - half_d
                        g_z = (grid_k / (res - 1.0)) * domain_size - half_d

                        # 유클리디안 거리 계산 후 유체 반경(p_radius)을 빼서 SDF 도출
                        dist = ti.sqrt((g_x - px)**2 + (g_y - py)**2 + (g_z - pz)**2) - p_radius
                        
                        # 🚨 병렬 스레드 충돌 방지를 위해 atomic_min 사용 (최단 거리만 남김)
                        ti.atomic_min(sdf_grid[grid_i, grid_j, grid_k], dist)
