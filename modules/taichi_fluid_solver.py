import taichi as ti
import numpy as np

@ti.data_oriented
class TaichiFluidSolver:
    def __init__(self, res=256, domain_size=2.0, particles = 200000):
        self.res = res
        self.domain_size = domain_size
        
        # 1. 시뮬레이션 기본 물리 상수 설정
        self.dt = 1e-4                # 타임스텝
        self.dx = domain_size / res   # 격자 한 칸의 크기
        self.inv_dx = 1.0 / self.dx
        self.rho = 1.0                # 유체 밀도
        self.gravity = 9.8            # 중력 가속도

        # Lame parameters (유체의 점성 및 탄성 조절)
        self.E = 1200.0
        self.nu = 0.2
        self.mu_0, self.lambda_0 = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        # 2. 파티클 및 격자 메모리 할당 (Taichi 필드)
        self.max_particles = particles
        self.p_x = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles) # 위치
        self.p_v = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles) # 속도
        self.p_C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.max_particles) # 아핀 속도 필드
        self.p_F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.max_particles) # 변형 구배
        self.p_J = ti.field(dtype=ti.f32, shape=self.max_particles)          # 체적 변화율
        
        # 활성화된 실제 파티클 수 관리
        self.num_particles = ti.field(dtype=ti.i32, shape=())

        # Grid 필드 (물리량 축적용)
        self.g_v = ti.Vector.field(3, dtype=ti.f32, shape=(res, res, res))    # 격자 속도
        self.g_m = ti.field(dtype=ti.f32, shape=(res, res, res))             # 격자 질량
        self.p_color = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)

    def setup_initial_fluid_block(self):
        """
        도메인 내부 정중앙 하단에 유체 덩어리(Dam Break 형태)를 배치합니다.
        """
        # 월드 좌표계 기준 유체 배치 범위 설정 (-domain_size/2 ~ domain_size/2)
        half_d = self.domain_size / 2.0
        
        # 예시: 0.6 x 0.6 x 0.6 크기의 유체 블록 생성
        x_range = np.linspace(-half_d + 0.2, -half_d + 0.8, 35)
        y_range = np.linspace(-half_d + 0.1, -half_d + 0.7, 35)
        z_range = np.linspace(-half_d + 0.2, -half_d + 0.8, 35)

        X, Y, Z = np.meshgrid(x_range, y_range, z_range)
        positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1).astype(np.float32)
        p_count = min(len(positions), self.max_particles)
        self.num_particles[None] = p_count

        # Taichi 외부에서 만든 넘파이 좌표를 내부 필드로 고속 전송
        self._init_particles_kernel(positions[:p_count], p_count)
        print(f"🌊 유체 블록 초기화 완료! 생성된 파티클 수: {p_count:,}개")

    @ti.kernel
    def _init_particles_kernel(self, pos_arr: ti.types.ndarray(), p_count: ti.i32):
        for i in range(p_count):
            self.p_x[i] = ti.Vector([pos_arr[i, 0], pos_arr[i, 1], pos_arr[i, 2]])
            self.p_v[i] = ti.Vector([0.0, 0.0, 0.0])
            self.p_F[i] = ti.Matrix.identity(ti.f32, 3)
            self.p_J[i] = 1.0

    def get_active_particle_count(self):
        return self.num_particles[None]

    @ti.kernel
    def copy_positions_to_field(self, target_field: ti.template()):
        """
        데이터셋 생성 메인 루프에서 파티클 위치를 한 번에 퍼갈 수 있도록 복사합니다.
        """
        for i in range(self.num_particles[None]):
            target_field[i] = self.p_x[i]

    @ti.kernel
    def update_colors(self):
        """
        파티클의 Y축 높이에 따라 심해(짙은 파랑)에서 수면(청록색)으로 
        자연스러운 그라데이션 색상을 부여합니다.
        """
        half_d = self.domain_size / 2.0
        for i in range(self.num_particles[None]):
            # 높이를 0.0 ~ 1.0 사이로 정규화
            h = (self.p_x[i].y + half_d) / self.domain_size
            h = ti.max(0.0, ti.min(1.0, h))
            
            # RGB 색상 결정 (물이 찰랑거리는 느낌)
            self.p_color[i] = ti.Vector([0.1, 0.3 + 0.5 * h, 1.0])
        
    def run_taichi_viewer(self):
        """
        Taichi GGUI를 활용한 초고속 실시간 물리 뷰어를 실행합니다.
        """
        print("\n[Taichi GGUI 실시간 뷰어 구동]")
        print("조작법: 마우스 우클릭 드래그(회전), W/A/S/D(이동), SPACE(일시정지), R(리셋)")
        
        # 1. 윈도우 및 캔버스 초기화
        window = ti.ui.Window("MPM Fluid Simulator (Taichi GGUI)", (1280, 720))
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        
        # 2. 카메라 초기 위치 설정 (도메인 크기에 맞춰 약간 뒤쪽 위에서 바라봄)
        camera.position(0.0, 1.0, self.domain_size * 2.0)
        camera.lookat(0.0, 0.0, 0.0)
        camera.up(0.0, 1.0, 0.0)
        
        paused = False
        
        # 3. 메인 렌더링 무한 루프
        while window.running:
            # 키보드 입력 이벤트 처리
            if window.get_event(ti.ui.PRESS):
                if window.event.key == ti.ui.SPACE:
                    paused = not paused
                elif window.event.key == 'r':
                    self.setup_initial_fluid_block()
            
            # 일시정지 상태가 아닐 때만 물리 연산 진행
            if not paused:
                self.step()
                self.update_colors()
            
            # 마우스 입력에 따른 카메라 위치 업데이트
            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            
            # 조명 설정
            scene.ambient_light((0.6, 0.6, 0.6))
            scene.point_light(pos=(2.0, 3.0, 2.0), color=(1.0, 1.0, 1.0))
            
            # 파티클 렌더링 (가장 핵심적인 GPU 다이렉트 드로우)
            # index_count를 통해 활성화된 파티클 개수만큼만 정확히 그립니다.
            scene.particles(
                self.p_x,
                radius=self.dx * 0.6,
                per_vertex_color=self.p_color,
                index_count=self.num_particles[None]
            )
            
            # 도메인 경계 박스를 그리기 위한 선 추가 (선택 사항)
            # 씬을 캔버스에 적용하고 윈도우 송출
            canvas.scene(scene)
            window.show()

    def step(self):
        """
        시뮬레이션의 안정성과 현실적인 유체 거동을 위해 
        하나의 프레임을 여러 서브스텝으로 나누어 물리 연산을 수행합니다.
        """
        substeps = 150
        for _ in range(substeps):
            self._substep()

    @ti.kernel
    def _substep(self):
        # 1. 그리드 초기화 (Grid Initialization)
        # 매 서브스텝마다 그리드의 속도와 질량을 0으로 초기화합니다.
        for i, j, k in self.g_m:
            self.g_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.g_m[i, j, k] = 0.0

        half_d = self.domain_size / 2.0
        p_vol = self.dx ** 3
        p_mass = p_vol * self.rho

        # 2. 입자에서 그리드로 전송 (Particle to Grid, P2G)
        # 입자의 질량과 운동량을 주변 3x3x3 그리드 노드로 분배합니다.
        for p in range(self.num_particles[None]):
            base = ((self.p_x[p] + half_d) * self.inv_dx - 0.5).cast(int)
            fx = (self.p_x[p] + half_d) * self.inv_dx - base.cast(ti.f32)
            
            # Quadratic B-spline 커널 가중치 계산
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            # 유체 특성 적용: 형태(F) 복원력을 배제하고 부피 변화율(J)만 누적
            self.p_J[p] = (1.0 + self.dt * self.p_C[p].trace()) * self.p_J[p]

            # 상태 방정식 (Equation of State): 부피 변화에 따른 압력 도출
            pressure = self.E * (self.p_J[p] - 1.0)

            # 응력 계산: 고체의 전단 응력을 제거하고 압력과 기본 점성만 적용
            viscosity = 0.01
            stress = ti.Matrix.identity(ti.f32, 3) * pressure + viscosity * self.p_C[p]
            eq_stress = -self.dt * p_vol * 4 * self.inv_dx ** 2 * stress
            affine = eq_stress + p_mass * self.p_C[p]

            # 주변 3x3x3 그리드에 가중치를 적용하여 물리량 누적
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(ti.f32) - fx) * self.dx
                weight = w[i].x * w[j].y * w[k].z
                grid_idx = base + offset
                
                # 도메인 인덱스 경계 방어
                if 0 <= grid_idx.x < self.res and 0 <= grid_idx.y < self.res and 0 <= grid_idx.z < self.res:
                    self.g_v[grid_idx] += weight * (p_mass * self.p_v[p] + affine @ dpos)
                    self.g_m[grid_idx] += weight * p_mass

        # 3. 그리드 속도 업데이트 (외력 적용 및 경계 조건 처리)
        for i, j, k in self.g_m:
            if self.g_m[i, j, k] > 0:
                # 운동량을 질량으로 나누어 속도 도출
                self.g_v[i, j, k] /= self.g_m[i, j, k]
                
                # 중력 적용
                self.g_v[i, j, k].y -= self.gravity * self.dt

                # 도메인 외벽 경계 조건 (Boundary Conditions) 처리
                boundary = 3
                if i < boundary and self.g_v[i, j, k].x < 0: self.g_v[i, j, k].x = 0
                if i > self.res - boundary and self.g_v[i, j, k].x > 0: self.g_v[i, j, k].x = 0
                if j < boundary and self.g_v[i, j, k].y < 0: self.g_v[i, j, k].y = 0
                if j > self.res - boundary and self.g_v[i, j, k].y > 0: self.g_v[i, j, k].y = 0
                if k < boundary and self.g_v[i, j, k].z < 0: self.g_v[i, j, k].z = 0
                if k > self.res - boundary and self.g_v[i, j, k].z > 0: self.g_v[i, j, k].z = 0

        # 4. 그리드에서 입자로 환원 (Grid to Particle, G2P)
        # 그리드의 속도장을 바탕으로 입자의 속도와 위치를 업데이트합니다.
        for p in range(self.num_particles[None]):
            base = ((self.p_x[p] + half_d) * self.inv_dx - 0.5).cast(int)
            fx = (self.p_x[p] + half_d) * self.inv_dx - base.cast(ti.f32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            new_v = ti.Vector([0.0, 0.0, 0.0])
            new_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(ti.f32) - fx) * self.dx
                weight = w[i].x * w[j].y * w[k].z
                grid_idx = base + offset
                
                if 0 <= grid_idx.x < self.res and 0 <= grid_idx.y < self.res and 0 <= grid_idx.z < self.res:
                    g_v_val = self.g_v[grid_idx]
                    new_v += weight * g_v_val
                    new_C += 4 * self.inv_dx ** 2 * weight * g_v_val.outer_product(dpos)

            self.p_v[p] = new_v
            self.p_x[p] += self.dt * self.p_v[p]
            self.p_C[p] = new_C