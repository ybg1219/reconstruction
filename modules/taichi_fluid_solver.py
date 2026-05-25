import taichi as ti
import numpy as np

@ti.data_oriented
class TaichiFluidSolver:
    def __init__(self, res=256, domain_size=2.0, particles = 100000):
        self.res = res
        self.domain_size = domain_size
        
        # 1. 시뮬레이션 기본 물리 상수 설정
        self.dt = 1e-4                # 타임스텝
        self.dx = domain_size / res   # 격자 한 칸의 크기
        self.inv_dx = 1.0 / self.dx
        self.rho = 1.0                # 유체 밀도
        self.gravity = 9.8            # 중력 가속도

        # Lame parameters (유체의 점성 및 탄성 조절)
        self.E = 400.0
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
        positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        
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

    def step(self):
        """
        서브스텝을 여러 번 쪼개어 돌림으로써 고해상도 시뮬레이션의 물리적 안정성을 확보합니다.
        """
        substeps = 20
        for _ in range(substeps):
            self._substep()

    @ti.kernel
    def _substep(self):
        # 1) 격자 초기화
        for i, j, k in self.g_m:
            self.g_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.g_m[i, j, k] = 0.0

        half_d = self.domain_size / 2.0
        p_vol = self.dx ** 3
        p_mass = p_vol * self.rho

        # 2) P2G (Particle to Grid): 파티클의 질량과 운동량을 격자로 전송 및 압력/점성 계산
        for p in range(self.num_particles[None]):
            # 월드 좌표를 격자 인덱스 스페이스로 매핑
            base = ((self.p_x[p] + half_d) * self.inv_dx - 0.5).cast(int)
            fx = (self.p_x[p] + half_d) * self.inv_dx - base.cast(ti.f32)
            
            # Quadratic B-spline 커널 가중치 계산
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            # 유체 연속성 물질 대수 연산 (Neo-Hookean 약화식)
            self.p_F[p] = (ti.Matrix.identity(ti.f32, 3) + self.dt * self.p_C[p]) @ self.p_F[p]
            h = ti.max(0.1, ti.min(5.0, ti.exp(1.0 * (self.p_J[p] - 1.0))))
            if self.p_J[p] < 1.0: h = self.p_J[p]
            
            stress = self.mu_0 * (self.p_F[p] @ self.p_F[p].transpose() - ti.Matrix.identity(ti.f32, 3)) + self.lambda_0 * (self.p_J[p] - 1.0) * h * ti.Matrix.identity(ti.f32, 3)
            eq_stress = -self.dt * p_vol * 4 * self.inv_dx ** 2 * stress
            affine = eq_stress + p_mass * self.p_C[p]

            # 이웃 3x3x3 격자에 분배
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(ti.f32) - fx) * self.dx
                weight = w[i].x * w[j].y * w[k].z
                grid_idx = base + offset
                
                # 경계 조건 방어
                if 0 <= grid_idx.x < self.res and 0 <= grid_idx.y < self.res and 0 <= grid_idx.z < self.res:
                    self.g_v[grid_idx] += weight * (p_mass * self.p_v[p] + affine @ dpos)
                    self.g_m[grid_idx] += weight * p_mass

        # 3) 격자 업데이트 (중력 반영 및 보더 벽면 충돌 처리)
        for i, j, k in self.g_m:
            if self.g_m[i, j, k] > 0:
                self.g_v[i, j, k] /= self.g_m[i, j, k]
                self.g_v[i, j, k].y -= self.gravity * self.dt # 외력: 중력 추가

                # 도메인 외벽 충돌 처리
                boundary = 3
                if i < boundary and self.g_v[i, j, k].x < 0: self.g_v[i, j, k].x = 0
                if i > self.res - boundary and self.g_v[i, j, k].x > 0: self.g_v[i, j, k].x = 0
                if j < boundary and self.g_v[i, j, k].y < 0: self.g_v[i, j, k].y = 0
                if j > self.res - boundary and self.g_v[i, j, k].y > 0: self.g_v[i, j, k].y = 0
                if k < boundary and self.g_v[i, j, k].z < 0: self.g_v[i, j, k].z = 0
                if k > self.res - boundary and self.g_v[i, j, k].z > 0: self.g_v[i, j, k].z = 0

        # 4) G2P (Grid to Particle): 격자 물리량을 다시 파티클로 역주입 및 위치 전진
        for p in range(self.num_particles[None]):
            base = ((self.p_x[p] + half_d) * self.inv_dx - 0.5).cast(int)
            fx = (self.p_x[p] + half_d) * self.inv_dx - base.cast(ti.f32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            new_v = ti.Vector([0.0, 0.0, 0.0])
            new_C = ti.Matrix([ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ])
            
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
            self.p_x[p] += self.dt * self.p_v[p] # 파티클 전진!
            self.p_C[p] = new_C