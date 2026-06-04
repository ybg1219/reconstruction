import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import taichi as ti

class SDFDataset(Dataset):
    """
    미리 계산된 SDF 그리드와 m_c 특징 그리드(.npy) 쌍을 불러와 
    8x8x8 슬라이딩 윈도우 패치(Patch) 단위로 분할하여 제공하는 학습 데이터셋.
    """
    # 🚨 [수정됨] feature_constructor 파라미터 추가
    def __init__(self, data_dir="dataset", patch_size=8, in_memory=True, 
                 use_narrow_band=True, use_bg_sample=True, bg_sample_ratio=0.05,
                 feature_constructor=None):
        super().__init__()
        self.patch_size = patch_size
        self.half_size = patch_size // 2
        self.in_memory = in_memory
        
        self.use_narrow_band = use_narrow_band  
        self.use_bg_sample = use_bg_sample
        self.bg_sample_ratio = bg_sample_ratio
        
        self.sdf_files = sorted(glob.glob(os.path.join(data_dir, "sdf_grid*.npy")))
        self.mc_files = sorted(glob.glob(os.path.join(data_dir, "mc_grid*.npy")))
        # 🚨 [추가됨] 파티클 파일도 검색합니다.
        self.particle_files = sorted(glob.glob(os.path.join(data_dir, "particles*.npy")))
        
        if len(self.sdf_files) == 0:
            print(f"⚠️ 경고: '{data_dir}' 경로에서 데이터셋 파일을 찾지 못했습니다.")
            self.num_shapes, self.total_samples = 0, 0
            return
            
        # =========================================================
        # 🚨 유연성 확보: mc_grid가 없고 particles만 있을 경우 자동 변환 및 저장
        # =========================================================
        if len(self.mc_files) == 0 and len(self.particle_files) > 0:
            if feature_constructor is None:
                raise ValueError("❌ mc_grid 파일이 없습니다. particles에서 자동 생성하려면 feature_constructor를 전달해주세요.")
            
            print(f"🔄 mc_grid 파일이 없어 particles 데이터로부터 자동 생성을 시작합니다...")
            for p_file in self.particle_files:
                particles_np = np.load(p_file)
                particles_tensor = torch.tensor(particles_np, dtype=torch.float32, device=feature_constructor.device)
                
                # 특징맵 계산
                with torch.no_grad():
                    _, m_c, grid_shape = feature_constructor(particles_tensor)
                mc_grid = m_c.reshape(grid_shape).cpu().numpy()
                
                # 다음 번 로드를 위해 파일로 저장 (particles_000.npy -> mc_grid_000.npy)
                base_name = os.path.basename(p_file).replace("particles_", "mc_grid_")
                mc_filename = os.path.join(data_dir, base_name)
                
                np.save(mc_filename, mc_grid)
                self.mc_files.append(mc_filename)
                print(f"  -> 변환 및 저장 완료: {mc_filename}")
            print("✅ mc_grid 자동 생성 완료!\n")
        # =========================================================

        self.num_shapes = min(len(self.sdf_files), len(self.mc_files))
        
        sample_grid = np.load(self.sdf_files[0])
        self.grid_shape = sample_grid.shape 
        self.num_nodes_per_shape = np.prod(self.grid_shape)
        
        self.sdf_data = []
        self.mc_data = []
        
        # 1. 데이터 캐싱
        if self.in_memory:
            print("💾 데이터를 메모리에 캐싱 중입니다...")
            for i in range(self.num_shapes):
                self.sdf_data.append(np.load(self.sdf_files[i]))
                self.mc_data.append(np.load(self.mc_files[i]))
            print("✅ 캐싱 완료.")

        # 2. 파라미터에 따라 Narrow Band 필터링 분기 처리
        if self.use_narrow_band:
            self._apply_narrow_band_filtering()
        else:
            self.total_samples = self.num_shapes * self.num_nodes_per_shape
            print(f"✅ 전체 노드 학습 모드 가동: 총 {self.total_samples}개 샘플 (필터링 안 함)")

    def _apply_narrow_band_filtering(self):
        """
        표면 근처 노드를 필터링하고, 옵션에 따라 배경 노드를 섞어주는 초고속 함수 (NumPy 벡터 연산 사용)
        """
        self.valid_samples = [] 
        dx = 1.0 / self.grid_shape[0]
        narrow_band_threshold = 4.0 * dx  
        
        mode_str = f"표면(|SDF| <= {narrow_band_threshold:.4f})"
        if self.use_bg_sample:
            mode_str += f" + 배경({self.bg_sample_ratio*100}%) 혼합"
        
        print(f"🔍 [초고속 필터링] {mode_str} 샘플링 중...")
        
        total_surface = 0
        total_bg = 0
        
        for i in range(self.num_shapes):
            if self.in_memory:
                sdf_grid = self.sdf_data[i]
            else:
                sdf_grid = np.load(self.sdf_files[i])
                
            # NumPy 벡터 연산을 위해 그리드를 1차원으로 펼침 (속도 향상의 핵심!)
            sdf_flat = sdf_grid.flatten()
            
            # 1. Narrow Band (표면) 마스크 및 인덱스 추출
            surface_mask = np.abs(sdf_flat) <= narrow_band_threshold
            surface_indices = np.where(surface_mask)[0] 
            
            for idx in surface_indices:
                self.valid_samples.append((i, idx))
            total_surface += len(surface_indices)
            
            # 2. 배경(Background) 샘플링 로직 (파라미터가 True일 때만 실행)
            if self.use_bg_sample:
                bg_mask = ~surface_mask # 표면이 아닌 모든 곳
                bg_indices = np.where(bg_mask)[0]
                
                # 지정된 비율만큼 랜덤 추출 (replace=False: 중복 방지)
                num_bg_to_sample = int(len(bg_indices) * self.bg_sample_ratio)
                if num_bg_to_sample > 0:
                    sampled_bg_indices = np.random.choice(bg_indices, size=num_bg_to_sample, replace=False)
                    for idx in sampled_bg_indices:
                        self.valid_samples.append((i, idx))
                    total_bg += num_bg_to_sample
                
        # 3. 모델이 편식하지 않도록(표면만 학습하다 배경만 학습하는 현상 방지) 최종 데이터 섞기
        random.shuffle(self.valid_samples)
        
        self.total_samples = len(self.valid_samples)
        total_possible_nodes = self.num_shapes * self.num_nodes_per_shape
        
        # 4. 결과 출력
        print(f"✅ 샘플링 완료!")
        print(f"   - 표면(Narrow Band) 데이터: {total_surface}개 (100% 사용)")
        if self.use_bg_sample:
            print(f"   - 배경(Background) 데이터: {total_bg}개 ({self.bg_sample_ratio*100}% 랜덤 추출)")
        print(f"   - 최종 학습 데이터: {self.total_samples}개 (전체 중 {(self.total_samples/total_possible_nodes)*100:.1f}%)")
    def __len__(self):
        return self.total_samples

    def extract_patch(self, grid, center_3d_idx, patch_size):
        """
        3D 그리드에서 (patch_size, patch_size, patch_size) 크기로 패치를 자릅니다.
        가장자리(Boundary)에 위치한 노드일 경우 Zero Padding을 적용합니다.
        """
        half_size = patch_size // 2
        start_idx = [c - half_size for c in center_3d_idx]
        end_idx = [s + patch_size for s in start_idx]
        # 1. 그리드를 벗어나는 '패딩 필요량' 계산
        pad_before = [max(0, -s) for s in start_idx]
        pad_after = [max(0, e - g) for e, g in zip(end_idx, self.grid_shape)]
        
        grid_start = [max(0, s) for s in start_idx]
        grid_end = [min(g, e) for g, e in zip(self.grid_shape, end_idx)]
        
        # 3. 안전 영역 잘라내기
        patch = grid[
            grid_start[0]:grid_end[0],
            grid_start[1]:grid_end[1],
            grid_start[2]:grid_end[2]
        ]
        
        # 4. 모자란 부분(그리드 바깥)을 0으로 채우기 (np.pad 사용)
        padded_patch = np.pad(
            patch, 
            pad_width=(
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2])
            ),
            mode='constant',
            constant_values=0
        )
        return padded_patch
    
    def __getitem__(self, idx):
        # Narrow Band 필터링 여부에 따른 인덱스 추출
        if getattr(self, 'use_narrow_band', True):
            shape_idx, node_idx = self.valid_samples[idx]
        else:
            shape_idx = idx // self.num_nodes_per_shape
            node_idx = idx % self.num_nodes_per_shape
            
        center_3d_idx = np.unravel_index(node_idx, self.grid_shape)
        
        if self.in_memory:
            sdf_grid = self.sdf_data[shape_idx]
            mc_grid = self.mc_data[shape_idx]
        else:
            sdf_grid = np.load(self.sdf_files[shape_idx])
            mc_grid = np.load(self.mc_files[shape_idx])
            
        # 1. 입력 패치 (8x8x8) 추출 및 (1, 8, 8, 8) 채널 차원 보장
        input_patch = self.extract_patch(mc_grid, center_3d_idx, self.patch_size)
        if input_patch.ndim == 3:
            input_patch = np.expand_dims(input_patch, axis=0) 
        
        # 2. 정답 패치 (3x3x3) 추출 및 1차원(27,) 평탄화
        target_patch = self.extract_patch(sdf_grid, center_3d_idx, patch_size=3)
        target_patch_flat = target_patch.flatten() 
        
        # 3. float32 텐서로 완벽히 포맷팅하여 반환 (학습 루프 병목 제거!)
        input_tensor = torch.tensor(input_patch, dtype=torch.float32)
        target_tensor = torch.tensor(target_patch_flat, dtype=torch.float32)
        
        return input_tensor, target_tensor
    
    @staticmethod
    def generate_dataset(
        output_dir: str = "dataset4",
        dataset_size: int = 50,
        start_index: int = 0,
        save_sdf: bool = True,
        save_particles: bool = True,
        save_mc: bool = False,
        config = None,
        device = 'cpu',
        cleanup_old_files: bool = False
    ):
        """
        학습용 SDF 데이터셋을 일괄 생성하고 저장하는 유틸리티 함수입니다.
        
        Args:
            output_dir: 데이터셋을 저장할 폴더 경로
            dataset_size: 총 생성할 데이터의 개수
            start_index: 생성을 시작할 인덱스 (이어서 생성할 때 유용)
            save_sdf: SDF 그리드(.npy) 저장 여부
            save_particles: 원본 파티클 좌표(.npy) 저장 여부
            save_mc: 모델 입력용 특징맵 m_c 그리드(.npy) 연산 및 저장 여부
            config: 전역 설정 (resolution, domain_size 등)
            device: 연산 디바이스 (m_c 계산 시 필요)
            cleanup_old_files: True일 경우 기존 폴더 내의 .npy 파일을 모두 삭제 후 시작
        """       
        
        import os
        import glob
        import numpy as np
        import torch
        
        from modules.sdf_generator import SDFGenerator
        from modules.sdf_network import FeatureConstruction
        from modules.particle_sampler import sample_particles_poisson

        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 기존 파일 정리 (옵션)
        if cleanup_old_files:
            old_files = glob.glob(os.path.join(output_dir, "*.npy"))
            for f in old_files:
                os.remove(f)
                print(f"🗑️ Removed old dataset file: {f}")

        # 2. PPC(Particles Per Cell) 리스트 준비 (1, 2, 3, 4 골고루 섞기)
        # dataset_size가 4의 배수가 아니어도 에러가 나지 않도록 넉넉하게 만든 뒤 자릅니다.
        ppc_list = [1, 2, 3, 4] * ((dataset_size // 4) + 1)
        ppc_list = ppc_list[:dataset_size]
        np.random.shuffle(ppc_list)

        # 3. 제너레이터 초기화
        generator = SDFGenerator(config)
        feature_constructor = None
        
        # mc_grid를 저장해야 할 때만 무거운 FeatureConstruction을 메모리에 올립니다.
        if save_mc:
            feature_constructor = FeatureConstruction(dx=config.dx, device=device)

        print(f"\n🚀 Generating {dataset_size - start_index} training samples in '{output_dir}'...")
        print(f"   [Options] SDF: {save_sdf} | Particles: {save_particles} | m_c Grid: {save_mc}")

        # 4. 본격적인 데이터 생성 루프
        for i in range(start_index, dataset_size):
            current_ppc = ppc_list[i]
            print(f"\n[{i+1}/{dataset_size}] Generating shape... (Target PPC: {current_ppc})")
            
            saved_files = []
            
            # [단계 A] SDF 생성
            random_shape = generator.create_random_shape(seed=42 + i)
            sdf_grid = generator.to_grid(random_shape)
            
            if save_sdf:
                sdf_filename = os.path.join(output_dir, f"sdf_grid_{i:03d}.npy")
                np.save(sdf_filename, sdf_grid)
                saved_files.append("SDF")
            
            # 파티클이나 mc가 필요할 때만 파티클 샘플링 진행
            if save_particles or save_mc:
                print(f"  Sampling particles with PPC={current_ppc}...")
                particles = sample_particles_poisson(sdf_grid, config, target_ppc=current_ppc)
                
                if save_particles:
                    particles_filename = os.path.join(output_dir, f"particles_{i:03d}.npy")
                    np.save(particles_filename, particles)
                    saved_files.append("Particles")
                
                # [단계 B] m_c 특징맵 계산 및 저장
                if save_mc:
                    print("  Calculating m_c features...")
                    particles_tensor = torch.tensor(particles, dtype=torch.float32, device=device)
                    # 모델 추론이나 학습과 동일한 로직으로 m_c 계산
                    with torch.no_grad():
                        grid_nodes, m_c, grid_shape = feature_constructor(particles_tensor)
                    mc_grid = m_c.reshape(grid_shape).cpu().numpy()
                    
                    mc_filename = os.path.join(output_dir, f"mc_grid_{i:03d}.npy")
                    np.save(mc_filename, mc_grid)
                    saved_files.append("m_c")
            
            print(f"  ✅ Saved: {', '.join(saved_files)} (Index: {i:03d})")

        print(f"\n🎉 Dataset generation complete! All requested files are ready in '{output_dir}'.")


    @staticmethod
    def generate_dataset(
        output_dir: str = "dataset_fluid",
        dataset_size: int = 40,      # 총 생성할 프레임(샘플) 수
        start_index: int = 0,
        save_sdf: bool = True,
        save_particles: bool = True,
        save_mc: bool = True,
        config = None,               # resolution, dx, domain_size 포함 필수
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
            feature_constructor = FeatureConstruction(dx=config.dx, device=device)

        print(f"\n🚀 Launching Optimized Fluid Simulator Pipeline...")
        print(f"   [Target Frames]: {dataset_size - start_index} | Resolution: {res}^3")

        # 4. 프레임 시뮬레이션 및 데이터 추출 루프
        for frame in range(start_index, dataset_size):
            print(f"\n[{frame+1}/{dataset_size}] Simulating & Extracting Fluid State...")
            
            # [단계 A] 유체 1스텝 전진 (내부적으로 파티클 이동)
            solver.step() 
            
            # 활성화된 파티클 Numpy 배열로 추출
            num_active = solver.get_active_particle_count()
            solver.copy_positions_to_field(tmp_particle_field)
            raw_particles_np = tmp_particle_field.to_numpy()[:num_active]

            saved_files = []

            # [단계 B] 파티클 -> SDF 필드 초고속 변환 (Spatial Hash Gather 적용)
            if save_sdf:
                # 💡 이중 루프 대신, 최적화된 모듈 함수를 호출합니다.
                sdf_grid = convert_particles_to_sdf(
                    particles_np=raw_particles_np,
                    res=res,
                    domain_size=config.domain_size,
                    radius_ratio=1.5 # 파티클 두께 조절 (필요시 튜닝)
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

            print(f"   ✅ Saved Frame {frame:03d}: {', '.join(saved_files)} (Particles: {num_active:,})")

            # (선택) 다양성 확보를 위해 N 프레임마다 물방울 초기화 
            # if (frame + 1) % 20 == 0:
            #     solver.setup_initial_fluid_block()

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
    def convert_particles_to_sdf(particles_np, res=256, domain_size=2.0, radius_ratio=1.5):
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
        dx = domain_size / res
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
