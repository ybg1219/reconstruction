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
    def __init__(self, data_dirs=["dataset1", "dataset2"], patch_size=8, in_memory=True, 
                 use_narrow_band=True, use_bg_sample=True, bg_sample_ratio=0.05,
                 feature_constructor=None, max_samples=None):
        super().__init__()
        self.patch_size = patch_size
        self.half_size = patch_size // 2
        self.in_memory = in_memory
        
        self.use_narrow_band = use_narrow_band  
        self.use_bg_sample = use_bg_sample
        self.bg_sample_ratio = bg_sample_ratio
        self.max_samples = max_samples
        
        # 리스트가 아니라 문자열(단일 경로) 하나만 들어오면 리스트로 감싸줍니다.
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        
        self.sdf_files = []
        self.mc_files = []
        self.particle_files = []
        
        for d_dir in data_dirs:
            # 해당 폴더 내의 파일들 검색
            s_files = sorted(glob.glob(os.path.join(d_dir, "sdf_grid*.npy")))
            m_files = sorted(glob.glob(os.path.join(d_dir, "mc_grid*.npy")))
            p_files = sorted(glob.glob(os.path.join(d_dir, "particles*.npy")))
            
            if len(s_files) == 0:
                print(f"⚠️ 경고: '{d_dir}' 경로에 SDF 데이터가 없어 건너뜁니다.")
                continue

            # 해당 폴더에 mc_grid가 없고 particles만 있을 경우 자동 생성
            if len(m_files) == 0 and len(p_files) > 0:
                if feature_constructor is None:
                    raise ValueError(f"❌ '{d_dir}' 폴더에 mc_grid가 없습니다. 생성하려면 feature_constructor를 전달해주세요.")
                
                print(f"🔄 '{d_dir}' 폴더: mc_grid 자동 생성을 시작합니다...")
                for p_file in p_files:
                    particles_np = np.load(p_file)
                    particles_tensor = torch.tensor(particles_np, dtype=torch.float32, device=feature_constructor.device)
                    
                    with torch.no_grad():
                        _, m_c, grid_shape = feature_constructor(particles_tensor)
                    mc_grid = m_c.reshape(grid_shape).cpu().numpy()
                    
                    base_name = os.path.basename(p_file).replace("particles_", "mc_grid_")
                    mc_filename = os.path.join(d_dir, base_name)
                    
                    np.save(mc_filename, mc_grid)
                    m_files.append(mc_filename)
                print(f"✅ '{d_dir}' 폴더: mc_grid 생성 완료!\n")
            
            # 🔥 각 폴더에서 수집한 파일들을 메인 리스트에 합칩니다 (짝이 맞도록 개수 제한)
            min_len = min(len(s_files), len(m_files))
            self.sdf_files.extend(s_files[:min_len])
            self.mc_files.extend(m_files[:min_len])
            self.particle_files.extend(p_files) # 파티클은 참고용으로 합침

        # =========================================================

        if len(self.sdf_files) == 0:
            print(f"⚠️ 경고: 제공된 모든 경로에서 데이터셋 파일을 찾지 못했습니다.")
            self.num_shapes, self.total_samples = 0, 0
            return

        self.num_shapes = len(self.sdf_files)
        
        sample_grid = np.load(self.sdf_files[0])
        self.grid_shape = sample_grid.shape 
        self.num_nodes_per_shape = np.prod(self.grid_shape)
        
        self.sdf_data = []
        self.mc_data = []
        
        # 1. 데이터 캐싱
        if self.in_memory:
            print(f"💾 총 {self.num_shapes}쌍의 데이터를 메모리에 캐싱 중입니다...")
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
        original_count = len(self.valid_samples)
        if self.max_samples is not None and original_count > self.max_samples:
            self.valid_samples = self.valid_samples[:self.max_samples]
            print(f"✂️ [데이터 제한] 전체 {original_count:,}개 샘플 중 무작위 {self.max_samples:,}개만 선택되었습니다.")
            
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

