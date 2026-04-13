import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

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