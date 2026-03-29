import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SDFDataset(Dataset):
    """
    미리 계산된 SDF 그리드와 m_c 특징 그리드(.npy) 쌍을 불러와 
    8x8x8 슬라이딩 윈도우 패치(Patch) 단위로 분할하여 제공하는 학습 데이터셋.
    """
    def __init__(self, data_dir="dataset", patch_size=8, in_memory=True, use_narrow_band=True):
        """
        Args:
            data_dir: .npy 파일 경로
            patch_size: 입력 특징 패치 크기
            in_memory: RAM 캐싱 여부
            use_narrow_band: True일 경우 논문처럼 표면 근처(Narrow Band) 데이터만 필터링하여 학습
        """
        super().__init__()
        self.patch_size = patch_size
        self.half_size = patch_size // 2
        self.in_memory = in_memory
        self.use_narrow_band = use_narrow_band  # 🚨 파라미터 저장
        
        self.sdf_files = sorted(glob.glob(os.path.join(data_dir, "sdf_grid*.npy")))
        self.mc_files = sorted(glob.glob(os.path.join(data_dir, "mc_grid*.npy")))
        
        if len(self.sdf_files) == 0:
            print(f"⚠️ 경고: '{data_dir}' 경로에서 데이터셋 파일을 찾지 못했습니다.")
            self.num_shapes, self.total_samples = 0, 0
            return
            
        self.num_shapes = min(len(self.sdf_files), len(self.mc_files))
        
        sample_grid = np.load(self.sdf_files[0])
        self.grid_shape = sample_grid.shape 
        self.num_nodes_per_shape = np.prod(self.grid_shape)
        
        self.sdf_data = []
        self.mc_data = []
        
        if self.in_memory:
            print("💾 데이터를 메모리에 캐싱 중입니다...")
            for i in range(self.num_shapes):
                self.sdf_data.append(np.load(self.sdf_files[i]))
                self.mc_data.append(np.load(self.mc_files[i]))
            print("✅ 캐싱 완료.")

        # 🚨 [추가된 분기 처리] 파라미터에 따라 Narrow Band 필터링을 할지 말지 결정합니다.
        if self.use_narrow_band:
            self._apply_narrow_band_filtering()
        else:
            self.total_samples = self.num_shapes * self.num_nodes_per_shape
            print(f"✅ 전체 노드 학습 모드 가동: 총 {self.total_samples}개 샘플")

    def _apply_narrow_band_filtering(self):
        """[신규 함수] 논문 기반 Narrow Band 필터링을 수행합니다."""
        self.valid_samples = [] 
        
        dx = 1.0 / self.grid_shape[0]
        narrow_band_threshold = 2.0 * dx  # 논문 기준 [-2dx, 2dx]
        
        print(f"🔍 표면 근처(Narrow Band: |SDF| <= {narrow_band_threshold:.4f}) 노드만 필터링 중...")
        
        for i in range(self.num_shapes):
            if self.in_memory:
                sdf_grid = self.sdf_data[i]
            else:
                sdf_grid = np.load(self.sdf_files[i])
                
            valid_coords = np.where(np.abs(sdf_grid) <= narrow_band_threshold)
            
            for x, y, z in zip(*valid_coords):
                linear_idx = np.ravel_multi_index((x, y, z), self.grid_shape)
                self.valid_samples.append((i, linear_idx))
                
        self.total_samples = len(self.valid_samples)
        total_possible_nodes = self.num_shapes * self.num_nodes_per_shape
        
        print(f"✅ 필터링 완료: 전체 {total_possible_nodes}개 중 핵심 {self.total_samples}개만 학습합니다! (약 {(self.total_samples/total_possible_nodes)*100:.1f}%)")
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
        # 1. 사용할 도형(쉐입) 인덱스 식별
        shape_idx = idx // self.num_nodes_per_shape
        
        # 2. 해당 도형(NxNxN 그리드) 내부에서의 선형 인덱스(0 ~ N^3 - 1)
        node_idx = idx % self.num_nodes_per_shape
        
        # 선형 인덱스를 3D (X, Y, Z) 좌표계 인덱스로 변환 (Sliding Window 핵심)
        center_3d_idx = np.unravel_index(node_idx, self.grid_shape)
        
        # 3. 데이터 로드
        if self.in_memory:
            sdf_grid = self.sdf_data[shape_idx]
            mc_grid = self.mc_data[shape_idx]
        else:
            sdf_grid = np.load(self.sdf_files[shape_idx])
            mc_grid = np.load(self.mc_files[shape_idx])
            
        # 4. 입력용 m_c 특징 패치 추출 (8x8x8) -> 형태: (1, 8, 8, 8)
        input_patch = self.extract_patch(mc_grid, center_3d_idx, self.patch_size)
        input_patch = np.expand_dims(input_patch, axis=0) 
        
        # 5. 정답용 SDF 패치 추출 (3x3x3) -> 형태: (27,)
        target_patch = self.extract_patch(sdf_grid, center_3d_idx, patch_size=3)
        target_patch_flat = target_patch.flatten() # 1D 배열로 평탄화
        
        return torch.tensor(input_patch, dtype=torch.float32), torch.tensor(target_patch_flat, dtype=torch.float32)


def create_dataloader(data_dir="dataset", batch_size=32, in_memory=True, num_workers=0, patch_size=8):
    """SDF 네트워크 학습용 데이터로더 생성기"""
    dataset = SDFDataset(data_dir=data_dir, patch_size=patch_size, in_memory=in_memory)
    
    # 학습 시에는 Sliding Window로 추출된 여러 도형의 노드들이 골고루 섞여야 하므로 shuffle=True 사용
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
