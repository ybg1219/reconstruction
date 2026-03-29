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
    def __init__(self, data_dir="dataset", patch_size=8, in_memory=True):
        """
        Args:
            data_dir: .npy 파일들이 저장된 최상단 폴더 (예: 'dataset')
            patch_size: 입력 특징 패치 크기 (기본값 8, 8x8x8)
            in_memory: True일 경우 RAM에 전체 그리드를 먼저 불러옴 (학습 속도 대폭 향상) 
        """
        super().__init__()
        self.patch_size = patch_size
        self.half_size = patch_size // 2
        self.in_memory = in_memory
        
        # 데이터 디렉토리 내의 파일 목록 검색 (이전 Phase에서 생성했던 파일 구조에 대응)
        # 예: sdf_grid_64.npy 형태나, 서브 폴더에 저장된 형태를 찾음
        # 현 프로젝트 구조상 단일 파일 형태나 여러 파일들이 섞여 있을 수 있으므로 sdf_grid_*, mc_grid_* 로 매치.
        self.sdf_files = sorted(glob.glob(os.path.join(data_dir, "sdf_grid*.npy")))
        self.mc_files = sorted(glob.glob(os.path.join(data_dir, "mc_grid*.npy")))
        
        if len(self.sdf_files) == 0:
            print(f"⚠️ 경고: '{data_dir}' 경로에서 데이터셋 파일을 찾지 못했습니다. 경로를 확인해주세요.")
            self.num_shapes = 0
            self.total_samples = 0
            return
            
        if len(self.sdf_files) != len(self.mc_files):
            print(f"⚠️ 경고: SDF 파일 개수({len(self.sdf_files)})와 m_c 파일 개수({len(self.mc_files)})가 다릅니다.")
            
        self.num_shapes = min(len(self.sdf_files), len(self.mc_files))
        
        # 첫 번째 파일의 형태를 읽어 전체 그리드 해상도 파악
        sample_grid = np.load(self.sdf_files[0])
        self.grid_shape = sample_grid.shape # 예: (64, 64, 64)
        self.num_nodes_per_shape = np.prod(self.grid_shape)
        
        # 전체 데이터 개수 = 도형 1개당 발생할 수 있는 sliding window 경우의 수 * 전체 도형 개수
        self.total_samples = self.num_shapes * self.num_nodes_per_shape
        
        self.sdf_data = []
        self.mc_data = []
        
        if self.in_memory:
            print("💾 데이터를 메모리에 캐싱 중입니다...")
            for i in range(self.num_shapes):
                self.sdf_data.append(np.load(self.sdf_files[i]))
                self.mc_data.append(np.load(self.mc_files[i]))
            print("✅ 캐싱 완료.")

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
