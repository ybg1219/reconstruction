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

    def extract_patch(self, grid, center_3d_idx):
        """
        3D 그리드에서 (patch_size, patch_size, patch_size) 크기로 패치를 자릅니다.
        가장자리(Boundary)에 위치한 노드일 경우 Zero Padding을 적용합니다.
        """
        start_idx = [max(0, c - self.half_size) for c in center_3d_idx]
        end_idx = [min(s, c + self.half_size) for s, c in zip(self.grid_shape, start_idx)]
        
        # 모자란 길이 보정 단계 (예: 끝자락이라 8칸이 안 될 경우 앞/뒤 공간 연장 시도)
        # 하지만 논문과 물리적으로 완벽한 형태를 위해 Zero padding을 명확하게 둡니다.
        end_idx = [e + (self.patch_size - (e - s)) for e, s in zip(end_idx, start_idx)]
        end_idx = [min(s, e) for s, e in zip(self.grid_shape, end_idx)]
        
        # 영역 슬라이싱
        patch = grid[
            start_idx[0]:end_idx[0],
            start_idx[1]:end_idx[1],
            start_idx[2]:end_idx[2]
        ]
        
        # 패딩 배열 생성 및 삽입 (빈 공간은 0으로 채워짐)
        padded_patch = np.zeros((self.patch_size, self.patch_size, self.patch_size), dtype=np.float32)
        
        # 실제 데이터가 들어가는 공간(인덱스) 계산
        start_pad = [self.half_size - (c - s) for s, c in zip(start_idx, center_3d_idx)]
        
        # numpy indexing을 통해 알맞은 구역에 패치 삽입
        padded_patch[
            start_pad[0] : start_pad[0] + patch.shape[0],
            start_pad[1] : start_pad[1] + patch.shape[1],
            start_pad[2] : start_pad[2] + patch.shape[2]
        ] = patch
        
        # 모델 입력 포맷 형태: (Channel=1, Depth, Height, Width)
        # return 시 Data Loader가 Batch 차원을 추가하므로 (1, 8, 8, 8)로 리턴해야
        # 최종적으로 (Batch, 1, 8, 8, 8) 이 됩니다.
        return np.expand_dims(padded_patch, axis=0)

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
            
        # 4. 특징 특징 패치 잘라내기
        patch = self.extract_patch(mc_grid, center_3d_idx)
        
        # 5. 정답 추출
        sdf_val = sdf_grid[center_3d_idx]
        
        # 텐서 변환 (입력 채널 1, 정답 스칼라 1)
        return torch.tensor(patch, dtype=torch.float32), torch.tensor([sdf_val], dtype=torch.float32)


def create_dataloader(data_dir="dataset", batch_size=32, in_memory=True, num_workers=0, patch_size=8):
    """SDF 네트워크 학습용 데이터로더 생성기"""
    dataset = SDFDataset(data_dir=data_dir, patch_size=patch_size, in_memory=in_memory)
    
    # 학습 시에는 Sliding Window로 추출된 여러 도형의 노드들이 골고루 섞여야 하므로 shuffle=True 사용
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
