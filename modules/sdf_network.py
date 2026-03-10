"""
3D SDF 재구성 네트워크 모듈
- Poly6 커널 기반 특징값(m_c) 계산
- 3D CNN 기반 SDF 예측 모델
- 전체 파이프라인 통합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ===========================
# 1. 전처리 모듈: Feature Construction
# ===========================

class FeatureConstruction:
    """파티클 위치에서 그리드 특징값(m_c) 계산"""
    
    def __init__(self, dx: float = 0.1, device: str = 'cpu'):
        """
        Args:
            dx: 그리드 간격
            device: 연산 기기 ('cpu' 또는 'cuda')
        """
        self.dx = dx
        self.R = 3.0 * dx  # 커널 반경
        self.device = device
        
    def poly6_kernel(self, r: torch.Tensor, R: float) -> torch.Tensor:
        """
        Poly6 스무딩 커널
        W(r, R) = (315 / 64π R^9) * (R^2 - r^2)^3
        
        Args:
            r: 거리 (텐서)
            R: 커널 반경
        
        Returns:
            커널 값 (텐서)
        """
        coeff = 315.0 / (64.0 * np.pi * R**9)
        mask = (r < R).float()
        value = coeff * torch.clamp(R**2 - r**2, min=0.0)**3
        return value * mask
    
    def compute_particle_density(self, particle_positions: torch.Tensor) -> torch.Tensor:
        """
        각 파티클의 밀도 계산
        ρ_p = Σ_{q ∈ N_p} W(||x_p - x_q||, R)
        
        Args:
            particle_positions: (N, 3) 파티클 좌표
        
        Returns:
            rho_p: (N,) 밀도 값
        """
        N = particle_positions.shape[0]
        
        # 모든 파티클 쌍에 대해 거리 계산
        distances = torch.cdist(particle_positions, particle_positions, p=2)
        
        # 커널 값 계산
        kernel_values = self.poly6_kernel(distances, self.R)
        
        # 각 파티클별 밀도 합산
        rho_p = kernel_values.sum(dim=1)
        
        return rho_p
    
    def create_grid(self, particle_positions: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
        """
        파티클 위치 기반 그리드 생성
        
        Args:
            particle_positions: (N, 3) 파티클 좌표
        
        Returns:
            grid_nodes: (G, 3) 그리드 노드 좌표
            grid_shape: (Gx, Gy, Gz) 그리드 크기
        """
        # 파티클 위치 기반 그리드 생성
        # min_coords = particle_positions.min(dim=0)[0] - self.dx * 2
        # max_coords = particle_positions.max(dim=0)[0] + self.dx * 2

        # 고정 범위 그리드 생성
        domain_size = 2.0  # config.domain_size 와 동일하게 맞추세요
        min_bound = -domain_size / 2.0
        max_bound = domain_size / 2.0
        
        # 그리드 생성
        x = torch.arange(min_bound, max_bound + 1e-5, self.dx, device=particle_positions.device)
        y = torch.arange(min_bound, max_bound + 1e-5, self.dx, device=particle_positions.device)
        z = torch.arange(min_bound, max_bound + 1e-5, self.dx, device=particle_positions.device)

        grid_nodes = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
        grid_shape = grid_nodes.shape[:3]
        grid_nodes = grid_nodes.reshape(-1, 3)
        
        return grid_nodes, grid_shape
    
    def compute_grid_features(self, particle_positions: torch.Tensor, 
                             grid_nodes: torch.Tensor, chunk_size: int = 10000) -> torch.Tensor:
        """
        그리드 노드 특징값 계산 (메모리 절약을 위해 Chunk 단위 처리)
        m_c = Σ_{p ∈ N_c} (1/ρ_p) * W(||x_c - x_p||, R)
        
        Args:
            particle_positions: (N, 3) 파티클 좌표
            grid_nodes: (G, 3) 그리드 노드 좌표
            chunk_size: 한 번에 처리할 노드 개수 (기본 10000개)
        
        Returns:
            m_c: (G,) 그리드 특징값
        """
        # 파티클 밀도 계산
        rho_p = self.compute_particle_density(particle_positions)
        rho_p = torch.clamp(rho_p, min=1e-6)  # 수치 안정성
        
        num_nodes = grid_nodes.shape[0]
        m_c_list = []
        
        # 메모리 초과를 방지하기 위해 노드를 chunk 단위로 처리
        for i in range(0, num_nodes, chunk_size):
            chunk_nodes = grid_nodes[i:i+chunk_size]
            
            # 그리드-파티클 거리 계산 (Chunk, N)
            distances = torch.cdist(chunk_nodes, particle_positions, p=2)
            
            # 커널 값 계산 (Chunk, N)
            kernel_values = self.poly6_kernel(distances, self.R)
            
            # 1/ρ_p 적용 (Chunk, N)
            weighted_kernel = kernel_values / rho_p.unsqueeze(0)
            
            # m_c 계산 (Chunk,)
            m_c_chunk = weighted_kernel.sum(dim=1)
            m_c_list.append(m_c_chunk)
            
        m_c = torch.cat(m_c_list, dim=0)
        return m_c
    
    def __call__(self, particle_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        전체 전처리 파이프라인
        
        Args:
            particle_positions: (N, 3) 파티클 좌표
        
        Returns:
            grid_nodes: (G, 3) 그리드 노드 좌표
            m_c: (G,) 그리드 특징값
            grid_shape: (Gx, Gy, Gz) 그리드 크기
        """
        particle_positions = particle_positions.float()
        
        # 그리드 생성
        grid_nodes, grid_shape = self.create_grid(particle_positions)
        
        # 그리드 특징값 계산
        m_c = self.compute_grid_features(particle_positions, grid_nodes)
        
        return grid_nodes, m_c, grid_shape


# ===========================
# 2. 3D CNN 모듈
# ===========================

class SDFNetwork(nn.Module):
    """3D CNN 기반 SDF 예측 네트워크"""
    
    def __init__(self, kernel_size: int = 3, padding: int = 1):
        """
        Args:
            kernel_size: Conv3d 커널 크기
            padding: Conv3d 패딩
        """
        super(SDFNetwork, self).__init__()
        
        # Conv3d Layer 1: (1, 8, 8, 8) -> (32, 8, 8, 8)
        self.conv1 = nn.Conv3d(1, 32, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(32)
        
        # Conv3d Layer 2: (32, 8, 8, 8) -> (64, 8, 8, 8)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(64)
        
        # MaxPool3d: (64, 8, 8, 8) -> (64, 4, 4, 4)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Conv3d Layer 3: (64, 4, 4, 4) -> (128, 4, 4, 4)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm3d(128)
        
        # Conv3d Layer 4: (128, 4, 4, 4) -> (256, 4, 4, 4)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm3d(256)
        
        # Flatten: 256 * 4 * 4 * 4 = 16384
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(256 * 4 * 4 * 4, 256)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(256, 128)
        
        # Fully Connected Layer 3 (Output)
        self.fc3 = nn.Linear(128, 1)
        
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 1, 8, 8, 8) 특징값 블록
        
        Returns:
            sdf: (Batch, 1) SDF 값
        """
        # Conv1 + BN + Activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Conv2 + BN + Activation
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        # MaxPool
        x = self.pool(x)
        
        # Conv3 + BN + Activation
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        
        # Conv4 + BN + Activation
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC1
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # FC2
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # FC3 (Output)
        sdf = self.fc3(x)
        
        return sdf

    def train_step(self, train_loader, optimizer, criterion, device='cpu', num_epochs=10, verbose=True):
        """
        SDFNetwork 학습 루프 (여러 에폭)
        Args:
            train_loader: DataLoader yielding (feature_patch, sdf_gt) tuples
            optimizer: torch.optim.Optimizer
            criterion: 손실 함수 (예: nn.MSELoss)
            device: 학습 기기
            num_epochs: 학습 에폭 수
            verbose: 진행 상황 출력 여부
        Returns:
            epoch_losses: 에폭별 평균 손실 리스트
        """
        self.train()
        epoch_losses = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (feature_patch, sdf_gt) in enumerate(train_loader):
                feature_patch = feature_patch.to(device)
                sdf_gt = sdf_gt.to(device)
                optimizer.zero_grad()
                output = self.forward(feature_patch)
                loss = criterion(output.squeeze(-1), sdf_gt)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * feature_patch.size(0)
            avg_loss = running_loss / len(train_loader.dataset)
            epoch_losses.append(avg_loss)
            if verbose:
                print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")
        return epoch_losses


# ===========================
# 3. 통합 파이프라인
# ===========================

class SDFReconstruction:
    """전처리 + 네트워크 통합 클래스"""
    
    def __init__(self, dx: float = 0.1, device: str = 'cpu'):
        """
        Args:
            dx: 그리드 간격
            device: 연산 기기
        """
        self.dx = dx
        self.device = device
        self.feature_construction = FeatureConstruction(dx=dx, device=device)
        self.network = SDFNetwork().to(device)
    
    def extract_local_features(self, m_c_grid: torch.Tensor, 
                              center_idx: int,
                              grid_shape: tuple,
                              patch_size: int = 8) -> torch.Tensor:
        """
        중심 노드 선형 인덱스를 받아서 주변 8×8×8 패치 추출 (패딩 포함)
        """
        c_idx = np.unravel_index(center_idx, grid_shape)
        half_size = patch_size // 2
        
        start_idx = [c - half_size for c in c_idx]
        end_idx = [c + half_size for c in c_idx]
        
        grid_start = [max(0, s) for s in start_idx]
        grid_end = [min(g, e) for g, e in zip(grid_shape, end_idx)]
        
        # 1. 원본 그리드에서 유효한 부분 자르기
        patch = m_c_grid[
            grid_start[0]:grid_end[0],
            grid_start[1]:grid_end[1],
            grid_start[2]:grid_end[2]
        ]
        
        # 2. 패딩 계산 (영역 밖으로 나간 양)
        pad_before = [max(0, -s) for s in start_idx]
        pad_after = [max(0, e - g) for e, g in zip(end_idx, grid_shape)]
        
        # 3. F.pad를 사용하기 위해 (뒤집힌 순서로) 패딩 나열
        # PyTorch F.pad는 (마지막차원_앞, 마지막차원_뒤, 그앞차원_앞, 그앞차원_뒤, 처음차원_앞, 처음차원_뒤)
        padding_tuple = (
            pad_before[2], pad_after[2], 
            pad_before[1], pad_after[1], 
            pad_before[0], pad_after[0]
        )
        
        # 4. 패딩 적용 (크기를 강제로 8x8x8로 만듦)
        padded_patch = F.pad(patch, padding_tuple, mode='constant', value=0.0)
        
        # 5. 차원 확장: (8, 8, 8) -> (1, 1, 8, 8, 8) [Batch=1, Channel=1]
        return padded_patch.unsqueeze(0).unsqueeze(0)
    
    def preprocess(self, particle_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        1단계: 전처리 (그리드 특징값 계산)
        
        Args:
            particle_positions: (N, 3) 파티클 좌표
        
        Returns:
            m_c_grid: (Gx, Gy, Gz) 3D 그리드 특징값
            grid_nodes: (G, 3) 그리드 노드 좌표
            grid_shape: (Gx, Gy, Gz) 그리드 크기
        """
        particle_positions = particle_positions.to(self.device)
        
        # 전처리 실행
        grid_nodes, m_c, grid_shape = self.feature_construction(particle_positions)
        
        # m_c를 3D 그리드로 변환
        m_c_grid = m_c.reshape(grid_shape)
        
        return m_c_grid, grid_nodes, grid_shape
    
    def forward(self, particle_positions: torch.Tensor, 
                batch_size: int = 512) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """
        [수정됨] 전체 파이프라인: 파티클 -> SDF 값
        """
        # 1. 전처리 (m_c 그리드 계산)
        m_c_grid, grid_nodes, grid_shape = self.preprocess(particle_positions)
        
        # 2. 전체 그리드에 패딩 적용 (8x8x8 패치 추출 시 경계 에러 방지용)
        # 사방으로 4칸(patch_size // 2)씩 0으로 채움
        pad_size = 4
        padded_m_c_grid = F.pad(m_c_grid, (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        
        num_nodes = grid_nodes.shape[0]
        sdf_values = torch.zeros(num_nodes, device=self.device)
        
        print(f"총 {num_nodes}개의 그리드 노드에 대해 SDF 추론을 시작합니다...")

        # 3. 배치 단위로 잘라서 3D CNN 통과
        self.network.eval() # 추론 모드로 전환
        with torch.no_grad():
            for i in range(0, num_nodes, batch_size):
                batch_end = min(i + batch_size, num_nodes)
                batch_patches = []
                
                # 배치 내의 노드들에 대해 패치 긁어오기
                for idx in range(i, batch_end):
                    # 1차원 인덱스를 3차원 (x, y, z) 인덱스로 변환
                    c_idx = np.unravel_index(idx, grid_shape)
                    patch = self.extract_local_features(padded_m_c_grid, c_idx, patch_size=8)
                    batch_patches.append(patch)
                
                # (Batch, 1, 8, 8, 8) 형태로 결합
                batch_tensor = torch.cat(batch_patches, dim=0).to(self.device)
                
                # 모델 통과
                batch_sdf = self.network(batch_tensor)
                sdf_values[i:batch_end] = batch_sdf.squeeze(-1)
                
                if (i % (batch_size * 10)) == 0:
                    print(f"추론 진행률: {i}/{num_nodes} ({(i/num_nodes)*100:.1f}%)")

        return grid_nodes, sdf_values, m_c_grid, grid_shape