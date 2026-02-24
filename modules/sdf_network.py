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
        min_coords = particle_positions.min(dim=0)[0] - self.dx * 2
        max_coords = particle_positions.max(dim=0)[0] + self.dx * 2
        
        # 그리드 생성
        x = torch.arange(min_coords[0], max_coords[0], self.dx, device=particle_positions.device)
        y = torch.arange(min_coords[1], max_coords[1], self.dx, device=particle_positions.device)
        z = torch.arange(min_coords[2], max_coords[2], self.dx, device=particle_positions.device)
        
        grid_nodes = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
        grid_shape = grid_nodes.shape[:3]
        grid_nodes = grid_nodes.reshape(-1, 3)
        
        return grid_nodes, grid_shape
    
    def compute_grid_features(self, particle_positions: torch.Tensor, 
                             grid_nodes: torch.Tensor) -> torch.Tensor:
        """
        그리드 노드 특징값 계산
        m_c = Σ_{p ∈ N_c} (1/ρ_p) * W(||x_c - x_p||, R)
        
        Args:
            particle_positions: (N, 3) 파티클 좌표
            grid_nodes: (G, 3) 그리드 노드 좌표
        
        Returns:
            m_c: (G,) 그리드 특징값
        """
        # 파티클 밀도 계산
        rho_p = self.compute_particle_density(particle_positions)
        rho_p = torch.clamp(rho_p, min=1e-6)  # 수치 안정성
        
        # 그리드-파티클 거리 계산 (G, N)
        distances = torch.cdist(grid_nodes, particle_positions, p=2)
        
        # 커널 값 계산 (G, N)
        kernel_values = self.poly6_kernel(distances, self.R)
        
        # 1/ρ_p 적용 (G, N)
        weighted_kernel = kernel_values / rho_p.unsqueeze(0)
        
        # m_c 계산 (G,)
        m_c = weighted_kernel.sum(dim=1)
        
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
        중심 노드 주변의 8×8×8 패치 추출
        
        Args:
            m_c_grid: (Gx, Gy, Gz) 3D 그리드 특징값
            center_idx: 중심 노드 선형 인덱스
            grid_shape: (Gx, Gy, Gz) 그리드 크기
            patch_size: 패치 크기 (기본값 8)
        
        Returns:
            patch: (1, 1, 8, 8, 8) 패치 특징값
        """
        # 선형 인덱스를 3D 인덱스로 변환
        center_3d_idx = np.unravel_index(center_idx, grid_shape)
        
        # 패치 범위 계산
        half_size = patch_size // 2
        start_idx = [max(0, c - half_size) for c in center_3d_idx]
        end_idx = [min(s, c + half_size) for s, c in zip(grid_shape, start_idx)]
        end_idx = [e + (patch_size - (e - s)) for e, s in zip(end_idx, start_idx)]
        end_idx = [min(s, e) for s, e in zip(grid_shape, end_idx)]
        
        # 패치 추출
        patch = m_c_grid[
            start_idx[0]:end_idx[0],
            start_idx[1]:end_idx[1],
            start_idx[2]:end_idx[2]
        ]
        
        # 패치를 8×8×8로 패딩
        padded_patch = torch.zeros(
            patch_size, patch_size, patch_size,
            device=m_c_grid.device,
            dtype=m_c_grid.dtype
        )
        padded_patch[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
        
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
                num_inference_nodes: Optional[int] = None,
                batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        전체 파이프라인: 파티클 -> SDF 값
        
        Args:
            particle_positions: (N, 3) 파티클 좌표
            num_inference_nodes: 추론할 노드 개수 (None이면 전체)
            batch_size: 배치 크기
        
        Returns:
            grid_nodes: (G, 3) 그리드 노드 좌표
            sdf_values: (G,) SDF 값
            m_c_grid: (Gx, Gy, Gz) 특징값 그리드
            grid_shape: 그리드 크기
        """
        # 1단계: 전처리
        m_c_grid, grid_nodes, grid_shape = self.preprocess(particle_positions)
        
        # 2단계: SDF 추론
        num_nodes = grid_nodes.shape[0]
        sdf_values = torch.zeros(num_nodes, device=self.device)
        
        if num_inference_nodes is None:
            num_inference_nodes = min(1000, num_nodes)
        
        with torch.no_grad():
            for i in range(0, num_inference_nodes, batch_size):
                batch_end = min(i + batch_size, num_inference_nodes)
                batch_patches = []
                
                for idx in range(i, batch_end):
                    patch = self.extract_local_features(m_c_grid, idx, grid_shape)
                    batch_patches.append(patch)
                
                batch_patches = torch.cat(batch_patches, dim=0).to(self.device)
                batch_sdf = self.network(batch_patches)
                sdf_values[i:batch_end] = batch_sdf.squeeze(-1)
        
        return grid_nodes, sdf_values, m_c_grid, grid_shape
