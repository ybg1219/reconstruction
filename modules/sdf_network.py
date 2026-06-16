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
from tqdm import tqdm


class PolynomialRegularizedSDFLoss(nn.Module):
    def __init__(self, lambda_reg=1.0, use_poly_loss=True):
        """
        Args:
            lambda_reg: 다항식 정규화 가중치 (논문 기준 1.0)
            use_poly_loss: True면 다항식 정규화 + MSE, False면 순수 MSE만 적용
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.use_poly_loss = use_poly_loss
        self.mse_loss = nn.MSELoss()

        # 1. 3x3x3 패치의 로컬 좌표 생성 (x, y, z ∈ {-1, 0, 1})
        # 논문에서는 중심을 기준으로 -1, 0, 1의 좌표계를 사용합니다.
        coords = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        # 2. 2차 다항식을 위한 Vandermonde 행렬 A (27 x 10) 생성
        A = torch.stack([
            x**2, y**2, z**2,
            x*y, y*z, x*z,
            x, y, z,
            torch.ones_like(x)
        ], dim=1)

        # 3. 투영 행렬 K = I - A(A^T A)^{-1} A^T 계산
        # torch.linalg.pinv를 사용하면 (A^T A)^{-1} A^T 계산을 수치적으로 매우 안정하게 수행할 수 있습니다.
        A_pinv = torch.linalg.pinv(A)
        I = torch.eye(27)
        K = I - torch.matmul(A, A_pinv)

        # 4. K 행렬을 buffer로 등록 
        # (학습되지 않는 상수 텐서로 취급되며, 모델이 GPU로 갈 때 자동으로 같이 이동합니다)
        self.register_buffer('K', K)

    def forward(self, pred, target):
        """
        Args:
            pred: 모델의 예측값. Shape (Batch, 27)
            target: 정답 SDF값. Shape (Batch, 27)
        """
        # 1. 기본 L2(MSE) 손실
        loss_data = self.mse_loss(pred, target)

        # False면 여기서 MSE만 반환하고 끝
        if not self.use_poly_loss:
            return loss_data

        # 2. 다항식 정규화 손실: L_p = ||K \Phi||_2^2
        K_pred = torch.matmul(pred, self.K) # (Batch, 27) * (27, 27) -> (Batch, 27)
        
        # 각 배치마다 27개 원소의 제곱합을 구한 뒤 평균을 냅니다.
        loss_reg = torch.mean(torch.sum(K_pred ** 2, dim=1))

        # 3. 최종 Loss 반환
        return loss_data + (self.lambda_reg * loss_reg)

# ===========================
# 1. 전처리 모듈: Feature Construction
# ===========================
class FeatureConstruction:
    """
    파티클 위치에서 그리드 특징값(m_c) 계산
    Splatting (Scatter-Add) 기법을 사용
    """
    
    def __init__(self, dx: float = 0.1, particle_spacing: float = None, device: str = 'cpu'):
        """
        Args:
            dx: 그리드 간격
            device: 연산 기기 ('cpu' 또는 'cuda')
        """
        self.dx = dx
        self.device = device
        self.domain_size = 2.0  # 전역 도메인 크기
        self.min_bound = -self.domain_size / 2.0
        self.max_bound = self.domain_size / 2.0
        
        if particle_spacing is not None:
            # 파티클 간격의 2배
            self.R = 2.0 * particle_spacing 
        else:
            # 기본값 (저해상도)
            self.R = 3.0 * dx
            
        print(f"self.R : {self.R}")

        # 공간 해싱을 위한 격자 크기 및 차원 설정
        self.cell_size = self.R * 0.9
        self.grid_dims = int(np.ceil(self.domain_size / self.cell_size))
        
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
        # r > R 이면 R^2 - r^2 가 음수가 되므로 clamp(min=0.0)에서 이미 0으로 처리됩니다.
        value = coeff * torch.clamp(R**2 - r**2, min=0.0)**3
        return value

    # ---------------------------------------------------------
    # 내부 최적화 유틸리티 (공간 해싱 및 VRAM 관리)
    # ---------------------------------------------------------
    def _spatial_hash_and_sort(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """파티클 해싱 및 공간 정렬"""
        cell_coords = torch.floor((positions - self.min_bound) / self.cell_size).long()
        cell_coords = torch.clamp(cell_coords, 0, self.grid_dims - 1)
        
        # 1D 해시 키 (z축, y축, x축 순 평탄화)
        hash_keys = (cell_coords[:, 0] * (self.grid_dims ** 2) + 
                     cell_coords[:, 1] * self.grid_dims + 
                     cell_coords[:, 2])
        
        sorted_keys, sorted_indices = torch.sort(hash_keys)
        sorted_pos = positions[sorted_indices]
        return sorted_pos, sorted_keys

    def _build_cell_offsets(self, sorted_keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """bincount와 cumsum을 이용한 O(1) 셀 슬라이싱 오프셋 구축"""
        total_cells = self.grid_dims ** 3
        counts = torch.bincount(sorted_keys, minlength=total_cells)
        end_indices = torch.cumsum(counts, dim=0)
        start_indices = end_indices - counts
        return start_indices, end_indices

    def _compute_density_hashed(self, sorted_pos: torch.Tensor, sorted_keys: torch.Tensor, 
                                start_indices: torch.Tensor, end_indices: torch.Tensor, 
                                chunk_size: int = 10000) -> torch.Tensor:
        """3단계: 27방향 이웃 탐색 및 최종 밀도 측정 (OOM 방지를 위한 Chunk 처리 포함)"""
        N = sorted_pos.shape[0]
        total_cells = self.grid_dims ** 3
        rho_p = torch.zeros(N, device=self.device)
        
        counts = end_indices - start_indices
        max_p_per_cell = counts.max().item()
        
        if max_p_per_cell == 0:
            return rho_p
            
        particle_idx = torch.arange(N, device=self.device)
        rank_in_cell = particle_idx - start_indices[sorted_keys]
        
        cell_particle_idx = torch.zeros((total_cells, max_p_per_cell), dtype=torch.long, device=self.device)
        cell_particle_idx[sorted_keys, rank_in_cell] = particle_idx
        
        cell_mask = torch.arange(max_p_per_cell, device=self.device).unsqueeze(0) < counts.unsqueeze(1)
        
        offsets = torch.tensor([-1, 0, 1], device=self.device)
        ox, oy, oz = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
        neighbor_offsets = torch.stack([ox, oy, oz], dim=-1).view(-1, 3) 
        
        # 메모리 터짐 방지를 위한 Chunk 루프 복구
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            p_chunk = sorted_pos[i:end]
            
            cell_coords = torch.floor((p_chunk - self.min_bound) / self.cell_size).long()
            
            neighbor_coords = cell_coords.unsqueeze(1) + neighbor_offsets.unsqueeze(0)
            valid_cells_mask = ((neighbor_coords >= 0) & (neighbor_coords < self.grid_dims)).all(dim=-1)
            neighbor_coords = neighbor_coords.clamp(0, self.grid_dims - 1)
            
            neighbor_keys = (neighbor_coords[..., 0] * (self.grid_dims ** 2) + 
                             neighbor_coords[..., 1] * self.grid_dims + 
                             neighbor_coords[..., 2])
                             
            gathered_indices = cell_particle_idx[neighbor_keys]
            gathered_masks = cell_mask[neighbor_keys] & valid_cells_mask.unsqueeze(-1)
            
            neighbor_pos = sorted_pos[gathered_indices] 
            diff = neighbor_pos - p_chunk.view(-1, 1, 1, 3)
            dist = torch.norm(diff, dim=-1)
            
            dist = torch.where(gathered_masks, dist, torch.tensor(self.R + 1.0, device=self.device))
            
            kernel_val = self.poly6_kernel(dist, self.R)
            rho_p[i:end] = kernel_val.sum(dim=(1, 2))
            
        return rho_p

    # ---------------------------------------------------------
    # 메인 파이프라인
    # ---------------------------------------------------------
    def compute_particle_density(self, sorted_positions: torch.Tensor, 
                                 sorted_keys: torch.Tensor, 
                                 start_indices: torch.Tensor, 
                                 end_indices: torch.Tensor) -> torch.Tensor:
        """
        각 파티클의 밀도 계산
        ρ_p = Σ_{q ∈ N_p} W(||x_p - x_q||, R)
        
        Args:
            sorted_positions: (N, 3) 정렬된 파티클 좌표
        
        Returns:
            rho_p: (N,) 밀도 값
        """
        if sorted_positions.shape[0] == 0:
            return torch.zeros(0, device=self.device)
        return self._compute_density_hashed(sorted_positions, sorted_keys, start_indices, end_indices)
    
    def create_grid(self, offset: tuple = (0.0, 0.0, 0.0)) -> Tuple[torch.Tensor, tuple]:
        """
        파티클 위치 기반 그리드 생성
        
        Returns:
            grid_nodes: (G, 3) 그리드 노드 좌표
            grid_shape: (Gx, Gy, Gz) 그리드 크기
        """
        steps = int(np.round((self.max_bound - self.min_bound) / self.dx)) + 1
        
        # offset을 더하여 그리드 전체 좌표를 이동시킵니다.
        x = torch.linspace(self.min_bound, self.max_bound, steps, device=self.device) + offset[0]
        y = torch.linspace(self.min_bound, self.max_bound, steps, device=self.device) + offset[1]
        z = torch.linspace(self.min_bound, self.max_bound, steps, device=self.device) + offset[2]

        grid_nodes = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
        grid_shape = (steps, steps, steps)
        grid_nodes = grid_nodes.reshape(-1, 3)

        return grid_nodes, grid_shape
    
    def compute_grid_features(self, sorted_positions: torch.Tensor, rho_p: torch.Tensor, 
                              grid_shape: tuple, chunk_size: int = 20000, 
                              offset: tuple = (0.0, 0.0, 0.0)) -> torch.Tensor:
        """
        그리드 노드 특징값 계산 (메모리 절약을 위해 Chunk 단위 처리)
        m_c = Σ_{p ∈ N_c} (1/ρ_p) * W(||x_c - x_p||, R)
        
        Args:
            sorted_positions: (N, 3) 정렬된 파티클 좌표
            rho_p: (N,) 파티클 밀도
            grid_shape: (Gx, Gy, Gz) 그리드 크기
            chunk_size: 한 번에 처리할 노드 개수
            offset: (추가) 엇갈린 그리드 생성을 위한 오프셋
        
        Returns:
            m_c: (G,) 그리드 특징값
        """
        rho_p = torch.clamp(rho_p, min=1e-6)
        
        # 1. 3D m_c 그리드 빈 도화지 생성 (역전파/가속에 유리한 1D 평탄화 사용)
        total_cells = grid_shape[0] * grid_shape[1] * grid_shape[2]
        m_c_flat = torch.zeros(total_cells, device=self.device)
        
        # 2. 커널 반경(R)이 닿는 격자 칸 수 계산 (R = 3*dx 이므로 3칸)
        rad_cells = int(np.ceil(self.R / self.dx)) 
        
        # 3. 파티클 주변의 국소 그리드 템플릿(343개 오프셋) 생성
        offsets = torch.arange(-rad_cells, rad_cells + 1, device=self.device)
        ox, oy, oz = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
        local_idx = torch.stack([ox, oy, oz], dim=-1).view(-1, 3)  # (343, 3)
        
        N = sorted_positions.shape[0]

        # 🚨 [수정됨] 기본 경계에 오프셋을 더해 기준점을 이동시킵니다.
        offset_tensor = torch.tensor(offset, device=self.device)
        base_bound = self.min_bound + offset_tensor
        
        # 메모리 효율을 위해 파티클도 Chunk 단위로 나누어 스플래팅
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            p_chunk = sorted_positions[i:end]
            rho_chunk = rho_p[i:end]
            
            # 파티클이 위치한 가장 가까운 '중앙 그리드 인덱스' 역산
            # 🚨 [수정됨] self.min_bound 대신 base_bound(이동된 경계) 사용
            idx_float = (p_chunk - base_bound) / self.dx
            idx_base = torch.round(idx_float).long()  # (Chunk, 3)
            
            # 중앙 인덱스에 343개의 템플릿 오프셋을 더해 '주변 이웃 인덱스들' 계산
            # (Chunk, 1, 3) + (1, 343, 3) = (Chunk, 343, 3)
            neighbor_idx = idx_base.unsqueeze(1) + local_idx.unsqueeze(0)
            
            # 그리드 영역 밖으로 삐져나간 인덱스 필터링 (마스킹)
            valid_x = (neighbor_idx[..., 0] >= 0) & (neighbor_idx[..., 0] < grid_shape[0])
            valid_y = (neighbor_idx[..., 1] >= 0) & (neighbor_idx[..., 1] < grid_shape[1])
            valid_z = (neighbor_idx[..., 2] >= 0) & (neighbor_idx[..., 2] < grid_shape[2])
            valid_mask = valid_x & valid_y & valid_z  # (Chunk, 343)
            
            # 실제 그리드 노드의 월드 좌표(x, y, z) 계산
            # 🚨 [수정됨] self.min_bound 대신 base_bound(이동된 경계) 사용
            neighbor_pos = base_bound + neighbor_idx.float() * self.dx
            
            # 파티클과 주변 343개 이웃 간의 정확한 유클리디안 거리 계산
            diff = neighbor_pos - p_chunk.unsqueeze(1)
            dist = torch.norm(diff, dim=-1)  # (Chunk, 343)
            
            # 커널 값 계산 및 밀도(rho) 적용
            kernel_val = self.poly6_kernel(dist, self.R)
            weighted_kernel = kernel_val / rho_chunk.unsqueeze(1)
            
            # 유효한 인덱스에만 값을 남기고 밖으로 삐져나간 곳은 0으로 처리
            valid_weights = weighted_kernel[valid_mask]
            
            # PyTorch index_put_ 연산을 위해 (x, y, z) 3D 인덱스를 1D 평면 인덱스로 쫙 폅니다.
            flat_indices = (neighbor_idx[..., 0] * grid_shape[1] * grid_shape[2] +
                            neighbor_idx[..., 1] * grid_shape[2] +
                            neighbor_idx[..., 2])
            valid_flat_indices = flat_indices[valid_mask]
            
            # 🚨 [핵심] 빈 도화지에 계산된 특징값들을 한 번에 더해줍니다 (Splatting!)
            m_c_flat.index_put_((valid_flat_indices,), valid_weights, accumulate=True)
            
        return m_c_flat
    
    def __call__(self, particle_positions: torch.Tensor, offset: tuple = (0.0, 0.0, 0.0)) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        전체 전처리 파이프라인
        
        Args:
            particle_positions: (N, 3) 파티클 좌표
            offset: 엇갈린 그리드 생성을 위한 오프셋
        
        Returns:
            grid_nodes: (G, 3) 그리드 노드 좌표
            m_c: (G,) 그리드 특징값
            grid_shape: (Gx, Gy, Gz) 그리드 크기
        """
        particle_positions = particle_positions.float()
        
        # 1. 공간 해싱 및 공간 정렬 (정렬된 데이터로 전체 파이프라인을 관통하여 VRAM/속도 최적화)
        sorted_positions, sorted_keys = self._spatial_hash_and_sort(particle_positions)
        start_indices, end_indices = self._build_cell_offsets(sorted_keys)
        
        # 2. 밀도 계산
        rho_p = self.compute_particle_density(sorted_positions, sorted_keys, start_indices, end_indices)
        
        # 3. 그리드 생성 및 피처 계산 (Splatting)
        grid_nodes, grid_shape = self.create_grid(offset=offset)
        m_c = self.compute_grid_features(sorted_positions, rho_p, grid_shape, offset=offset)
        
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
        super().__init__()
        
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
        self.fc3 = nn.Linear(128, 27)
        
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 1, 8, 8, 8) 특징값 블록
        
        Returns:
            sdf: (Batch, 27) 주변 3x3x3 영역의 SDF 값
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

    def train_step(self, train_loader, optimizer, criterion, device='cuda', num_epochs=5, max_batches_per_epoch=None):
        
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
            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{num_epochs:02d}]")
            
            for batch_idx, (feature_patch, sdf_target) in enumerate(progress_bar):
                # 조기 종료 조건
                if max_batches_per_epoch and batch_idx >= max_batches_per_epoch:
                    break
                    
                # 군더더기 없이 GPU로 직행
                feature_patch = feature_patch.to(device)
                sdf_target = sdf_target.to(device)
                
                optimizer.zero_grad()
                pred_sdf = self.forward(feature_patch)
                loss = criterion(pred_sdf, sdf_target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * feature_patch.size(0)
                
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix({'loss': f"{loss.item():.5f}"})
                    
            # 정확한 평균 Loss 계산
            actual_samples_seen = (batch_idx * train_loader.batch_size) if max_batches_per_epoch else len(train_loader.dataset)
            avg_loss = running_loss / actual_samples_seen
            epoch_losses.append(avg_loss)
            
            print(f"✅ Epoch {epoch+1} 완료! 평균 Loss: {avg_loss:.6f}\n")
            
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
                batch_size: int = 256) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
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
                
                # 모델 통과 (🚨 이 부분이 누락되어 추가했습니다)
                batch_sdf = self.network(batch_tensor)
                
                # 예측된 27개 값 중 타겟 노드의 정중앙 SDF 값(인덱스 13) 추출
                center_sdf = batch_sdf[:, 13]
                
                sdf_values[i:batch_end] = center_sdf
                
                if (i % (batch_size * 10)) == 0:
                    print(f"추론 진행률: {i}/{num_nodes} ({(i/num_nodes)*100:.1f}%)")

        return grid_nodes, sdf_values, m_c_grid, grid_shape
    
    def inference(self, m_c_grid_tensor: torch.Tensor, patch_size: int = 8, 
                  batch_size: int = 2048, use_pruning: bool = True) -> torch.Tensor:
        """
        단일 그리드 SDF 추론 함수. 
        매개변수 use_pruning을 통해 파티클이 없는 영역의 계산 생략 여부를 결정합니다.
        
        Args:
            m_c_grid_tensor: 입력 특징 그리드 (HxWxD)
            patch_size: 입력 패치 크기 (기본 8)
            batch_size: 한 번에 모델에 넣을 패치 개수
            use_pruning: True일 경우 파티클이 없는 영역의 CNN 연산을 건너뜁니다.
        """
        self.network.eval() 
        
        grid_shape = m_c_grid_tensor.shape
        num_nodes = grid_shape[0] * grid_shape[1] * grid_shape[2]
        
        # 1. 패치 추출을 위한 제로 패딩 (8x8x8 패치가 경계에서도 중앙 노드를 포함하도록 함)
        pad_before = patch_size // 2         
        pad_after = patch_size - pad_before - 1 
        padding_tuple = (pad_before, pad_after, pad_before, pad_after, pad_before, pad_after)
        padded_grid = F.pad(m_c_grid_tensor, padding_tuple, mode='constant', value=0)
        
        # 2. Sliding Window (Unfold)를 이용해 모든 노드에 대한 패치 일괄 생성
        # (N, 1, 8, 8, 8) 형태의 텐서로 변환
        patches = padded_grid.unfold(0, patch_size, 1).unfold(1, patch_size, 1).unfold(2, patch_size, 1)
        all_patches = patches.contiguous().view(num_nodes, 1, patch_size, patch_size, patch_size)
        
        # 3. [가지치기 로직] 연산 대상 선별
        if use_pruning:
            # 패치 내부의 m_c 합이 0에 가까우면 파티클이 없는 '허공'으로 판단
            patch_sums = all_patches.view(num_nodes, -1).abs().sum(dim=1)
            active_mask = patch_sums > 1e-6  
            
            # 결과 텐서를 1.0(허공)으로 초기화 (계산 생략 구역은 자동으로 허공 처리)
            sdf_values = torch.ones(num_nodes, device=self.device)
            active_patches = all_patches[active_mask]
            
            num_active = active_patches.size(0)
            print(f"   ✂️ [Pruning ON] {num_nodes}개 중 {num_active}개 노드 연산 ({(num_active/num_nodes)*100:.1f}%)")
        else:
            # 가지치기를 쓰지 않을 경우 모든 노드를 연산 대상으로 설정
            active_mask = torch.ones(num_nodes, dtype=torch.bool, device=self.device)
            sdf_values = torch.zeros(num_nodes, device=self.device)
            active_patches = all_patches
            print(f"   🚫 [Pruning OFF] 전체 {num_nodes}개 노드 모두 연산 중...")

        # 4. 선별된 패치들에 대해 배치 단위 추론 수행
        if active_patches.size(0) > 0:
            patch_batches = torch.split(active_patches, batch_size)
            results_list = []
            
            with torch.no_grad():
                for batch_tensor in patch_batches:
                    batch_tensor = batch_tensor.to(self.device)
                    batch_sdf = self.network(batch_tensor)
                    
                    # 예측된 27개 값 중 타겟 노드(중앙)인 13번째 인덱스만 추출
                    results_list.append(batch_sdf[:, 13])
            
            # 계산된 결과를 원래 위치(Mask)에 맞게 삽입
            sdf_values[active_mask] = torch.cat(results_list)
                
        return sdf_values.reshape(grid_shape)
    
    def staggered_inference(self, particle_positions: torch.Tensor, patch_size: int = 8, batch_size: int = 2048) -> torch.Tensor:
        """
        Multiple, staggered reconstructions
        8개의 엇갈린 그리드를 생성하고 추론하여 기존 해상도의 2배(2N x 2N x 2N) SDF cnfhs.
        """
        particle_positions = particle_positions.to(self.device).float()
        
        print("🚀 [Staggered Reconstruction] 해상도(2배) 레벨셋 복원")
        
        # 1. 속도 최적화: 파티클 간의 밀도(rho_p)는 오프셋과 무관하므로 한 번만 미리 계산합니다.
        sorted_pos, sorted_keys = self.feature_construction._spatial_hash_and_sort(particle_positions)
        start_idx, end_idx = self.feature_construction._build_cell_offsets(sorted_keys)
        rho_p = self.feature_construction.compute_particle_density(sorted_pos, sorted_keys, start_idx, end_idx)
        
        # 기본 그리드 크기(N x N x N) 파악을 위해 더미 호출
        _, base_shape = self.feature_construction.create_grid(offset=(0.0, 0.0, 0.0))
        
        # 결과물을 담을 2배 해상도(2N x 2N x 2N) 텐서 준비
        high_res_shape = (base_shape[0] * 2, base_shape[1] * 2, base_shape[2] * 2)
        high_res_sdf = torch.zeros(high_res_shape, device=self.device)
        
        half_dx = self.dx / 2.0
        
        # 2. 8가지 조합(a, b, c)에 대해 오프셋 생성 및 추론 루프
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    offset = (a * half_dx, b * half_dx, c * half_dx)
                    print(f" ⏳ 엇갈린 그리드 추론 중... 조합: a={a}, b={b}, c={c} (Offset: {offset})")
                    
                    # 오프셋이 적용된 피처맵(m_c) 계산
                    m_c_flat = self.feature_construction.compute_grid_features(
                        sorted_pos, rho_p, base_shape, offset=offset
                    )
                    m_c_grid_tensor = m_c_flat.reshape(base_shape)
                    
                    # 단일 모델 초고속 추론 (N x N x N 결과 획득)
                    pred_sdf = self.inference(m_c_grid_tensor, patch_size=patch_size, batch_size=batch_size)
                    
                    # 3. 🚨 [요구사항 3] 슬라이싱을 이용한 지그재그 병합 (Interweaving)
                    # 원본 논문의 수식 (2i+a, 2j+b, 2k+c)를 파이썬 텐서 슬라이싱 구문으로 완벽히 치환
                    high_res_sdf[a::2, b::2, c::2] = pred_sdf
                    
        print(f"✅ 해상도 2배 뻥튀기 완료! 최종 형태: {high_res_sdf.shape}")
        
        return high_res_sdf