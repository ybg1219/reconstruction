"""
Phase 3: Particle Sampler
- SDF 내부 파티클 샘플링 (Dart Throwing)
"""
import numpy as np
from scipy.stats import qmc
from modules.sdf_generator import to_grid # SDF 값 확인을 위해 필요하지 않으나, 원리상 SDF 함수가 필요

def sample_particles_poisson(sdf_grid, domain_size=2.0, num_particles=10000):
    """
    SDF 내부 부피를 고려하여 목표 개수(num_particles)만큼 파티클을 생성합니다.
    """
    resolution = sdf_grid.shape[0]
    
    # [핵심 수정 1] 도형이 차지하는 비율(Fill Ratio) 계산
    # 전체 복셀 중 내부(음수)인 복셀의 비율을 구합니다.
    valid_voxels = np.sum(sdf_grid < 0)
    total_voxels = sdf_grid.size
    
    if valid_voxels == 0:
        print("⚠️ 경고: SDF 내부에 공간이 없습니다 (너무 작거나 없음).")
        return np.empty((0, 3))
        
    fill_ratio = valid_voxels / total_voxels
    
    # [핵심 수정 2] 전체 부피가 아닌 '도형 부피'를 기준으로 반지름 설정
    # 전체 박스 부피
    total_volume = domain_size ** 3
    
    # 실제 도형의 추정 부피
    shape_volume = total_volume * fill_ratio
    
    # 목표 개수를 맞추기 위한 반지름 계산
    # (도형 부피 안에 num_particles개가 들어가도록 밀도 설정)
    # * 0.45는 안전 계수 (목표보다 약간 더 많이 생성해서 잘라내기 위함)
    radius = (shape_volume / num_particles) ** (1/3) * 0.45
    
    print(f"   -> 부피 비율: {fill_ratio*100:.1f}%")
    print(f"   -> 목표: {num_particles}개 / 보정된 반지름(r): {radius:.4f}")

    # 2. SciPy 엔진 초기화
    engine = qmc.PoissonDisk(d=3, radius=radius, hypersphere='volume', ncandidates=30)
    
    # 3. 샘플링 (전체 공간 채우기)
    try:
        sample_points = engine.fill_space()
    except Exception as e:
        print(f"⚠️ 샘플링 실패: {e}")
        return np.empty((0, 3))
        
    # 4. 좌표 변환 (0~1 -> -domain/2 ~ +domain/2)
    min_bound = -domain_size / 2.0
    max_bound = domain_size / 2.0
    world_points = min_bound + sample_points * (max_bound - min_bound)
    
    # 5. SDF 내부 필터링
    voxel_size = domain_size / (resolution - 1)
    grid_indices = (world_points - min_bound) / voxel_size
    grid_indices = np.round(grid_indices).astype(int)
    grid_indices = np.clip(grid_indices, 0, resolution - 1)
    
    sdf_values = sdf_grid[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
    
    mask = sdf_values < 0
    valid_particles = world_points[mask]
    
    # [추가] 너무 많이 생성됐으면 목표 개수만큼 자르기 (정확한 개수 맞춤)
    if len(valid_particles) > num_particles:
        # 랜덤하게 섞어서 자르거나 앞에서부터 자름
        valid_particles = valid_particles[:num_particles]
        
    return valid_particles

def sample_particles_jittered(sdf_grid, domain_size=2.0, jitter_scale=0.8):
    """
    SDF 그리드 내부(음수 영역)에 파티클을 생성합니다.
    (Jittered Grid Sampling 방식 적용)
    
    Args:
        sdf_grid: (N, N, N) 크기의 SDF Numpy 배열
        domain_size: 물리적 공간 크기 (Phase 1과 동일하게 맞춰야 함, 기본 2.0)
        jitter_scale: 0.0이면 정가운데 정렬, 1.0이면 복셀 꽉 채워서 랜덤
        
    Returns:
        particles: (M, 3) 크기의 파티클 좌표 배열 (Numpy)
    """
    resolution = sdf_grid.shape[0]
    
    # 1. SDF 값이 0보다 작은(내부) 인덱스만 추출
    # indexing='ij'를 사용했으므로 indices[0]=x, indices[1]=y, indices[2]=z
    indices = np.where(sdf_grid < 0)
    
    # (3, M) -> (M, 3) 형태로 변환 (x, y, z 순서)
    grid_indices = np.stack(indices, axis=-1).astype(np.float32)
    
    if len(grid_indices) == 0:
        print("⚠️ 경고: SDF 내부에 공간이 없습니다. 파티클이 생성되지 않습니다.")
        return np.empty((0, 3))

    # 2. 물리적 좌표로 변환 (Index -> World Coordinate)
    # voxel_size: 한 칸의 물리적 길이
    voxel_size = domain_size / (resolution - 1)
    min_bound = -domain_size / 2.0
    
    # 기본 위치: 각 복셀의 정중앙 (indices * size + min)
    base_positions = min_bound + grid_indices * voxel_size
    
    # 3. Jittering (무작위 흔들기) - Poisson Disk 효과 흉내
    # -0.5 ~ 0.5 사이의 랜덤 값 생성
    noise = (np.random.rand(*base_positions.shape) - 0.5)
    
    # 위치에 노이즈 추가
    particles = base_positions + (noise * voxel_size * jitter_scale)
    
    return particles

def sample_particles_dart_throwing(sdf_grid, domain_size=2.0, spacing=0.1, max_trials=10000):
    """
    SDF < 0 영역에 파티클 샘플링 (Dart Throwing) - World Coordinate 수정판
    
    Args:
        sdf_grid: (N,N,N) SDF 배열
        domain_size: 물리적 공간 크기 (기본 2.0 -> -1.0 ~ 1.0 범위)
        spacing: 파티클 간 최소 간격 (World Unit 기준)
        max_trials: 최대 시도 횟수
    """
    res = sdf_grid.shape[0]
    
    # 1. 내부 영역 인덱스 추출
    indices = np.argwhere(sdf_grid < 0) # (K, 3) 인덱스 배열
    
    # 2. 인덱스 -> 월드 좌표 변환 계수 미리 계산
    # 공식: min + index * (domain / (res-1))
    min_bound = -domain_size / 2.0
    voxel_size = domain_size / (res - 1)
    
    particles = []
    
    # 인덱스 리스트에서 무작위로 뽑아서 시도
    for _ in range(max_trials):
        if len(indices) == 0: break
            
        # 내부 복셀 중 하나 랜덤 선택
        rand_idx = np.random.randint(0, len(indices))
        idx_point = indices[rand_idx]
        
        # [핵심 수정] 월드 좌표로 변환 (-1.0 ~ 1.0)
        # 약간의 랜덤성(Jitter)을 주어 격자 무늬 방지
        jitter = (np.random.rand(3) - 0.5) # -0.5 ~ 0.5
        p = min_bound + (idx_point + jitter) * voxel_size
        
        # 거리 검사 (기존 파티클들과 비교)
        # (주의: 파티클이 많아지면 이 부분 속도가 매우 느려짐)
        is_valid = True
        for q in particles:
            if np.linalg.norm(p - q) < spacing:
                is_valid = False
                break
        
        if is_valid:
            particles.append(p)
            
        # 너무 많이 모이면 중단 (옵션)
        if len(particles) >= 2000: # 예시로 2000개 제한
            break
            
    return np.array(particles)