import numpy as np
from scipy.stats import qmc

def sample_particles_poisson(sdf_grid, config):
    """
    SDF 내부 부피를 고려하여 목표 개수만큼 파티클을 생성합니다.
    """
    resolution = config.resolution
    domain_size = config.domain_size
    num_particles = config.num_particles
    
    # 전체 복셀 중 내부(음수)인 복셀의 비율을 구합니다.
    valid_voxels = np.sum(sdf_grid < 0)
    total_voxels = sdf_grid.size
    
    if valid_voxels == 0:
        print("⚠️ 경고: SDF 내부에 공간이 없습니다 (너무 작거나 없음).")
        return np.empty((0, 3))
        
    fill_ratio = valid_voxels / total_voxels
    
    total_volume = domain_size ** 3
    
    # 실제 도형의 추정 부피
    shape_volume = total_volume * fill_ratio
    
    # 목표 개수를 맞추기 위한 반지름 계산 (* 0.45는 안전 계수)
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
    
    # 목표 개수만큼 자르기
    if len(valid_particles) > num_particles:
        valid_particles = valid_particles[:num_particles]
        
    return valid_particles

def sample_particles_jittered(sdf_grid, config, jitter_scale=0.8):
    """
    SDF 그리드 내부에 파티클을 생성 (Jittered Grid Sampling)
    """
    resolution = config.resolution
    domain_size = config.domain_size
    
    indices = np.where(sdf_grid < 0)
    
    # (3, M) -> (M, 3) 형태로 변환 (x, y, z 순서)
    grid_indices = np.stack(indices, axis=-1).astype(np.float32)
    
    if len(grid_indices) == 0:
        print("⚠️ 경고: SDF 내부에 공간이 없습니다.")
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