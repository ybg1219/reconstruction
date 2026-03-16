import numpy as np
from scipy.stats import qmc

import numpy as np
from scipy.stats import qmc

def sample_particles_poisson(sdf_grid, config, target_ppc=3.0):
    """
    [수정됨] 전체 개수를 고정하지 않고, 목표 PPC(밀도)를 유지하도록 파티클을 샘플링합니다.
    도형의 부피가 크면 파티클이 많이, 작으면 적게 생성됩니다.
    """
    resolution = config.resolution
    domain_size = config.domain_size
    
    # 1. 델타 x (격자 간격) 계산
    dx = domain_size / (resolution - 1)
    
    # 2. 목표 PPC를 만족하기 위한 Poisson Disk 반지름(r) 역산
    # packing_factor: 빈 공간을 고려해 거리를 살짝 좁혀서 더 촘촘히 채우는 계수 (0.8 ~ 0.85 추천)
    packing_factor = 0.85 
    radius = (dx / (target_ppc ** (1/3))) * packing_factor

    unit_radius = radius / domain_size
    
    # 전체 복셀 중 내부(음수)인 복셀의 비율 확인 (모니터링 용도)
    valid_voxels = np.sum(sdf_grid < 0)
    total_voxels = sdf_grid.size
    fill_ratio = valid_voxels / total_voxels
    
    print(f"   -> 도형 부피 비율: {fill_ratio*100:.1f}%")
    print(f"   -> 타겟 PPC: {target_ppc} / 고정된 반지름(r): {radius:.4f}")

    # 3. SciPy 엔진 초기화 (고정된 radius 사용)
    engine = qmc.PoissonDisk(d=3, radius=unit_radius, hypersphere='volume', ncandidates=30)
    
    # 4. 샘플링 (전체 공간에 대해 꽉 채워서 생성)
    try:
        sample_points = engine.fill_space()
    except Exception as e:
        print(f"⚠️ 샘플링 실패: {e}")
        return np.empty((0, 3))
        
    # 5. 좌표 변환 (0~1 -> -domain/2 ~ +domain/2)
    min_bound = -domain_size / 2.0
    max_bound = domain_size / 2.0
    world_points = min_bound + sample_points * (max_bound - min_bound)
    
    # 6. SDF 내부(음수 영역)만 필터링 (가위질)
    voxel_size = domain_size / (resolution - 1)
    grid_indices = (world_points - min_bound) / voxel_size
    grid_indices = np.round(grid_indices).astype(int)
    grid_indices = np.clip(grid_indices, 0, resolution - 1)
    
    sdf_values = sdf_grid[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
    mask = sdf_values < 0
    valid_particles = world_points[mask]
    
    # 🚨 [핵심] 이제 목표 개수(num_particles)로 자르지 않습니다! 
    # 모양이 크면 3만 개, 작으면 5천 개 등 유동적으로 반환됩니다.
    print(f"   -> 최종 생성된 파티클 수: {len(valid_particles)}개")
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