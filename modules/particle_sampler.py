import numpy as np
from scipy.stats import qmc

def sample_particles_poisson(sdf_grid, config, target_ppc=3.0):
    """
    [최적화됨] Bounding Box 기반 Poisson Disk 샘플링.
    도형이 있는 영역만 계산하여 병목을 해결합니다.
    """
    resolution = config.resolution
    domain_size = config.domain_size
    dx = domain_size / (resolution - 1)
    min_bound_global = -domain_size / 2.0

    # 전체 복셀 중 내부(음수)인 복셀의 비율 확인 (모니터링 용도)
    valid_voxels = np.sum(sdf_grid < 0)
    total_voxels = sdf_grid.size
    fill_ratio = valid_voxels / total_voxels
    
    print(f"   -> 도형 부피 비율: {fill_ratio*100:.1f}%")
    
    # 1. SDF 내부(음수) 영역의 인덱스 범위(Bounding Box) 찾기
    internal_coords = np.argwhere(sdf_grid < 0)
    if len(internal_coords) == 0:
        print("⚠️ 내부 영역이 없습니다.")
        return np.empty((0, 3))
    
    # BB의 최소/최대 인덱스 추출
    min_idx = internal_coords.min(axis=0)
    max_idx = internal_coords.max(axis=0)
    
    # BB의 월드 좌표 계산 (약간의 여유 공간 부여: +dx)
    bb_min_world = min_bound_global + min_idx * dx
    bb_max_world = min_bound_global + max_idx * dx
    bb_size_world = bb_max_world - bb_min_world
    
    # 2. 목표 PPC를 만족하기 위한 Poisson Disk 반지름(r) 계산 (기존 로직 유지)
    packing_factor = 0.85 
    radius = (dx / (target_ppc ** (1/3))) * packing_factor
    print(f"   -> 타겟 PPC: {target_ppc} / 고정된 반지름(r): {radius:.4f}")
    
    # [중요] 반지름을 BB의 최대 길이에 대한 비율로 변환 (qmc는 0~1 기준이므로)
    # BB가 작을수록 unit_radius는 상대적으로 커져서 샘플링이 빨라집니다.
    max_bb_dim = np.max(bb_size_world)
    if max_bb_dim == 0: return np.empty((0, 3))
    unit_radius = radius / max_bb_dim

    # 3. SciPy 엔진 초기화 및 BB 내 샘플링
    # 전체 도메인이 아닌 BB 영역만큼의 가로세로비(d=3)를 채웁니다.
    engine = qmc.PoissonDisk(d=3, radius=unit_radius, hypersphere='volume', ncandidates=10)
    print(f"   -> BB 비율: {(np.prod(bb_size_world)/domain_size**3)*100:.1f}%")
    
    # 4. 샘플링 (BBox 공간에 대해 꽉 채워서 생성)
    try:
        # BB를 0~1 공간으로 간주하고 샘플 생성
        print(f"   -> 샘플링 실행")
        sample_points = engine.fill_space()
    except Exception as e:
        print(f"⚠️ 샘플링 실패: {e}")
        return np.empty((0, 3))

    # 5. 좌표 변환 (0~1 -> BBox 월드 좌표계로 변환)
    world_points = bb_min_world + sample_points * max_bb_dim

    # BB 외부로 나가는 포인트 필터링 (max_bb_dim 사용 시 발생 가능)
    mask_in_bb = np.all((world_points >= bb_min_world) & (world_points <= bb_max_world), axis=1)
    world_points = world_points[mask_in_bb]

    # 5. SDF 내부(음수 영역)만 필터링 (최종 가위질)
    grid_indices = (world_points - min_bound_global) / dx
    grid_indices = np.round(grid_indices).astype(int)
    grid_indices = np.clip(grid_indices, 0, resolution - 1)
    
    sdf_values = sdf_grid[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
    mask_sdf = sdf_values < 0
    valid_particles = world_points[mask_sdf]
    
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