import numpy as np
from sdf import sphere, box, cylinder, rounded_box, torus, capsule, capped_cylinder, capped_cone
def create_sample_shape():
    """
    테스트용 SDF 도형 정의
    예: 구(Sphere)에서 원기둥(Cylinder)을 뺀 형태
    """
    # 반지름 0.8인 구
    s = sphere(0.8)
    # Z축 방향의 원기둥 (구멍 뚫기용)
    c = cylinder(0.4)
    
    # 구 - 원기둥 (Boolean Difference)
    final_shape = s - c
    return final_shape

def to_grid(sdf_func, resolution=64, domain_size=2.0):
    """
    SDF 함수를 받아서 3D Numpy Grid로 변환
    
    Args:
        sdf_func: sdf 라이브러리로 만든 도형 함수
        resolution: 그리드 해상도 (기본 64)
        domain_size: 공간 크기 (예: 2.0이면 -1.0 ~ 1.0 범위)
    
    Returns:
        sdf_grid: (N, N, N) 모양의 Numpy 배열
    """
    # 좌표계 생성 (-1 ~ 1 사이를 resolution 등분)
    min_bound = -domain_size / 2
    max_bound = domain_size / 2
    
    x = np.linspace(min_bound, max_bound, resolution)
    y = np.linspace(min_bound, max_bound, resolution)
    z = np.linspace(min_bound, max_bound, resolution)
    
    # 3D 그리드 좌표 생성 (indexing='ij'는 행렬 순서인 z, y, x 순서에 맞춤)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # (N, 3) 형태로 변환하여 SDF 계산
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    # 계산 후 원래 모양 (N, N, N)으로 복원
    # sdf 라이브러리는 표면=0, 내부=음수, 외부=양수 리턴
    sdf_values = sdf_func(points).reshape(resolution, resolution, resolution)
    
    return sdf_values


# =========================================================
# 랜덤 모양 생성기 (새로 추가된 부분)
# =========================================================
def create_random_shape(seed=None):
    """
    SDF 라이브러리의 다양한 도형(캡슐, 도넛, 원뿔대 등)을 
    랜덤하게 결합하여 복잡한 SDF 모양을 생성합니다.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. 도형 개수 결정
    num_shapes = np.random.randint(3, 8)
    
    final_sdf = None
    
    # 문서에 있는 3D Primitives 적극 활용
    shape_choices = ['sphere', 'box', 'rounded_box', 'capped_cylinder', 'capsule', 'torus', 'capped_cone']
    
    for i in range(num_shapes):
        shape_type = np.random.choice(shape_choices)
        
        obj = None
        
        # 공통적으로 사용할 랜덤 벡터 (시작점 a, 끝점 b)
        # 중심에서 약간 벗어난 두 지점을 잡아서 도형의 방향성을 만듦
        p_start = np.random.uniform(-0.3, 0.3, 3)
        p_end = np.random.uniform(-0.3, 0.3, 3)
        
        # --- 도형 생성 로직 ---
        if shape_type == 'sphere':
            radius = np.random.uniform(0.1, 0.4)
            obj = sphere(radius)
            
        elif shape_type == 'box':
            dims = np.random.uniform(0.1, 0.4, 3)
            obj = box(dims)
            
        elif shape_type == 'rounded_box':
            dims = np.random.uniform(0.1, 0.4, 3)
            r = np.random.uniform(0.02, 0.1)
            obj = rounded_box(dims, r)
            
        elif shape_type == 'capped_cylinder':
            # 문서: capped_cylinder(a, b, radius)
            radius = np.random.uniform(0.05, 0.2)
            obj = capped_cylinder(p_start, p_end, radius)
            
        elif shape_type == 'capsule':
            # 문서: capsule(a, b, radius)
            radius = np.random.uniform(0.05, 0.2)
            obj = capsule(p_start, p_end, radius)
            
        elif shape_type == 'torus':
            # 문서: torus(r1, r2) -> r1=major, r2=minor
            r_major = np.random.uniform(0.2, 0.5)
            r_minor = np.random.uniform(0.05, 0.15)
            if r_minor >= r_major: r_minor = r_major * 0.5
            obj = torus(r_major, r_minor)
            
        elif shape_type == 'capped_cone':
            # 문서: capped_cone(a, b, ra, rb)
            ra = np.random.uniform(0.1, 0.3)
            rb = np.random.uniform(0.0, 0.2) # 0이면 뾰족한 원뿔
            obj = capped_cone(p_start, p_end, ra, rb)

        # 4. 회전 (Rotation) 
        # a, b 점을 사용하는 도형(캡슐, 원기둥 등)은 이미 회전된 상태지만,
        # 구, 박스, 도넛은 추가 회전이 필요할 수 있음
        if shape_type in ['sphere', 'box', 'rounded_box', 'torus']:
            angle = np.random.uniform(0, 360)
            axis = np.random.uniform(-1, 1, 3)
            if np.linalg.norm(axis) > 0:
                axis = axis / np.linalg.norm(axis)
                obj = obj.rotate(angle, axis)

        # 5. 위치 이동 (Translation)
        # a, b를 사용하는 도형은 이미 위치가 잡혀있으므로, 추가 이동은 살짝만
        pos = np.random.uniform(-0.2, 0.2, 3)
        obj = obj.translate(pos)
        
        # 6. 결합
        if final_sdf is None:
            final_sdf = obj
        else:
            op = np.random.choice(['union', 'diff', 'smooth'], p=[0.6, 0.2, 0.2])
            if op == 'union':
                final_sdf = final_sdf | obj
            elif op == 'diff':
                final_sdf = final_sdf - obj
            elif op == 'smooth':
                # k값이 너무 크면 형태가 뭉개지므로 작게(0.05~0.1) 설정
                try:
                    final_sdf = final_sdf.smooth_union(obj, k=0.1)
                except:
                    final_sdf = final_sdf | obj

    return final_sdf