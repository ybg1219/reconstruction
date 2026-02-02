import numpy as np
from sdf import sphere, box, cylinder

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
    3~7개의 기본 도형을 랜덤하게 결합하여 복잡한 SDF 모양을 생성합니다.
    
    Args:
        seed: 랜덤 시드 (재현 가능성 확보용)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. 도형 개수 결정 (3개 ~ 7개)
    num_shapes = np.random.randint(3, 8)
    
    # 최종 결과 변수
    final_sdf = None
    
    for i in range(num_shapes):
        # 2. 도형 타입 랜덤 선택 (구, 박스, 원기둥)
        shape_type = np.random.choice(['sphere', 'box', 'cylinder'])
        
        # 3. 크기 랜덤 설정 (0.1 ~ 0.5 사이)
        # 너무 크면 화면을 꽉 채우고, 너무 작으면 안 보임
        size_param = np.random.uniform(0.1, 0.5)
        
        if shape_type == 'sphere':
            obj = sphere(size_param)
        elif shape_type == 'box':
            # 박스는 x,y,z 비율을 다르게 해서 직육면체로 만듦
            dims = np.random.uniform(0.1, 0.5, 3)
            obj = box(dims)
        elif shape_type == 'cylinder':
            obj = cylinder(size_param)

        # 4. 회전 (Rotation) - 3D 공간감을 위해 임의의 축으로 회전
        # (원기둥이나 박스는 회전해야 자연스러움)
        angle = np.random.uniform(0, 360)
        axis = np.random.uniform(-1, 1, 3)
        axis = axis / np.linalg.norm(axis) # 단위 벡터화
        obj = obj.rotate(angle, axis)

        # 5. 위치 이동 (Translation) - 중심(-1~1) 내에서 이동
        # 너무 멀리 가면 잘리므로 -0.5 ~ 0.5 범위 내로 제한
        pos = np.random.uniform(-0.5, 0.5, 3)
        obj = obj.translate(pos)
        
        # 6. 결합 (Union vs Difference)
        if final_sdf is None:
            final_sdf = obj # 첫 번째 도형은 그대로 사용
        else:
            # 합집합(Union) 확률 70%, 차집합(Difference) 확률 30%
            # 차집합이 너무 많으면 도형이 다 사라질 수 있어서 비율 조절
            op = np.random.choice(['union', 'diff'], p=[0.7, 0.3])
            
            if op == 'union':
                final_sdf = final_sdf | obj
            else:
                final_sdf = final_sdf - obj
                
    return final_sdf