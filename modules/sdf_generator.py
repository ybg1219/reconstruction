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