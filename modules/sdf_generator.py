import numpy as np
from sdf import sphere, box, cylinder, rounded_box, torus, capsule, capped_cylinder, capped_cone

class SDFGenerator:
    """
    SDF(Signed Distance Field) 도형을 생성하고 3D 그리드로 래스터라이징하는 클래스입니다.
    """
    def __init__(self, config):
        """
        Args:
            config: 전역 설정을 담고 있는 Config 객체
        """
        self.resolution = config.resolution
        self.domain_size = config.domain_size

    def create_sample_shape(self):
        """
        기본 테스트용 구-원기둥 조합 도형 생성
        """
        s = sphere(0.8)
        c = cylinder(0.4)
        return s - c
    
    def create_multi_thickness_plates(self):
        """
        하나의 씬에 3가지 두께(Thin, Medium, Thick)의 판을 나란히 배치합니다.
        무작위 회전을 배제하고 고정된 각도로 비틀어 Aliasing(계단 현상)을 정밀하게 비교합니다.
        """
        # 한 화면에 3개가 들어가야 하므로 너비를 약간 줄입니다.
        width = 0.4  
        height = 0.8
        
        # 1. 3가지 두께의 판 생성 및 X축으로 나란히 이동(Translation)
        # 도메인이 -1 ~ 1 이므로, 왼쪽(-0.6), 중앙(0), 오른쪽(0.6)에 배치합니다.
        plate_thin = box([width, height, 0.01]).translate([-0.6, 0, 0])
        plate_medium = box([width, height, 0.04]) # 중앙은 이동하지 않음
        plate_thick = box([width, height, 0.15]).translate([0.6, 0, 0])
        
        # 2. 세 개의 판을 하나의 SDF 덩어리로 합치기 (Union)
        combined_plates = plate_thin | plate_medium | plate_thick
        
        # 3. 고정된 각도로 그룹 전체를 동일하게 회전
        # 딱 보기 좋게 대각선으로 30도 정도 비틀어줍니다. (난수 사용 X)
        axis = np.array([1.0, 1.0, 0.2])
        axis = axis / np.linalg.norm(axis)
        final_shape = combined_plates.rotate(30, axis)
        
        return final_shape

    def create_random_shape(self, seed=None):
        """
        SDF 라이브러리의 다양한 도형을 조합하여 복잡한 무작위 형태를 생성
        """
        if seed is not None:
            np.random.seed(seed)
            
        num_shapes = np.random.randint(4, 8) 
        final_sdf = None
        
        # 🚨 [해결책 1] 공간 분리: 도메인 절반 크기를 구하고, 도형이 벽에 닿지 않는 최대 스폰 구역을 엄격히 계산합니다.
        half_domain = self.domain_size / 2.0
        max_shape_size = 0.7  # (기존 0.35 -> 0.55) 도형 하나가 가질 수 있는 최대 뻗음 거리
        spawn_bound = half_domain - max_shape_size - 0.05 # 벽에 닿지 않는 중심점 스폰 구역
        
        # (만약 도메인 크기가 2.0이라면, 중심은 -0.4~0.4 사이에서 생기고 
        # 최대 0.55만큼 뻗어나가므로 0.95에서 딱 멈춰서 절대 벽에 닿지 않습니다!)
        
        shape_choices = ['sphere', 'box', 'thin_plate', 'rounded_box', 'capped_cylinder', 'capsule', 'torus', 'capped_cone']
        
        for i in range(num_shapes):
            shape_type = np.random.choice(shape_choices)
            obj = None
            
            p_start = np.random.uniform(-spawn_bound, spawn_bound, 3)
            p_end = np.random.uniform(-spawn_bound, spawn_bound, 3)
            
            # 🚨 [수정 2] 개별 도형들의 스펙(반지름, 가로세로 등)을 1.5배 이상 큼직하게 조정
            if shape_type == 'sphere':
                obj = sphere(np.random.uniform(0.3, max_shape_size))
            elif shape_type == 'box':
                obj = box(np.random.uniform(0.3, max_shape_size, 3))
            elif shape_type == 'thin_plate':
                # 판도 훨씬 넓적하게 만듭니다.
                obj = box([np.random.uniform(0.35, max_shape_size), np.random.uniform(0.35, max_shape_size), np.random.uniform(0.02, 0.05)])
            elif shape_type == 'rounded_box':
                obj = rounded_box(np.random.uniform(0.25, max_shape_size, 3), np.random.uniform(0.05, 0.15))
            elif shape_type == 'capped_cylinder':
                obj = capped_cylinder(p_start, p_end, np.random.uniform(0.2, max_shape_size))
            elif shape_type == 'capsule':
                obj = capsule(p_start, p_end, np.random.uniform(0.2, 0.45))
            elif shape_type == 'torus':
                # 튜브도 큼직하고 굵게
                r_major = np.random.uniform(0.3, 0.45)
                r_minor = np.random.uniform(0.05, 0.12)
                obj = torus(r_major, r_minor)
            elif shape_type == 'capped_cone':
                obj = capped_cone(p_start, p_end, np.random.uniform(0.2, 0.4), np.random.uniform(0.05, 0.15))
    
            if shape_type in ['sphere', 'box', 'thin_plate', 'rounded_box', 'torus']:
                angle = np.random.uniform(0, 360)
                axis = np.random.uniform(-1, 1, 3)
                if np.linalg.norm(axis) > 0:
                    axis = axis / np.linalg.norm(axis)
                    obj = obj.rotate(angle, axis)
                
                pos = np.random.uniform(-spawn_bound, spawn_bound, 3)
                obj = obj.translate(pos)
            
            if final_sdf is None:
                final_sdf = obj
            else:
                op = np.random.choice(['union', 'diff', 'smooth'], p=[0.7, 0.1, 0.2])
                if op == 'union':
                    final_sdf = final_sdf | obj
                elif op == 'diff':
                    final_sdf = final_sdf - obj
                elif op == 'smooth':
                    try:
                        final_sdf = final_sdf.smooth_union(obj, k=0.15) # 스무딩(융합) 범위도 살짝 넓혔습니다
                    except:
                        final_sdf = final_sdf | obj

        # 얇은 선(Thin lines) (전체 스케일에 맞춰 선도 살짝 굵게 조절)
        num_lines = np.random.randint(1, 3)
        for _ in range(num_lines):
            p1 = np.random.uniform(-spawn_bound, spawn_bound, 3)
            p2 = np.random.uniform(-spawn_bound, spawn_bound, 3)
            r = np.random.uniform(0.02, 0.05)  
            line = capsule(p1, p2, r)
            final_sdf = final_sdf | line  
            
        # 물방울 흩뿌리기
        # num_spheres = np.random.randint(2, 8)
        # droplet_bound = half_domain - 0.1
        # for _ in range(num_spheres):
        #     p = np.random.uniform(-droplet_bound, droplet_bound, 3)
        #     r = np.random.uniform(0.02, 0.08) # 큰 도형들에 묻히지 않게 물방울도 살짝 키움
        #     s = sphere(r).translate(p)
        #     final_sdf = final_sdf | s  
                        
        return final_sdf

    def to_grid(self, sdf_func):
        """
        SDF 함수를 3D Numpy Grid 배열로 래스터라이징
        """
        min_bound = -self.domain_size / 2
        max_bound = self.domain_size / 2
        
        x = np.linspace(min_bound, max_bound, self.resolution)
        y = np.linspace(min_bound, max_bound, self.resolution)
        z = np.linspace(min_bound, max_bound, self.resolution)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        
        sdf_values = sdf_func(points).reshape(self.resolution, self.resolution, self.resolution)
        return sdf_values

    def generate_random_grid(self, seed=None):
        """랜덤 도형 생성부터 그리드 변환까지 한번에 수행"""
        func = self.create_random_shape(seed=seed)
        return self.to_grid(func)