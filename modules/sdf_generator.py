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

    def create_random_shape(self, seed=None):
        """
        SDF 라이브러리의 다양한 도형을 조합하여 복잡한 무작위 형태를 생성
        """
        if seed is not None:
            np.random.seed(seed)
            
        num_shapes = np.random.randint(3, 8)
        final_sdf = None
        
        shape_choices = ['sphere', 'box', 'rounded_box', 'capped_cylinder', 'capsule', 'torus', 'capped_cone']
        
        for i in range(num_shapes):
            shape_type = np.random.choice(shape_choices)
            obj = None
            
            p_start = np.random.uniform(-0.3, 0.3, 3)
            p_end = np.random.uniform(-0.3, 0.3, 3)
            
            if shape_type == 'sphere':
                obj = sphere(np.random.uniform(0.1, 0.4))
            elif shape_type == 'box':
                obj = box(np.random.uniform(0.1, 0.4, 3))
            elif shape_type == 'rounded_box':
                obj = rounded_box(np.random.uniform(0.1, 0.4, 3), np.random.uniform(0.02, 0.1))
            elif shape_type == 'capped_cylinder':
                obj = capped_cylinder(p_start, p_end, np.random.uniform(0.05, 0.2))
            elif shape_type == 'capsule':
                obj = capsule(p_start, p_end, np.random.uniform(0.05, 0.2))
            elif shape_type == 'torus':
                r_major = np.random.uniform(0.2, 0.5)
                r_minor = np.random.uniform(0.05, 0.15)
                if r_minor >= r_major: r_minor = r_major * 0.5
                obj = torus(r_major, r_minor)
            elif shape_type == 'capped_cone':
                obj = capped_cone(p_start, p_end, np.random.uniform(0.1, 0.3), np.random.uniform(0.0, 0.2))
    
            if shape_type in ['sphere', 'box', 'rounded_box', 'torus']:
                angle = np.random.uniform(0, 360)
                axis = np.random.uniform(-1, 1, 3)
                if np.linalg.norm(axis) > 0:
                    axis = axis / np.linalg.norm(axis)
                    obj = obj.rotate(angle, axis)
    
            pos = np.random.uniform(-0.2, 0.2, 3)
            obj = obj.translate(pos)
            
            if final_sdf is None:
                final_sdf = obj
            else:
                op = np.random.choice(['union', 'diff', 'smooth'], p=[0.6, 0.2, 0.2])
                if op == 'union':
                    final_sdf = final_sdf | obj
                elif op == 'diff':
                    final_sdf = final_sdf - obj
                elif op == 'smooth':
                    try:
                        final_sdf = final_sdf.smooth_union(obj, k=0.1)
                    except:
                        final_sdf = final_sdf | obj
                        
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