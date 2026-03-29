class Config:
    """전역 파라미터를 관리하는 설정 클래스입니다."""
    def __init__(self):
        # 1. 공간(도메인) 설정
        self.domain_size = 2.0
        self.resolution = 128
        
        # 2. 파티클 샘플링 설정
        self.num_particles = 40000
        
        # 3. 모델 네트워크 설정
        self.patch_size = 8
        self.dx = self.domain_size / (self.resolution - 1)

config = Config()
