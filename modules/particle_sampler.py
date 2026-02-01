"""
Phase 3: Particle Sampler
- SDF 내부 파티클 샘플링 (Dart Throwing)
"""
import numpy as np

def sample_particles(sdf_grid, spacing=0.1, max_trials=10000):
    """
    SDF < 0 영역에 파티클 샘플링 (간단한 dart throwing)
    Returns: (M,3) ndarray
    """
    res = sdf_grid.shape[0]
    coords = np.array(np.where(sdf_grid < 0)).T / res
    particles = []
    for _ in range(max_trials):
        idx = np.random.randint(0, len(coords))
        p = coords[idx]
        if all(np.linalg.norm(p - np.array(q)) > spacing for q in particles):
            particles.append(p)
        if len(particles) > 0 and len(particles) % 100 == 0:
            break
    return np.array(particles)
