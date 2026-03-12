import time
import numpy as np

def FA(particles,evaluate_fitness,lower_limit, upper_limit,num_generations):
    """Fireworks Algorithm (FWA) implementation ."""
    alpha = 0.1
    beta = 1
    delta_t = 1
    num_particles, num_dimensions = particles.shape
    F = evaluate_fitness(particles)
    # Best solution initialization
    best_idx = np.argmin(F)
    best_X = particles[best_idx].copy()
    best_F = F[best_idx]
    conv_ = np.zeros((num_generations))
    ct = time.time()
    for t in range(1, num_generations + 1):
        for i in range(num_particles):
            best_index = np.argmin(F)
            best_fitness = F[best_index]
            best_particle = particles[best_index]
            num_sparks = np.random.poisson(beta)
            sparks = best_particle + alpha * np.random.randn(num_sparks, num_dimensions) * delta_t
            all_particles = np.vstack((particles, sparks))
            all_particles = np.clip(all_particles, lower_limit, upper_limit)
            fitness = evaluate_fitness(all_particles)
            sorted_indices = np.argsort(fitness)
            particles =  all_particles[sorted_indices][:particles.shape[0]]
            best_X = best_particle
            best_F = best_fitness
        conv_[t] = best_F
    ct = time.time()-ct
    return best_X,conv_, best_F,ct


