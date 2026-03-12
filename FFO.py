import time
import numpy as np

def FFO(X,fitness_func, lb, ub,max_iter):
    """Fennec Fox Optimization Algorithm"""
    pop_size,dim = X.shape
    alpha = 0.2
    F = fitness_func(X)
    # Best solution initialization
    best_idx = np.argmin(F)
    best_X = X[best_idx].copy()
    best_F = F[best_idx]
    conv_ = np.zeros((max_iter))
    ct = time.time()
    for t in range(max_iter):
        R = alpha * (1 - t / max_iter) * X  # Compute neighborhood radius

        # Phase 1: Digging for prey (exploitation)
        XP1 = X + (2 * np.random.rand(pop_size, dim) - 1) * R
        XP1 = np.clip(XP1, lb, ub)  # Apply boundary constraints
        FP1 = np.apply_along_axis(fitness_func, 1, XP1)

        # Update positions
        improved = FP1 < F
        X[improved] = XP1[improved]
        F[improved] = FP1[improved]

        # Phase 2: Escape from predators (exploration)
        rand_indices = np.random.randint(0, pop_size, pop_size)
        X_rand = X[rand_indices]
        I = np.random.choice([1, 2], size=(pop_size, dim))

        XP2 = np.where(
            np.expand_dims(F[rand_indices], axis=1) < np.expand_dims(F, axis=1),
            X + np.random.rand(pop_size, dim) * (X_rand - I * X),
            X + np.random.rand(pop_size, dim) * (X - X_rand)
        )
        XP2 = np.clip(XP2, lb, ub)
        FP2 = np.apply_along_axis(fitness_func, 1, XP2)

        # Update positions
        improved = FP2 < F
        X[improved] = XP2[improved]
        F[improved] = FP2[improved]

        # Update best solution
        best_idx = np.argmin(F)
        if F[best_idx] < best_F:
            best_F = F[best_idx]
            best_X = X[best_idx].copy()
        conv_[t] = best_F
    ct = time.time()-ct

    return best_F,conv_,best_X,ct



