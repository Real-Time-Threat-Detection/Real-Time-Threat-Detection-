import time
import numpy as np

def PROPOSED(X, fitness, lowerbound, upperbound, Max_iterations):
    SearchAgents,dimension = X.shape

    # Initialize population
    X = lowerbound + np.random.rand(SearchAgents, dimension) * (upperbound - lowerbound)
    fit = np.array([fitness(X[i, :]) for i in range(SearchAgents)])

    # Best solution tracking
    best_idx = np.argmin(fit)
    Xbest = X[best_idx, :].copy()
    fbest = fit[best_idx]

    SOA_curve = np.zeros(Max_iterations)
    ct = time.time()
    for t in range(1, Max_iterations + 1):
        # Update SOA population (Exploration Phase)
        for i in range(SearchAgents):
            K = np.where(fit < fit[i])[0]
            if K.size != 0:
                KK = np.random.choice(K)
            else:
                KK = i
            expert = X[KK, :]

            if np.random.rand() < 0.5:
                I = np.round(1 + np.random.rand())
                RAND = np.random.rand()
            else:
                I = np.round(1 + np.random.rand(dimension))
                RAND = np.random.rand(dimension)

            X_P1 = X[i, :] + RAND * (expert - I * X[i, :])  # Eq. (3)
            X_P1 = np.clip(X_P1, lowerbound, upperbound)

            # Update position based on Eq. (4)
            F_P1 = fitness(X_P1)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1

        # Exploitation Phase (Local Search)
        for i in range(SearchAgents):
            r = -t * (0.02 / Max_iterations)   # updated
            if r < 0.5:
                X_P2 = X[i, :] + ((1 - 2 * np.random.rand(dimension)) / t) * X[i, :]  # Eq. (5)
            else:
                X_P2 = X[i, :] + lowerbound / t + np.random.rand() * (upperbound / t - lowerbound / t)  # Eq. (5)
                X_P2 = np.clip(X_P2, lowerbound / t, upperbound / t)
            X_P2 = np.clip(X_P2, lowerbound, upperbound)

            # Update position based on Eq. (6)
            F_P2 = fitness(X_P2)
            if F_P2 < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2

        # Update the best member
        best_idx = np.argmin(fit)
        if fit[best_idx] < fbest:
            fbest = fit[best_idx]
            Xbest = X[best_idx, :].copy()

        SOA_curve[t - 1] = fbest
    ct = time.time()-ct
    return fbest, SOA_curve,Xbest,ct
