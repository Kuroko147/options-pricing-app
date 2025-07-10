import numpy as np

def simulate_cir_v_paths(v0, kappa, theta, sigma, T, N, M, seed=None):

    """
    SIMULATE VARIANVE PATHS USING THE COX-INGERSOLL-ROSS(CIR) MODEL

    dv_t = kappa * (theta - v_t) dt + sigma * sqrt(v_t) dW_t

    v0 = INITIAL VARINACE (SCALER, >= 0)
    KAPPA = MEAN REVERSION SPEED (>0)
    THETA = LONG TERM MEAN VARIANCE (>0)
    SIGMA = VOLATILITY OF VOLATILITY
    T = TIME HORIZON IN YERAS
    N = NUMBER OF TIME STEPS
    M = NUMBER OF SIMULATION PATHS
    SEED = RANDOM SEED (OPTIONAL)

    RETURNS:
    - V_PATHS = NDARRY OF SHAPE (M, N+1) WITH VARIANCE PATHS
    """
    if seed is not None:
        np.random.seed(seed)


    dt = T / N
    v_paths = np.zeros((M, N + 1))
    v_paths[:, 0] = v0

    for t in range(1, N + 1):
        vt = np.maximum(v_paths[:, t - 1], 0)

        vt = np.maximum(vt, 0)
        dW = np.random.normal(0, np.sqrt(dt), size=M)
        dv = kappa * (theta - vt) * dt + sigma * np.sqrt(vt) * dW
        v_paths[:, t] = vt + dv
        ## ENSURE VARIANCE STAYS POSITIVE
        v_paths[:, t] = np.maximum(vt + dv, 0)

    return v_paths