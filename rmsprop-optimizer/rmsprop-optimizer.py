import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)
    
    s_next = beta * s + (1 - beta) * (g ** 2)
    
    w_next = w - (lr / (np.sqrt(s_next) + eps)) * g
    y = w_next.tolist(), s_next.tolist()
    return tuple(y)