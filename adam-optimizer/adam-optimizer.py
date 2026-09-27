import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    params_new = []
    m_new = []
    v_new = []
    
    for p, g, mi, vi in zip(param, grad, m, v):
        # Step 1: Update First Moment
        m_next = beta1 * mi + (1 - beta1) * g
        
        # Step 2: Update Second Moment
        v_next = beta2 * vi + (1 - beta2) * (g ** 2)
        
        # Step 3: Bias Correction
        m_hat = m_next / (1 - (beta1 ** t))
        v_hat = v_next / (1 - (beta2 ** t))
        
        # Step 4: Parameter Update
        p_next = p - lr * m_hat / ((v_hat ** 0.5) + eps)
        
        params_new.append(p_next)
        m_new.append(m_next)
        v_new.append(v_next)
        
    return params_new, m_new, v_new