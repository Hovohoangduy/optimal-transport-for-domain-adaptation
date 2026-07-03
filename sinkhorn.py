import torch

def sinhorn_knopp(M, reg, num_itermax=1000, stop_thr=1e-9):
    # We assume uniform weights for now (1/N everywhere) 
    # # Each point has the same importance.
    n, m = M.shape
    a = torch.ones(n, device=M.device) / n
    b = torch.ones(m, device=M.device) / m

    # 1. Calculation of the Gibbs Kernel (Kernel K) 
    # # This is where regularization comes into play
    K = torch.exp(-M / reg)

    # 2. Initialization of the scale vectors u and v
    u = torch.ones(n, device=M.device) / n
    v = torch.ones(m, device=M.device) / m

    # 3. Sinkhorn loop
    for i in range(num_itermax):
        v_prev = v.clone()

        # Updating u: Projection onto the lines 
        # # u = a / (K * v)
        Kv = torch.mv(K, v)
        u = a / (Kv + 1e-16)

        # Update of v: Projection onto columns 
        # # v = b / (K.T * u)
        Ktu = torch.mv(K.t(), u)
        v = b / (Ktu + 1e-16)

        # Convergence check (if v stops moving, we stop)
        err = torch.norm(v - v_prev)
        if err < stop_thr:
            break
    
    # 4. Final calculation from the transport matrix P 
    # # P = diag(u) * K * diag(v)
    P = torch.diag(u) @ K @ torch.diag(v)
    return P

if __name__=="__main__":
    M = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    reg = 0.1

    P = sinhorn_knopp(M, reg)
    print(f"Matrix transport calc: {P}")
    print(f"Sum of probability: {P.sum().item()}")