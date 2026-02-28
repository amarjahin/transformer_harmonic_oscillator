import torch 

def ho_step(state, omega, gamma, dt):
    """
    Damped harmonic oscillator with m=1:
        x' = p
        p' = -omega^2 x - 2*gamma p

    Exact discrete-time map for the underdamped case (gamma < omega).
    state: (..., 2) where [...,0]=x and [...,1]=p
    omega, gamma, dt: broadcastable tensors or floats
    """
    x, p = state[..., 0], state[..., 1]

    # Ensure tensors on same device/dtype
    omega = torch.as_tensor(omega, device=state.device, dtype=state.dtype)
    gamma = torch.as_tensor(gamma, device=state.device, dtype=state.dtype)
    dt    = torch.as_tensor(dt,    device=state.device, dtype=state.dtype)

    Omega = torch.sqrt(torch.clamp(omega**2 - gamma**2, min=0.0))  # underdamped if > 0
    e = torch.exp(-gamma * dt)

    c = torch.cos(Omega * dt)
    s = torch.sin(Omega * dt)

    # Avoid division by zero if Omega is extremely small (near critical)
    eps = torch.tensor(1e-12, device=state.device, dtype=state.dtype)
    Omega_safe = torch.where(Omega > eps, Omega, eps)

    x_next = e * ( x * (c + (gamma / Omega_safe) * s) + p * (s / Omega_safe) )
    p_next = e * (
        - x * (omega**2 / Omega_safe) * s
        + p * (c - (gamma / Omega_safe) * s)
    )

    return torch.stack([x_next, p_next], dim=-1)

def make_batch(B, T, omegas=[0.5,2], gammas=[0.1,0.4], dt=0.1, x_scale=1.0, p_scale=1.0, device="cpu", full_seq=False):
    # random initial conditions
    x0 = x_scale * torch.rand(B, device=device)
    p0 = p_scale * torch.rand(B, device=device)
    state = torch.stack([x0, p0], dim=-1)  # (B, 2)
    if len(omegas) == 2:
        omegas = (omegas[1] - omegas[0]) * torch.rand(B, device=device) + omegas[0] 
    elif len(omegas) == 1: 
        omegas = torch.tensor(omegas[0], device=device)

    if len(gammas) == 2:
        gammas = (gammas[1] - gammas[0]) * torch.rand(B, device=device) + gammas[0] 
    elif len(gammas) == 1: 
        gammas = torch.tensor(gammas[0], device=device)

    seq = []
    seq.append(state)
    for _ in range(T):
        state = ho_step(state, omega=omegas, gamma=gammas, dt=dt)
        seq.append(state)

    seq = torch.stack(seq, dim=1)
    x = seq[:, :T, :]   # (B, T, 2) input
    if full_seq:
        y = seq[:, 1:T+1, :]  # (B, T, 2) target at each pos = next token
    else:
        y = seq[:, T, :]      # (B, 2)
    return x, y