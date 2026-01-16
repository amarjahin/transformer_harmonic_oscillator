import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from olt import olt 

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

def make_batch(B, T, omegas=[0.5,2], gammas=[0.1,0.4], dt=0.1, x_scale=1.0, p_scale=1.0, device="cpu"):
    # random initial conditions
    x0 = x_scale * torch.randn(B, device=device)
    p0 = p_scale * torch.randn(B, device=device)
    state = torch.stack([x0, p0], dim=-1)  # (B, 2)
    if len(omegas) == 2:
        omegas = (omegas[1] - omegas[0]) * torch.randn(B, device=device) + omegas[0] 
    elif len(omegas) == 1: 
        omegas = torch.tensor(omegas[0], device=device)

    seq = []
    for _ in range(T + 1):  # T tokens + 1 target step
        seq.append(state)
        state = ho_step(state, omega=omegas, gamma=gammas, dt=dt)

    seq = torch.stack(seq, dim=1)      # (B, T+1, 2)
    x = seq[:, :T, :]                  # (B, T, 2)
    y = seq[:, T, :]                   # (B, 2) = (x_{t+1}, p_{t+1}) for last t in input
    return x, y

T = 1
dt = 0.1
# training
device = "cuda" if torch.cuda.is_available() else "cpu"
model = olt(d_model=2, d_head=2).to(device) # can try changing the number of heads 
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.00)

for iter in range(10000):
    x_in, y = make_batch(B=256, T=T, omegas=[2.0], gammas=[0.1], dt=dt, device=device)
    pred = model(x_in)
    loss = F.mse_loss(pred, y)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if iter % 200 == 0:
        print(iter, loss.item())

@torch.no_grad()
def rollout(model, init_seq, steps):
    # init_seq: (1, T, 2)
    seq = init_seq.clone()
    preds = []
    for _ in range(steps):
        y = model(seq)                 # (1, 2)
        preds.append(y)
        seq = torch.cat([seq[:, 1:, :], y[:, None, :]], dim=1)
    return torch.stack(preds, dim=1)   # (1, steps, 2)


steps = 200
trajectory = make_batch(B=1, T=steps+1, omegas=[2.0],gammas=[0.1], dt=dt, x_scale=10, p_scale=4, device=device)[0]
pred_trajectory = rollout(model, trajectory[:, 0:T, :], steps)


pred_len = pred_trajectory.shape[1]
t = torch.arange(pred_len, device=trajectory.device) * dt
true_x = trajectory[0, T:pred_len+T, 0]
true_p = trajectory[0, T:pred_len+T, 1]
pred_x = pred_trajectory[0, :, 0]
pred_p = pred_trajectory[0, :, 1]

fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.5))

# Top-left: phase space
axs[0, 0].plot(true_x, true_p)
axs[0, 0].plot(pred_x, pred_p, linestyle='--')
axs[0, 0].set_xlabel(r'$x$')
axs[0, 0].set_ylabel(r'$p$')
axs[0, 0].legend(['true', 'pred'])

# Top-right: p vs time
axs[0, 1].plot(t, true_p)
axs[0, 1].plot(t, pred_p, linestyle='--')
axs[0, 1].set_xlabel(r'$t$')
axs[0, 1].set_ylabel(r'$p$')

# Bottom-left: x vs time
axs[1, 0].plot(t, true_x)
axs[1, 0].plot(t, pred_x, linestyle='--')
axs[1, 0].set_xlabel(r'$t$')
axs[1, 0].set_ylabel(r'$x$')

# Bottom-right: x error vs time
x_err = (pred_x - true_x)/true_x
axs[1, 1].plot(t, x_err)
axs[1, 1].set_xlabel(r'$t$')
axs[1, 1].set_ylabel(r'$(x_{pred} - x_{true})/x_{true}$')
axs[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

for ax in axs.flat:
    # labelpad is in points; use a small negative value for tighter spacing
    ax.xaxis.labelpad = -2
    ax.yaxis.labelpad = -2

fig.subplots_adjust(wspace=0.27, hspace=0.27)
fig.tight_layout(pad=0.3)
plt.show()