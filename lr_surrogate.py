import torch 
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt 
from ho_dynamics import make_batch
from nlt import nlt 

# class LinearStep(nn.Module):
#     def __init__(self, bias=True):
#         super().__init__()
#         self.lin = nn.Linear(2, 2, bias=bias)  # y = W x (+ b)

#     def forward(self, x):
#         return self.lin(x)


class LinearStep(nn.Module):
    def __init__(self, T, bias=True):
        super().__init__()
        self.T = T
        self.lin = nn.Linear(2 * T, 2, bias=bias)

    def forward(self, x):
        """
        x: (B, T, 2)   -- ordered from oldest → newest
        """
        B, T, D = x.shape
        assert D == 2 and T == self.T

        x_flat = x.reshape(B, 2 * T)  # concatenate time
        return self.lin(x_flat)

T = 40
dt = 0.1
d_model, d_head = 32, 16
n_layers = 8
use_mlp = False
checkpoint_path = "saved_models/8la_32_16_40_nc_w.pt"
rollout_steps = 50
omegas = [1.0,4.0]
gammas = [0.1]
# training
device = "cuda" if torch.cuda.is_available() else "cpu"
model = nlt(d_model=d_model, d_head=d_head, n_layers=n_layers, use_mlp=use_mlp).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
surrogate_model = LinearStep(T=T,bias=True).to(device) 
opt = torch.optim.AdamW(surrogate_model.parameters(), lr=1e-3, weight_decay=0.00)

for iter in range(10000):
    x_in, _ = make_batch(B=256, T=T, omegas=omegas, gammas=gammas, dt=dt, device=device, full_seq=True)
    with torch.no_grad():
        y = model(x_in)[:, -1, :]
    pred = surrogate_model(x_in)
    loss = F.mse_loss(pred, y)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if iter % 200 == 0:
        print(iter, loss.item())

@torch.no_grad()
def rollout_1(model, init_seq, steps):
    # init_seq: (1, T, 2)
    seq = init_seq.clone()
    preds = []
    for _ in range(steps):
        y = model(seq)                 # (1, 2)
        preds.append(y[:, None, :])
        seq = torch.cat([seq[:, 1:, :], y[:, None, :]], dim=1)
    pred_seq = torch.cat(preds, dim=1)  # (1, steps, 2)
    return torch.cat([init_seq, pred_seq], dim=1)  # (1, T+steps, 2)

@torch.no_grad()
def rollout(model, init_seq, steps):
    # init_seq: (1, T, 2)
    seq = init_seq.clone()
    preds = []
    for _ in range(steps):
        out = model(seq)               # (1, T, 2)
        y = out[:, -1, :]              # (1, 2) - next token from last position
        preds.append(y[:, None, :])
        seq = torch.cat([seq[:, 1:, :], y[:, None, :]], dim=1)
    pred_seq = torch.cat(preds, dim=1)  # (1, steps, 2)
    return torch.cat([init_seq, pred_seq], dim=1)  # (1, T+steps, 2)

steps = 50
trajectory = make_batch(
        B=1, T=steps + T, omegas=[2.0], gammas=[0.1],
        dt=dt, x_scale=1, p_scale=1, device=device, full_seq=True
    )[0]
lr_model = LinearStep(T=T, bias=True).to(device)
lr_model.load_state_dict(torch.load("saved_models/lr_40.pt", map_location=device))
lr_model.eval()

model_trajectory = rollout(model, trajectory[:, 0:T, :], steps)
surrogate_trajectory = rollout_1(surrogate_model, trajectory[:, 0:T, :], steps)
lr_trajectory = rollout_1(lr_model, trajectory[:, 0:T, :], steps)

t = torch.arange(T, steps+T, device=trajectory.device) * dt
true_x = trajectory[0, T:T+steps, 0]
true_p = trajectory[0, T:T+steps, 1]
model_x = model_trajectory[0, T:T+steps, 0]
model_p = model_trajectory[0, T:T+steps, 1]
surro_x = surrogate_trajectory[0, T:T+steps, 0]
surro_p = surrogate_trajectory[0, T:T+steps, 1]
lr_x = lr_trajectory[0, T:T+steps, 0]
lr_p = lr_trajectory[0, T:T+steps, 1]

fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.5))

# Top-left: phase space
axs[0, 0].plot(model_x, model_p)
axs[0, 0].plot(surro_x, surro_p, linestyle='--')
axs[0, 0].plot(lr_x, lr_p, linestyle=':', color='green')
axs[0, 0].plot(true_x, true_p, color='black')
axs[0, 0].set_xlabel(r'$x$')
axs[0, 0].set_ylabel(r'$p$')
axs[0, 0].legend(['model', 'surrogate', 'LR', 'true'])

# Top-right: p vs time
axs[0, 1].plot(t, model_p)
axs[0, 1].plot(t, surro_p, linestyle='--')
axs[0, 1].plot(t, lr_p, linestyle=':', color='green')
axs[0, 1].plot(t, true_p, color='black')
axs[0, 1].set_xlabel(r'$t$')
axs[0, 1].set_ylabel(r'$p$')

# Bottom-left: x vs time
axs[1, 0].plot(t, model_x)
axs[1, 0].plot(t, surro_x, linestyle='--')
axs[1, 0].plot(t, lr_x, linestyle=':', color='green')
axs[1, 0].plot(t, true_x, color='black')
axs[1, 0].set_xlabel(r'$t$')
axs[1, 0].set_ylabel(r'$x$')

# Bottom-right: x error vs time
surro_err = (surro_x - model_x) / model_x
lr_err = (lr_x - model_x) / model_x
axs[1, 1].plot(t, surro_err)
axs[1, 1].plot(t, lr_err, linestyle=':', color='green')
axs[1, 1].legend(['surrogate', 'LR (saved)'])
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