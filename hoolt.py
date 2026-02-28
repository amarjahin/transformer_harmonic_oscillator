import torch
import torch.nn.functional as F
from olt import olt
from ho_dynamics import make_batch
from plotting import plot_from_checkpoint

T = 40
dt = 0.1
d_model, d_head = 32, 16
n_layers = 8
use_mlp = False
checkpoint_path = "saved_models/8la_32_16_40_nc_w.pt"
rollout_steps = 50
omegas = [1.0,4.0]
gammas = [0.1]
plot_omegas = [2.0]
plot_gammas = gammas
x_scale, p_scale = 1, 1

# Training
device = "cuda" if torch.cuda.is_available() else "cpu"
model = olt(d_model=d_model, d_head=d_head, n_layers=n_layers, use_mlp=use_mlp).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

for iter in range(5000):
    x_in, y = make_batch(B=256, T=T, omegas=omegas, gammas=gammas, dt=dt, device=device, full_seq=True)
    pred = model(x_in)  # (B, T, 2)
    loss = F.mse_loss(pred, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if iter % 200 == 0:
        print(iter, loss.item())

torch.save(model.state_dict(), checkpoint_path)

plot_from_checkpoint(
    checkpoint_path=checkpoint_path,
    d_model=d_model,
    d_head=d_head,
    n_layers=n_layers,
    use_mlp=use_mlp,
    T=T,
    steps=rollout_steps,
    omegas=plot_omegas,
    gammas=plot_gammas,
    x_scale=x_scale,
    p_scale=p_scale,
)

