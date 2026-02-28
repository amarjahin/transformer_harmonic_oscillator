import torch
import matplotlib.pyplot as plt
from olt import olt
from ho_dynamics import make_batch


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


def plot_pred_true_trajectories(true_x, true_p, pred_x, pred_p, t, title=None):
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
    if title is not None:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(pad=0.3)
    plt.show()


def plot_from_checkpoint(
    checkpoint_path="saved_models/8la_32_16_40_nc_w.pt",
    d_model=32,
    d_head=16,
    n_layers=8,
    use_mlp=False,
    T=40,
    steps=50,
    omegas=[2.0],
    gammas=[0.1],
    x_scale=1,
    p_scale=1,
):
    """Load trained model and plot rollout vs ground truth."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = 0.1

    model = olt(d_model=d_model, d_head=d_head, n_layers=n_layers, use_mlp=use_mlp).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    trajectory = make_batch(
        B=1, T=steps + T, omegas=omegas, gammas=gammas,
        dt=dt, x_scale=x_scale, p_scale=p_scale, device=device, full_seq=True
    )[0]
    pred_trajectory = rollout(model, trajectory[:, 0:T, :], steps)

    t = torch.arange(steps + T, device=trajectory.device) * dt
    true_x = trajectory[0, T:T + steps, 0]
    true_p = trajectory[0, T:T + steps, 1]
    pred_x = pred_trajectory[0, T:, 0]
    pred_p = pred_trajectory[0, T:, 1]

    title = rf"{n_layers}-LA model $d_{{\text{{model}}}}={d_model}$, $d_{{\text{{head}}}}={d_head}$, $T={T}$"
    plot_pred_true_trajectories(true_x, true_p, pred_x, pred_p, t[T:], title=title)


if __name__ == "__main__":
    plot_from_checkpoint()