import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
import torch

def animate_barycentric_transport(features_source, features_target, P, save_path="results/transport_animation.gif"):
    fs = features_source.detach().cpu().numpy()
    ft = features_target.detach().cpu().numpy()

    combined = np.concatenate([fs, ft], axis=0)
    pca = PCA(n_components=2)
    pca.fit(combined)

    fs_2d = pca.transform(fs)
    ft_2d = pca.transform(ft)

    P_torch = P.detach()
    ft_torch = features_target.detach()

    row_sums = P_torch.sum(dim=1, keepdim=True) * 1e-8
    P_norm = P_torch / row_sums

    # [N, M] @ [M, D] -> [N, D]
    destination_features = torch.mm(P_norm, ft_torch)
    dest_2d = pca.transform(destination_features.cpu().numpy())

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(ft_2d[:, 0], ft_2d[:, 1], c="blue", alpha=0.15, label='Target Distribution (USPS)', s=30)

    scatter_source = ax.scatter(fs_2d[:, 0], fs_2d[:, 1], c='red', alpha=0.6, label="Transported Source (MNIST)", s=20)
    ax.set_title("Optimal Transport Interpolation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    all_x = np.concatenate([fs_2d[:, 0], ft_2d[:, 0]])
    all_y = np.concatenate([fs_2d[:, 1], ft_2d[:, 1]])
    ax.set_xlim(all_x.min()-1, all_x.max()+1)
    ax.set_ylim(all_y.min()-1, all_y.max()+1)

    total_frames = 60
    
    def update(frame):
        t = frame / total_frames
        current_pos = (1 - t) * fs_2d + t * dest_2d
        scatter_source.set_offsets(current_pos)
        return scatter_source
    
    anim = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=20))
    plt.close()
    print("Animation gif save: {save_path}")