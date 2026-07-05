import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_matching_pairs(images_source, images_target, P, num_pairs=5, save_path=None):
    if isinstance(P, torch.Tensor):
        P = P.detach().cpu().numpy()
    
    matches = np.argmax(P, axis=1)
    indices = np.random.choice(len(images_source), num_pairs, replace=False)
    
    plt.figure(figsize=(6, 2 * num_pairs))
    for i, idx_source in enumerate(indices):
        idx_target = matches[idx_source]

        # image source (MNIST)
        plt.subplot(num_pairs, 2, 2*i + 1)
        img_s = images_source[idx_source].permute(1, 2, 0).cpu().numpy()
        img_s = (img_s * 0.5) + 0.5
        plt.imshow(img_s.squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Source (MNIST)")
        
        # image target (USPS)
        plt.subplot(num_pairs, 2, 2*i + 2)
        img_t = images_target[idx_target].permute(1, 2, 0).cpu().numpy()
        img_t = (img_t * 0.5) + 0.5
        plt.imshow(img_t.squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Target (USPS Match)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Image saved: {save_path}")
    plt.show()