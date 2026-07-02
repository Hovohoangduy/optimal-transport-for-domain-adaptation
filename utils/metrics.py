import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
    plt.xlabel("Predicted (USPS via OT)")
    plt.ylabel("True (Hidden USPS label)")
    plt.title("Confusion Matrix: Domain Adaptation")

    if save_path:
        plt.savefig(save_path)
    plt.show()