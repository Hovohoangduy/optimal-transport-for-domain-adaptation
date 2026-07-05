import torch
import os
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from utils.mnist_usps_dataloader import get_data_loaders
from feature_extractor import FeatureExtractor
from utils.cost_matrix import compute_cost_matrix
from sinkhorn import sinhorn_knopp
from utils.visualization import plot_matching_pairs
from utils.metrics import plot_confusion_matrix
from utils.animation import animate_barycentric_transport

def one_hot_encoding(labels, num_classes=10):
    return torch.eye(num_classes)[labels].to(labels.device)

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_path = "results/"
    model_path = "saved_models/mnist_extractor.pth"

    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    
    source_loader, target_loader = get_data_loaders(batch_size=32)
    batch_s = next(iter(source_loader))
    batch_t = next(iter(target_loader))

    x_s, y_s = batch_s[0].to(DEVICE), batch_s[1].to(DEVICE)
    x_t, y_t = batch_t[0].to(DEVICE), batch_t[1].to(DEVICE)

    model = FeatureExtractor().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Load model DONE!")
    else:
        print("Don't loading model!!")
    
    # feature and OT
    print("Extraction and Normalization...")
    with torch.no_grad():
        feat_s = F.normalize(model(x_s), p=2, dim=1)
        feat_t = F.normalize(model(x_t), p=2, dim=1)
    
    print("Sinkhorn ...")
    C = compute_cost_matrix(feat_s, feat_t)
    P = sinhorn_knopp(C, reg=0.1)

    y_s_oh = one_hot_encoding(y_s)
    y_t_soft = torch.mm(P.t(), y_s_oh)
    y_t_pred = torch.argmax(y_t_soft, dim=1)

    acc = accuracy_score(y_t.cpu().numpy(), y_t_pred.cpu().numpy())
    print(f"Accuracy: {acc*100:.2f}")
    
    save_match = os.path.join(visual_path, "matches.png")
    plot_matching_pairs(x_s, x_t, P, num_pairs=5, save_path=save_match)
    save_cm = os.path.join(visual_path, "confusion_matrix.png")
    plot_confusion_matrix(y_t.cpu().numpy(), y_t_pred.cpu().numpy(), save_path=save_cm)

    save_anim = os.path.join(visual_path, "transport_viz.gif")
    animate_barycentric_transport(feat_s, feat_t, P, save_path=save_anim)

if __name__=="__main__":
    main()