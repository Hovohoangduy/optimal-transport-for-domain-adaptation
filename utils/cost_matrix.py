import torch

def compute_cost_matrix(x, y):
    n = x.size(0)
    m = y.size(0)
    # x_norm shape: [N, 1]
    x_norm = (x**2).sum(1).view(-1, 1)
    # y_norm shape: [1, M]
    y_norm = (y**2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    dist = torch.clamp(dist, min=0.0)
    return dist

if __name__=="__main__":
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[4.0, 4.0]])

    cost = compute_cost_matrix(a, b)
    print(f"Cost calculate: {cost.item()}")