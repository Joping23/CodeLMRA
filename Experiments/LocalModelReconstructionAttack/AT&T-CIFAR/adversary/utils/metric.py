import torch


def evaluate_loss_gradient_network(net,x,y,criterion):
    with torch.no_grad():
        net.eval()
        predictions = net(x)
        loss = criterion(predictions, y)
    return loss.item()
