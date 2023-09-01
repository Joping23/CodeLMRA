import torch


def accuracy(preds, y):
    """
    :param preds:
    :param y:
    :return:
    """
    prediction_prob = torch.softmax(preds, dim=1)
    _, predicted = torch.max(prediction_prob, 1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(correct)
    return acc



def mape(preds,y):
    mape =  torch.mean(torch.abs((y - preds) / torch.abs(y)))   
    return mape
# def mape(output, target):
#     target_mean = torch.mean(target.float())
#     ss_tot = torch.sum((target - target_mean) ** 2)
#     ss_res = torch.sum((target - output) ** 2)
#     r2 = 1 - ss_res / ss_tot
#     return r2


