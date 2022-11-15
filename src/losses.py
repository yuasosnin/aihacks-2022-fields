import torch
import torch.nn as nn
import torch.nn.functional as F


class SCELoss(nn.Module):
    """
    https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels
    """
    def __init__(self, alpha, beta, num_classes):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)

        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

    
class CDBLoss(nn.Module):
    """
    https://github.com/hitachi-rd-cv/CDB-loss/blob/main/EGTEA/losses/cdb_loss.py
    """
    def __init__(self, class_difficulty, tau='dynamic', reduction='mean', device='cpu'):
        super().__init__()
        self.tau = tau
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(reduction=self.reduction)
        self.update_weights(class_difficulty, device)
        
    def _get_tau(self, class_difficulty):
        if self.tau == 'dynamic':
            bias = (1 - torch.min(class_difficulty))/(1 - torch.max(class_difficulty) + 0.01)
            return torch.sigmoid(bias)
        else:
            return self.tau
    
    def update_weights(self, class_difficulty, device):
        tau = self._get_tau(class_difficulty)
        weights = class_difficulty ** tau
        weights = weights / weights.sum() * len(weights)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        self.loss.weight = weights

    def forward(self, input, target):
        return self.loss(input, target)
