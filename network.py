import torch.nn as nn
import torch
from utils import get_resnet


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs, labels):
        dist = torch.norm(outputs[0] - outputs[1], p=2, dim=1)
        loss = torch.sum(0.5 * labels.squeeze() * dist ** 2
                         + 0.5 * (1 - labels.squeeze()) * torch.clamp((self.margin - dist), min=0) ** 2)
        prediction = (dist < self.margin*0.5).float().unsqueeze(1)
        return loss, prediction


class OnlineFeatureNetwork(nn.Module):
    def __init__(self, num_outputs=128, resnet_type="rn18"):
        super(OnlineFeatureNetwork, self).__init__()
        self.resnet = get_resnet(resnet_type=resnet_type, pretrained=True)
        self.resnet._modules['avgpool'] = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.resnet._modules['fc'] = nn.Linear(in_features=self.resnet._modules['fc'].in_features, out_features=num_outputs)
        self.num_outputs = num_outputs


    def forward(self, input):
        input = input.view(-1, 3, input.shape[3], input.shape[4])
        output = self.resnet.forward(input)
        output = self.samplewise_l2norm(output)
        return output

    @staticmethod
    def samplewise_l2norm(batch):
        return batch / (torch.norm(batch, p=2, dim=1, keepdim=True))


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.running_avg = torch.Tensor([1,1]).cuda()
        self.decay_factor = 0.05

    def predict_tuplet(self, tuplet):
        dist = torch.norm(tuplet[0] - tuplet[1], p=2, dim=1)
        m = self.mean(self.running_avg)
        prediction = (dist < m)
        prediction = prediction.float().unsqueeze(1)
        return prediction

    def mean(self, avg):
        return (avg[0] + avg[1])/2.

    def forward(self, triplet):
        dist_pos = torch.norm(triplet[0] - triplet[1], p=2, dim=1)
        dist_neg = torch.norm(triplet[0] - triplet[2], p=2, dim=1)
        loss = torch.clamp((self.margin + dist_pos  ** 2 - dist_neg  ** 2), min=0)
        loss = torch.sum(loss)
        with torch.no_grad():
            self.running_avg = self.running_avg * (1 - self.decay_factor) + self.decay_factor * torch.stack((torch.mean(dist_pos), torch.mean(dist_neg)))
        return loss