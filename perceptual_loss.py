import torch
import torchvision
import torch.nn as nn

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            pretrained=True
        ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # Normalize the image so that it is in the appropriate range
        h_relu1 = self.slice1(X)
        out = [h_relu1]
        return out


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = VGG19(
            requires_grad=False
        )  # Set to false so that this part of the network is frozen
        self.criterion = nn.L1Loss()
        # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights = [1.0]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1)

    def forward(self, pred_img, gt_img):

        # normalize images
        pred_img = (pred_img - self.mean.to(pred_img)) / self.std.to(pred_img)
        gt_img = (gt_img - self.mean.to(gt_img)) / self.std.to(gt_img)


        gt_fs = self.model(gt_img)
        pred_fs = self.model(pred_img)

        # Collect the losses at multiple layers (need unsqueeze in
        # order to concatenate these together)
        loss = 0
        for i in range(0, len(gt_fs)):
            loss += self.weights[i] * self.criterion(pred_fs[i], gt_fs[i])

        return loss