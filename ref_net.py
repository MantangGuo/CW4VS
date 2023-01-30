import torch
import torchvision
import torch.nn as nn


def make_layer(block, nf, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(nf))
    return nn.Sequential(*layers)


class ImNormalizer(object):
    def __init__(self, in_fmt="-11"):
        self.in_fmt = in_fmt
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def apply(self, x):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        if self.in_fmt == "-11":
            x = (x + 1) / 2
        elif self.in_fmt != "01":
            raise Exception("invalid input format")
        return (x - self.mean) / self.std


class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out


class CtxNet(nn.Module):
    def __init__(
        self, out_channels_0=64, out_channels=-1, depth=5, resnet="resnet18"
    ):
        super().__init__()

        if resnet == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=True)
        else:
            raise Exception("invalid resnet model")

        self.normalizer = ImNormalizer()

        if depth < 1 or depth > 5:
            raise Exception("invalid depth of UNet")

        encs = nn.ModuleList()
        enc_translates = nn.ModuleList()
        decs = nn.ModuleList()
        enc_channels = 0
        if depth == 5:
            encs.append(resnet.layer4)
            enc_translates.append(self.convrelu(512, 512, 1))
            enc_channels = 512
        if depth >= 4:
            encs.append(resnet.layer3)
            enc_translates.append(self.convrelu(256, 256, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 256, 256))
            enc_channels = 256
        if depth >= 3:
            encs.append(resnet.layer2)
            enc_translates.append(self.convrelu(128, 128, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 128, 128))
            enc_channels = 128
        if depth >= 2:
            encs.append(nn.Sequential(resnet.maxpool, resnet.layer1))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        if depth >= 1:
            encs.append(nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        enc_translates.append(
            nn.Sequential(self.convrelu(3, 64), self.convrelu(64, 64))
        )
        decs.append(self.convrelu(enc_channels + 64, out_channels_0))

        self.encs = nn.ModuleList(reversed(encs))
        self.enc_translates = nn.ModuleList(reversed(enc_translates))
        self.decs = nn.ModuleList(decs)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(
                out_channels_0, out_channels, kernel_size=1, padding=0
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    # disable batchnorm learning in self.encs
    def train(self, mode=True):
        super().train(mode=mode)
        if not mode:
            return
        for mod in self.encs.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad_(False)

    def forward(self, x):
        x = self.normalizer.apply(x)

        outs = [self.enc_translates[0](x)]
        for enc, enc_translates in zip(self.encs, self.enc_translates[1:]):
            x = enc(x)
            outs.append(enc_translates(x))

        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)
        x = outs.pop()

        if self.out_conv:
            x = self.out_conv(x)
        return x


class PSVNet(nn.Module):
    def __init__(self, opt):        
        super(PSVNet, self).__init__()
        self.D = opt.D
        self.relu = nn.ReLU(inplace=True)
        self.fea_conv0 = nn.Conv2d(6, 64, 3, 1, 1, bias=True)
        self.featureExtractor = make_layer(ResidualBlock, nf=64, n_layers=4)
        self.fea_conv_last = nn.Conv2d(64, 65, 3, 1, 1, bias=True)
    def forward(self, input):
        bD,_,ps,_ = input.shape
        D = self.D
        b = bD//D
        out = self.relu(self.fea_conv0(input)) #[bD,64,r_patch_size,r_patch_size]
        out = self.featureExtractor(out) #[bD,64,r_patch_size,r_patch_size]
        out = self.fea_conv_last(out) #[bD,65,r_patch_size,r_patch_size]
        feat = out[:,1:].reshape(b,D,64,ps,ps) #[b,D,64,r_patch_size,r_patch_size]
        weight = torch.nn.functional.softmax(out[:,0].mean(1).mean(1).reshape(b,D),dim=1) #[b,D]
        weight_out = torch.sum(feat*weight.view(b,D,1,1,1).expand(-1,-1,64,ps,ps),dim=1) #[b,64,r_patch_size,r_patch_size]
        return weight_out

class FlowRefNet(nn.Module):
    
    def __init__(self, opt):        
        
        super(FlowRefNet, self).__init__()
        self.cin = (opt.num_source-1)*2+3+(opt.num_source-1)*3
        self.relu = nn.ReLU(inplace=True)
        self.fea_conv0 = nn.Conv2d(self.cin, 64, 3, 1, 1, bias=True)
        self.featureExtractor = make_layer(ResidualBlock, nf=64, n_layers=4)
        self.fea_conv_last = nn.Conv2d(64, 64, 3, 1, 1, bias=True)

    def forward(self, input):
        out = self.relu(self.fea_conv0(input)) #[b,64,H,W]
        out = self.featureExtractor(out) #[b,64,H,W]
        out = self.fea_conv_last(out) #[b,64,H,W]
        return out


class ViewRefNet(nn.Module):
    
    def __init__(self, opt):        
        
        super(ViewRefNet, self).__init__()
        self.cin = 3 + opt.num_source*64 + opt.num_source*64
        self.cout = opt.cout
        self.relu = nn.ReLU(inplace=True)
        self.fea_conv0 = nn.Conv2d(self.cin, self.cout, 3, 1, 1, bias=True)
        self.featureExtractor = make_layer(ResidualBlock, nf=self.cout, n_layers=4)
        self.fea_conv_last = nn.Conv2d(self.cout, 3, 3, 1, 1, bias=True)

    def forward(self, input):
        out = self.relu(self.fea_conv0(input)) #[b,256,r_patch_size,r_patch_size]
        out = self.featureExtractor(out) #[b,256,r_patch_size,r_patch_size]
        out = self.fea_conv_last(out) #[b,3,r_patch_size,r_patch_size]
        return out + input[:,:3]