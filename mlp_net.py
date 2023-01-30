import torch
import math
import torch.nn as nn

class WeightNet(nn.Module):
    
    def __init__(self, opt):        
        
        super(WeightNet, self).__init__()
        self.cn_weight = 256
        self.cin = (opt.num_source-1)*2 + 64 + 64 + 64 + 2 + 6

 
        self.weightAgg = nn.Sequential(
            nn.Linear(self.cin, self.cn_weight),
            nn.ReLU(inplace=True),
            nn.Linear(self.cn_weight, self.cn_weight),
            nn.ReLU(inplace=True),
            nn.Linear(self.cn_weight, self.cn_weight),
            nn.ReLU(inplace=True),
            nn.Linear(self.cn_weight, self.cn_weight),
            nn.ReLU(inplace=True),
            nn.Linear(self.cn_weight, self.cn_weight),
            nn.ReLU(inplace=True),
            nn.Linear(self.cn_weight, 1)
        )

    def forward(self, features):
        '''
        features [b*N,range_disp,197]
        '''
        # regress weights
        weight = self.weightAgg(features) # [b*N,range_disp,1]
        weight = torch.nn.functional.softmax(weight.squeeze(-1), dim = 1) #[b*N,range_disp]
        return weight


class ConfNet(nn.Module):
    
    def __init__(self, opt):        
        
        super(ConfNet, self).__init__()
        self.cn_conf = 256
        self.cin = opt.num_source * (self.cn_conf//8)
        self.cn_features = (opt.num_source-1)*2 + 64 + 64 + 64 + 2 + 6

        self.confFeatAgg = nn.Sequential(
            nn.Linear(self.cn_features, self.cn_conf),
            nn.ReLU(inplace=True),
            nn.Linear(self.cn_conf, self.cn_conf//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.cn_conf//2, self.cn_conf//4),
            nn.ReLU(inplace=True),
            nn.Linear(self.cn_conf//4, self.cn_conf//8)
        )

        self.confAgg = nn.Sequential(
            nn.Linear(self.cin, self.cn_conf),
            nn.ReLU(inplace=True),
            nn.Linear(self.cn_conf, self.cn_conf//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.cn_conf//2, self.cn_conf//4),
            nn.ReLU(inplace=True),
            nn.Linear(self.cn_conf//4, self.cn_conf//8),
            nn.ReLU(inplace=True),
            nn.Linear(self.cn_conf//8, opt.num_source)
        )

    def forward(self, x):
        b,np,ns,_ = x.shape

        # regress feature
        x = self.confFeatAgg(x) #[b,np,ns,self.cn_conf//8]
        # regress confidence
        confs = self.confAgg(x.reshape(b, np, ns*(self.cn_conf//8))) # [b, np, ns] 
        confs = torch.nn.functional.softmax(confs,dim=2)
        return confs

























