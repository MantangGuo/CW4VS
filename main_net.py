import torch
from ref_net import FlowRefNet
from ref_net import PSVNet
from ref_net import CtxNet
from ref_net import ViewRefNet
from mlp_net import WeightNet
from mlp_net import ConfNet
from functions import epipolar_unfold


class MainNet(torch.nn.Module):
    '''Main Network construction
    Args: 
        source_views: [b,ns,3,h,w]
        source_flows: [b,ns,ns-1,2,h,w]
        warped_source_views: [b,ns,ns-1,3,h,w]
        neighbor_xy: [b,ns, np, epipolar_length, 2]
        spa_code: [b,ns,2,(2*band_width+1)*epipolar_length,np]
        ang_code: [b,ns,6,(2*band_width+1)*epipolar_length,np]
        band_width: [1]
    '''
    def __init__(self,opt):
        super(MainNet,self).__init__()
        self.weightNet = WeightNet(opt)
        self.confNet = ConfNet(opt)
        self.ctxNet = CtxNet(depth=3)
        self.psvNet = PSVNet(opt)
        self.flowRefNet = FlowRefNet(opt)
        self.viewRefNet = ViewRefNet(opt)

        self.neighbor_size = (2*opt.band_width+1)*opt.epipolar_length
        self.cn_base = 64
        self.cn_embedding = (opt.num_source-1)*2 + self.cn_base + self.cn_base+ self.cn_base + 2 + 6

    def forward(self, batch, band_width):
        source_views = batch[1]['view'] #[b,ns,3,h,w]
        source_flows = batch[1]['flow'] #[b,ns,ns-1,2,h,w]
        psvs = batch[1]['psv'] #[b,ns,c,D,ps,ps]
        warped_source_views = batch[1]['warped_image'] #[b,ns,ns-1,3,h,w]
        source_posemaps = batch[1]['posemap'] #[b,ns,6]
        neighbor_xy = batch[1]['neighbor_xy'] # [b,ns, np, epipolar_length, 2]
        spa_code = batch[1]['spa_code'] #[b,ns,2,(2*band_width+1)*epipolar_length,np]
        ang_code = batch[1]['ang_code'] #[b,ns,6,(2*band_width+1)*epipolar_length,np]

        b,ns,c,h,w = source_views.shape
        np,el = neighbor_xy.shape[2:4]
        ps = int(np**0.5)
        D = psvs.shape[3]

        # reshape inputs
        source_posemaps = source_posemaps.reshape(b,ns,6,1,1).expand(-1,-1,-1,h,w).reshape(b*ns,6,h,w) #[b*ns,6,h,w]
        source_views = source_views.reshape(b*ns,c,h,w) #[b*ns,3,h,w]
        source_flows = source_flows.reshape(b*ns,(ns-1)*2,h,w) #[b*ns,(ns-1)*2,h,w]
        warped_source_views = warped_source_views.reshape(b*ns,(ns-1)*c,h,w) #[b*ns,(ns-1)*3,h,w]
        neighbor_xy = neighbor_xy.reshape(b*ns,np,el,2) #[b*ns,np,el,2]
        spa_code = spa_code.reshape(b*ns,2,self.neighbor_size,np) #[b*ns,2,neighbor_size,np]
        ang_code = ang_code.reshape(b*ns,6,self.neighbor_size,np) #[b*ns,6,neighbor_size,np]

        # extract content embedding
        content_embedding = self.flowRefNet(torch.cat((source_flows, 
                                                       source_views, 
                                                       warped_source_views),
                                                       dim=1)) #[b*ns, cn_base, h, w]

        # extract ctx feature from source views
        ctx = self.ctxNet(source_views) #[b*ns, cn_base, h, w]

        # epipolar unfolding
        source_views = epipolar_unfold(source_views, neighbor_xy, band_width) #[b*ns, 3, neighbor_size, np]
        ctx = epipolar_unfold(ctx, neighbor_xy, band_width) #[b*ns, cn_base, neighbor_size, np]
        content_embedding = epipolar_unfold(content_embedding, neighbor_xy, band_width) #[b*ns, cn_base, neighbor_size, np]
        source_flows = epipolar_unfold(source_flows, neighbor_xy, band_width) #[b*ns,(ns-1)*2,neighbor_size, np]

        # concate spatial, angular and geometry codes on the content embedding
        embedding = torch.cat((source_flows, 
                               content_embedding, 
                               torch.mean(content_embedding,dim=2,keepdim=True).expand(-1,-1,self.neighbor_size,-1), 
                               torch.var(content_embedding,dim=2,keepdim=True).expand(-1,-1,self.neighbor_size,-1), 
                               spa_code, 
                               ang_code),1) #[b*ns, cn_embedding, neighbor_size, np]

        # calculate weight
        weight = self.weightNet(embedding.permute(0,3,2,1).reshape(b*ns*np,self.neighbor_size,self.cn_embedding)) #[b*ns*np, neighbor_size, cn_embedding] -> [b*ns*np, neighbor_size]

        # interpolate novel view, ctx, and embedding
        interp_novel_view = torch.bmm( source_views.permute(0,3,1,2).reshape(b*ns*np*c, 1, self.neighbor_size), 
                                       weight.reshape(b*ns*np, 1, self.neighbor_size, 1).expand(-1,c,-1,-1).reshape(b*ns*np*c,self.neighbor_size, 1)) #[b*ns*np*c,1,1]

        interp_novel_ctx = torch.bmm( ctx.permute(0,3,1,2).reshape(b*ns*np*self.cn_base, 1, self.neighbor_size), 
                                      weight.reshape(b*ns*np, 1, self.neighbor_size, 1).expand(-1,self.cn_base,-1,-1).reshape(b*ns*np*self.cn_base,self.neighbor_size, 1).detach()) #[b*ns*np*cn_base,1,1]
        
        interp_novel_embedding = torch.bmm( embedding.permute(0,3,1,2).reshape(b*ns*np*self.cn_embedding, 1, self.neighbor_size), 
                                            weight.reshape(b*ns*np, 1, self.neighbor_size, 1).expand(-1,self.cn_embedding,-1,-1).reshape(b*ns*np*self.cn_embedding,self.neighbor_size, 1).detach()) #[b*ns*np*cn_embedding,1,1]

        # calculate confidence
        interp_novel_embedding = interp_novel_embedding.reshape(b,ns,np,self.cn_embedding).permute(0,2,1,3) # [b,np,ns,cn_embedding]
        confs = self.confNet(interp_novel_embedding) # [b,np,ns] 

        # blend novel view
        blended_novel_view = torch.sum(interp_novel_view.reshape(b,ns,np,c) * confs.unsqueeze(3).expand(-1,-1,-1,c).permute(0,2,1,3),dim=1)  #[b,np,c]

        # extract psv feature
        psv_feat = self.psvNet(torch.cat([psvs.permute(0,1,3,2,4,5).reshape(b*ns*D,c,ps,ps),
                                          blended_novel_view.permute(0,2,1).reshape(b,1,1,c,ps,ps).expand(-1,ns,D,-1,-1,-1).reshape(b*ns*D,c,ps,ps)],
                                          dim=1)) #[b*ns,cn_base,ps,ps]

        # novel view refinement
        blended_novel_view = blended_novel_view.permute(0,2,1).reshape(b,c,ps,ps) #[b,c,ps, ps]
        interp_novel_ctx = interp_novel_ctx.reshape(b,ns,ps,ps,self.cn_base).permute(0,1,4,2,3).reshape(b,ns*self.cn_base,ps,ps)  #[b,ns*cn_base,ps, ps]
        ref_novel_view = self.viewRefNet(torch.cat((blended_novel_view,
                                                    interp_novel_ctx,
                                                    psv_feat.reshape(b,ns*self.cn_base,ps,ps)),
                                                    dim=1)) # [b,c+ns*cn_base,ps,ps]

        return ref_novel_view, blended_novel_view, interp_novel_view.reshape(b,ns,ps,ps,c).permute(0,1,4,2,3),  weight.reshape(b,ns,ps,ps,self.neighbor_size), confs.permute(0,2,1).reshape(b,ns,ps,ps)
        
