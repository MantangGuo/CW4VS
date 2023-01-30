import torch
import os
import numpy as np
import scipy.io as scio 
import logging
import argparse
from torch.utils.data import DataLoader
from pandas import DataFrame
from data_loader import TestDataset
from main_net import MainNet
from functions import to_device
from functions import crop_image 
from functions import merge_image 
from functions import generate_samples
from functions import extract_batch
from functions import ComputeQuant


# Testing settings
parser = argparse.ArgumentParser(description="Content-aware warping for view synthesis")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--num_source", type=int, default=2, help="Number of source views")
parser.add_argument("--patch_size", type=int, default=46, help="The size of the croped view patch")
parser.add_argument("--band_width", type=int, default=0, help="The width of the epipolar line")
parser.add_argument("--epipolar_length", type=int, default=200, help="The lenghth of the epipolar line")
parser.add_argument("--test_data_path", type=str, default='./Dataset/test_DTU_RGB_18x49_flow_18x49x2x1_6dof_18x49x6_sc_18x49x2.h5', help="Path for loading testing data ")
parser.add_argument("--model_name", type=str, default='dtu_s2.pth', help="loaded model")
# network hyper-parameters
parser.add_argument("--depth_range", nargs='+', type=int, default=[425,900], help="Depth range of the dataset")
parser.add_argument("--D", type=int, default=32, help="The number of depth layers")
parser.add_argument("--cout", type=int, default=256, help="The number of network channels")
opt = parser.parse_args()

# make log dirs
exp_name = 'DTU'
exp_path = os.path.join('./logs',exp_name)

# log infomation
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
fh = logging.FileHandler(os.path.join(exp_path,'Testing.log'))
log.addHandler(fh)
logging.info(opt)

if __name__ == '__main__':

    # load data
    test_dataset = TestDataset(opt)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size,shuffle=False)
    device = torch.device("cuda:0")

    # load model
    model=MainNet(opt)
    pretrained_dict = torch.load(os.path.join(exp_path,opt.model_name))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.eval()
    to_device(model,device)


    with torch.no_grad():
        scene_name_list = []
        scene_psnr_ref = []
        scene_ssim_ref = []
        scene_psnr_blended = []
        scene_ssim_blended = []
        scene_psnr_interp = []
        scene_ssim_interp = []
     
        # inference
        for ind_scene,test_scene in enumerate(test_dataloader):

            # load samples
            scene_name = test_scene['scene_name']
            scene_name_list.append(scene_name)
            del test_scene['scene_name']
            test_scene = to_device(test_scene,device) # scene_name, views, pose_maps, Ks, Rs, Ts

            # extract current scene and its name
            pose_maps=test_scene['pose_maps'] #[b,t,h,w]
            source_clusters=test_scene['source_clusters'] #[b,t,ns]
            views=test_scene['views'] #[b,t,h,w,3]
            flows=test_scene['flows'] #[b,t,ns,ns-1,2,h,w]
            Ks=test_scene['Ks'] #[b,t,3,3]
            Rs=test_scene['Rs'] #[b,t,3,3]
            Ts=test_scene['Ts'] #[b,t,3]
            
            b,t,h,w = views.shape[:4]
            ns = source_clusters.shape[2]
            ps = opt.patch_size
            neighbor_size = (2*opt.band_width+1)*opt.epipolar_length
            ref_esti_views = views.clone() #[b,t,h,w,c]
            blended_esti_views = views.clone() #[b,t,h,w,c]
            interp_esti_views = views.unsqueeze(2).expand(-1,-1,ns,-1,-1,-1).clone() #[b,t,ns,h,w,c]
            target_view_indexes = list(np.arange(1,t-1)) #[1,2,3,...]

################################################################################################################################################################################################################          
            for ind_t in target_view_indexes:
                # extract input
                target_sample, source_sample = generate_samples(views, flows, pose_maps, Ks, Rs, Ts, source_clusters, index=ind_t, mode='test') #{view, posemap, K, R, T}
                test_samples = [target_sample, source_sample]

                # calculate patch positions
                _, left_top_xy, coordinate = crop_image(test_samples[0]['view'],ps) #[b,3,patch_size,patch_size,n]
                n=coordinate[0]*coordinate[1]
                
                # inference for patches
                ref_novel_stack = []
                blended_novel_stack = []
                interp_novel_stack = []
                for ind_n in range(n):
                    # crop views smaller patches
                    test_batch = extract_batch(test_samples, opt, left_top_xy[ind_n], mode='test')
                    ref_novel_patch,blended_novel_patch,interp_novel_patch = model(test_batch, opt.band_width)[:3]
                    ref_novel_stack.append(ref_novel_patch)
                    blended_novel_stack.append(blended_novel_patch)
                    interp_novel_stack.append(interp_novel_patch)
                ref_novel_stack = torch.stack(ref_novel_stack,dim=4) #[b,3,patch_size,patch_size,n]
                blended_novel_stack = torch.stack(blended_novel_stack,dim=4) #[b,3,patch_size,patch_size,n]
                interp_novel_stack = torch.stack(interp_novel_stack,dim=5) #[b,ns,3,patch_size,patch_size,n]

                # merge the patches to intact image
                ref_novel_view = merge_image(ref_novel_stack,coordinate) #[b,3,h_croped,w_croped]
                blended_novel_view = merge_image(blended_novel_stack,coordinate) #[b,3,h_croped,w_croped]
                interp_novel_view = merge_image(interp_novel_stack.reshape(b*ns,3,ps,ps,n),coordinate) #[b*ns,3,h_croped,w_croped]

                # replace novel view in the sequences
                h_croped,w_crop = ref_novel_view.shape[2:4]
                ref_esti_views[:,ind_t,0:h_croped,0:w_crop,:] = ref_novel_view.permute(0,2,3,1) #[b,t,h_croped,w_crop,3]
                blended_esti_views[:,ind_t,0:h_croped,0:w_crop,:] = blended_novel_view.permute(0,2,3,1) #[b,t,h_croped,w_crop,3]
                interp_esti_views[:,ind_t,:,0:h_croped,0:w_crop,:] = interp_novel_view.reshape(b,ns,3,h_croped,w_crop).permute(0,1,3,4,2) #[b,t,ns,h_croped,w_crop,3]
                print('View:', ind_t)
################################################################################################################################################################################################################
            ref_esti_views = ref_esti_views[0,:,0:h_croped,0:w_crop,:].cpu().numpy() #[t,h_croped,w_croped,c]
            blended_esti_views = blended_esti_views[0,:,0:h_croped,0:w_crop,:].cpu().numpy() #[t,h_croped,w_croped,c]
            interp_esti_views = interp_esti_views[0,:,:,0:h_croped,0:w_crop,:].cpu().numpy() #[t,ns,h_croped,w_croped,c]
            gt_views = views[0,:,0:h_croped,0:w_crop,:].cpu().numpy() #[t,h_croped,w_croped,c]

            # save
            scio.savemat(os.path.join(exp_path,scene_name[0]+'_ref.mat'),
                         {'lf_recons':ref_esti_views}) #[t,h_croped,w_croped,c]
            scio.savemat(os.path.join(exp_path,scene_name[0]+'_blended.mat'),
                         {'lf_recons':blended_esti_views}) #[t,h_croped,w_croped,c]
            scio.savemat(os.path.join(exp_path,scene_name[0]+'_interp.mat'),
                         {'lf_recons':interp_esti_views}) #[t,h_croped,w_croped,c]

            # evaluation
            # ref
            scene_psnr, scene_ssim = ComputeQuant(ref_esti_views[target_view_indexes],gt_views[target_view_indexes])
            scene_psnr_ref.append(scene_psnr)
            scene_ssim_ref.append(scene_ssim)

            # blend
            scene_psnr, scene_ssim = ComputeQuant(blended_esti_views[target_view_indexes],gt_views[target_view_indexes])
            scene_psnr_blended.append(scene_psnr)
            scene_ssim_blended.append(scene_ssim)

            # interp
            scene_psnr = 0
            scene_ssim = 0
            for i in range(interp_esti_views.shape[1]):
                temp_psnr, temp_ssim = ComputeQuant(interp_esti_views[target_view_indexes,i],gt_views[target_view_indexes])
                scene_psnr += temp_psnr
                scene_ssim += temp_ssim
            scene_psnr_interp.append(scene_psnr/interp_esti_views.shape[1])
            scene_ssim_interp.append(scene_ssim/interp_esti_views.shape[1])
            
            log.info('''
                        Index: %d  
                        Scene: %s
                        ref_PSNR: %.2f  ref_SSIM: %.3f
                        blended_PSNR: %.2f  blended_SSIM: %.3f
                        interp_PSNR: %.2f  interp_SSIM: %.3f
                    '''
                    %(ind_scene+1,
                      scene_name_list[ind_scene],
                      scene_psnr_ref[ind_scene],scene_ssim_ref[ind_scene],
                      scene_psnr_blended[ind_scene],scene_ssim_blended[ind_scene],
                      scene_psnr_interp[ind_scene],scene_ssim_interp[ind_scene]
                    ))

        log.info('''
                    Average
                    ref_PSNR: %.2f  ref_SSIM: %.3f
                    blended_PSNR: %.2f  blended_SSIM: %.3f
                    interp_PSNR: %.2f  interp_SSIM: %.3f
                 '''
                    %(np.mean(scene_psnr_ref),np.mean(scene_ssim_ref),
                      np.mean(scene_psnr_blended),np.mean(scene_ssim_blended),
                      np.mean(scene_psnr_interp),np.mean(scene_ssim_interp)
                    ))  

        # log in Excel
        scene_name_list.append('Average')
        scene_psnr_ref.append(np.mean(scene_psnr_ref))
        scene_ssim_ref.append(np.mean(scene_ssim_ref))
        scene_psnr_blended.append(np.mean(scene_psnr_blended))
        scene_ssim_blended.append(np.mean(scene_ssim_blended))
        scene_psnr_interp.append(np.mean(scene_psnr_interp))
        scene_ssim_interp.append(np.mean(scene_ssim_interp))

        data = {'Scenes': scene_name_list, 
                'psnr_ref': scene_psnr_ref, 'ssim_ref': scene_ssim_ref,
                'psnr_blended': scene_psnr_blended, 'ssim_blended': scene_ssim_blended,
                'psnr_interp': scene_psnr_interp, 'ssim_interp': scene_ssim_interp
                }
        df = DataFrame(data)
        df.to_excel(os.path.join(exp_path,'Testing.xlsx'))              