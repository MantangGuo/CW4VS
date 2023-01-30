import torch
import torchvision
import logging
import itertools
import argparse
import matplotlib
import os
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from kornia.losses import ssim_loss
from collections import defaultdict
from data_loader import TrainingDataset,ValidationDataset
from functions import to_device, weights_init, setup_seed, extract_batch, WeightSmoothLoss, calculate_psnr,l1_loss
from main_net import MainNet
from perceptual_loss import PerceptualLoss


parser = argparse.ArgumentParser(description="Light Field Compressed Sensing")
# Training settings
parser.add_argument("--learningRate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--num_source", type=int, default=2, help="Number of source views")
parser.add_argument("--patch_size", type=int, default=32, help="The size of the croped view patch")
parser.add_argument("--band_width", type=int, default=0, help="The width of the epipolar line")
parser.add_argument("--epipolar_length", type=int, default=200, help="The lenghth of the epipolar line")
parser.add_argument("--epochNum", type=int, default=10001, help="Total number of epochs")
parser.add_argument("--validNum", type=int, default=1000, help="The number of epochs for validation")
parser.add_argument("--saveNum", type=int, default=1000, help="The number of epochs for saving model")
parser.add_argument("--logNum", type=int, default=50, help="The number of epochs for saving logs")
# network hyper-parameters
parser.add_argument("--depth_range", nargs='+', type=int, default=[425,900], help="Depth range of the dataset")
parser.add_argument("--D", type=int, default=32, help="The number of depth layers")
parser.add_argument("--cout", type=int, default=256, help="The number of network channels")
# log information
parser.add_argument("--summaryPath", type=str, default='./logs/', help="Path for saving logs")
parser.add_argument("--training_data_path", type=str, default='./Dataset/train_DTU_RGB_79x49_flow_79x49x2x1_6dof_79x49x6_sc_79x49x2.h5', help="Path for loading training data ")
parser.add_argument("--validation_data_path", type=str, default='./Dataset/train_DTU_RGB_79x49_flow_79x49x2x1_6dof_79x49x6_sc_79x49x2.h5', help="Path for loading validation data")
opt = parser.parse_args()

# make log dirs
exp_name = 'DTU'
exp_path = os.path.join(opt.summaryPath,exp_name)
os.makedirs(exp_path, exist_ok=True)

# log infomation
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
fh = logging.FileHandler(os.path.join(exp_path,'Training.log'))
log.addHandler(fh)
logging.info(opt)


if __name__ == '__main__':

    # specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # build model
    model=MainNet(opt)
    to_device(model, device)
    weights_init(model)
    model.train()

    # calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Parameters: %d; Training parameters: %d" %(total_params,total_trainable_params))

    # loss
    criterion = torch.nn.L1Loss() # Loss 

    # optimizer
    optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=opt.learningRate) #optimizer
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.epochNum*0.8, gamma=0.1, last_epoch=-1)
    
    # load training data
    setup_seed(1)
    training_dataset = TrainingDataset(opt)
    training_dataloader = DataLoader(training_dataset, batch_size=opt.batch_size, shuffle=True, num_workers = 0)
    validation_dataset = ValidationDataset(opt)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers = 0)

    # initialization
    lossLogger = defaultdict(list)
    validLogger = defaultdict(list)
    writer = SummaryWriter(exp_path)
    perceptual_loss = PerceptualLoss()
    to_device(perceptual_loss, device)
    iterCount = 0
    validCount = 0
    min_loss = 100
    

    for epoch in range(opt.epochNum):
        loss_sum = 0
        start = time.time()
        for _,training_samples in enumerate(training_dataloader):
            iterCount = iterCount +1
            
            # load training samples
            training_samples = to_device(training_samples,device)
            
            # extract batch
            training_batch = extract_batch(training_samples,opt)       

            # model inference
            ref_novel, blended_novel, interp_novel, weight, confidence  = model(training_batch, opt.band_width)
            
            # calculate loss
            gt_taget_view = training_batch[0]['view']
            pred_loss, refined_loss, blended_loss, interpolated_loss = l1_loss(ref_novel, blended_novel, interp_novel, gt_taget_view)
            percep_loss = perceptual_loss(ref_novel,gt_taget_view)
            loss = pred_loss + WeightSmoothLoss(weight) + percep_loss + ssim_loss(ref_novel,gt_taget_view,11)
            loss_sum += loss.item()
            writer.add_scalars('Loss', 
                             {'refined_loss':refined_loss,
                              'blended_loss':blended_loss,
                              'interpolated_loss':interpolated_loss,
                              'weightsmoothness_loss':WeightSmoothLoss(weight),
                              'percep_loss':percep_loss,
                              'ssim_loss': ssim_loss(ref_novel,gt_taget_view,11)},
                              iterCount)
            print("Epoch: %d Batch: %d Loss: %.6f" %(epoch,iterCount,loss.item()))
            
            # clean gridients and back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        end = time.time()   
        print(end-start)

        # make grid for visualization
        if epoch%opt.logNum==0:
            b,ns,c,h,w = training_batch[1]['view'].shape
            ps = ref_novel.shape[2]
            input_view_grid = torchvision.utils.make_grid(training_batch[1]['view'].permute(0,2,3,1,4).reshape(b,c,h,ns*w))
            warped_view_grid = torchvision.utils.make_grid(training_batch[1]['warped_image'].permute(0,3,2,4,1,5).reshape(b,c,(ns-1)*h,ns*w))
            interpolated_novel_grid = torchvision.utils.make_grid(interp_novel.permute(0,2,3,1,4).reshape(b,c,ps,ns*ps))
            blended_novel_grid = torchvision.utils.make_grid(blended_novel)
            ref_novel_grid = torchvision.utils.make_grid(ref_novel)
            confidence_grid = torchvision.utils.make_grid(confidence.permute(0,2,1,3).reshape(b,ps,ns*ps))

            writer.add_image('input_view', input_view_grid, epoch)
            writer.add_image('warped_view', warped_view_grid, epoch)
            writer.add_image('interpolated_novel', interpolated_novel_grid, epoch)
            writer.add_image('blended_novel', blended_novel_grid, epoch)
            writer.add_image('ref_novel', ref_novel_grid, epoch)
            writer.add_image('confidence', confidence_grid, epoch)
        
        # validation
        if epoch%opt.validNum==0:
            validCount = validCount+1
            with torch.no_grad():
                psnr_ref_novel = 0
                for _,valid_samples in enumerate(validation_dataloader):
                    
                    # load validation samples
                    valid_samples = to_device(valid_samples,device)
                    
                    # extract batch
                    valid_batch = extract_batch(valid_samples,opt)       

                    # model inference
                    ref_novel, blended_novel, interp_novel, weight, confidence  = model(valid_batch,opt.band_width)
                    gt_taget_view = valid_batch[0]['view']
                    psnr_ref_novel = psnr_ref_novel + calculate_psnr(ref_novel, gt_taget_view)

                # record the validation loss
                validLogger['Epoch'].append(validCount-1)
                validLogger['PSNR'].append(psnr_ref_novel.cpu().numpy()/len(validation_dataloader))
                plt.figure()
                plt.title('PSNR')
                plt.plot(validLogger['Epoch'],validLogger['PSNR'])
                plt.savefig(exp_path+'/Validation_{}.jpg'.format(opt.learningRate))
                plt.close()

        # save model 
        if epoch%opt.saveNum==0:
            torch.save(model.state_dict(),exp_path+'/ALFR_{}.pth'.format(epoch))
            if loss_sum/len(training_dataloader) <= min_loss:
                torch.save(model.state_dict(),exp_path+'/ALFR_optimal.pth')
                log.info("Epoch: %d Loss: %.6f is the optimal!" %(epoch,loss_sum/len(training_dataloader)))
                min_loss = loss_sum/len(training_dataloader)
        log.info("Epoch: %d Loss: %.6f" %(epoch,loss_sum/len(training_dataloader)))

        scheduler.step()

        # record the training loss
        lossLogger['Epoch'].append(epoch)
        lossLogger['Loss'].append(loss_sum/len(training_dataloader))
        # training loss
        plt.figure()
        plt.title('Loss')
        plt.plot(lossLogger['Epoch'],lossLogger['Loss'])
        plt.savefig(exp_path+'/Training_total.jpg')
        plt.close()
    

