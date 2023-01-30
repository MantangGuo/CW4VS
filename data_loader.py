import torch
import h5py
import os
import numpy as np
from torch.utils.data import Dataset
from functions import generate_samples

class TrainingDataset(Dataset):
    """
    load training dataset into batches
    """
    def __init__(self, opt):
        super(TrainingDataset, self).__init__()     
        self.data_set = h5py.File(opt.training_data_path, 'r') 
        self.scene_list = list(self.data_set.keys()) 
        self.num_source = opt.num_source
        
    def __getitem__(self, idx):

        scene_name = self.scene_list[idx]
        views = self.data_set[scene_name+'/views'] #[T, H, W, 3]
        views = np.array(views)
        views = views[:,:,:,::-1] # BGR to RGB
        flows = self.data_set[scene_name+'/flows'] #[T, num_source, num_source-1, 2, H, W]
        source_clusters = self.data_set[scene_name+'/source_clusters'] #[T, num_source]
        pose_maps = self.data_set[scene_name+'/pose_maps'] #[T, 6]
        Ks = self.data_set[scene_name+'/Ks'] #[T, 3, 3]
        Rs = self.data_set[scene_name+'/Rs'] #[T, 3, 3]
        Ts = self.data_set[scene_name+'/Ts'] #[T, 3]
        

        taget_sample, source_sample = generate_samples(views, flows, pose_maps, Ks, Rs, Ts, source_clusters)
        
        taget_sample['view'] = torch.from_numpy(taget_sample['view'].astype(np.float32)/255).permute(2,0,1) #[3,H,W]
        taget_sample['posemap'] = torch.from_numpy(taget_sample['posemap'].astype(np.float32)) #[6]
        taget_sample['K'] = torch.from_numpy(taget_sample['K'].astype(np.float32)) #[3,3]
        taget_sample['R'] = torch.from_numpy(taget_sample['R'].astype(np.float32)) #[3,3]
        taget_sample['T'] = torch.from_numpy(taget_sample['T'].astype(np.float32)) #[3,1]

        source_sample['view'] = torch.from_numpy(source_sample['view'].astype(np.float32)/255).permute(0,3,1,2) #[num_source, 3,H,W]
        source_sample['flow'] = torch.from_numpy(source_sample['flow'].astype(np.float32)) #[num_source,num_source-1,2,H,W]
        source_sample['posemap'] = torch.from_numpy(source_sample['posemap'].astype(np.float32)) #[num_source,6]
        source_sample['K'] = torch.from_numpy(source_sample['K'].astype(np.float32)) #[num_source,3,3]
        source_sample['R'] = torch.from_numpy(source_sample['R'].astype(np.float32)) #[num_source,3,3]
        source_sample['T'] = torch.from_numpy(source_sample['T'].astype(np.float32)) #[num_source,3,1]

        return [taget_sample, source_sample]
        
    def __len__(self):
        return len(self.scene_list)


class ValidationDataset(Dataset):
    """
    load validation dataset into batches
    """
    def __init__(self, opt):
        super(ValidationDataset, self).__init__()     
        self.data_set = h5py.File(opt.validation_data_path, 'r') 
        self.scene_list = list(self.data_set.keys())
        
        
    def __getitem__(self, idx):
        scene_name = self.scene_list[idx]
        views = self.data_set[scene_name+'/views'] #[T, H, W, 3]
        views = np.array(views)
        views = views[:,:,:,::-1] # BGR to RGB
        flows = self.data_set[scene_name+'/flows'] #[T, num_source, num_source-1, 2, H, W]
        source_clusters = self.data_set[scene_name+'/source_clusters'] #[T, num_source]
        pose_maps = self.data_set[scene_name+'/pose_maps'] #[T, H, W]
        Ks = self.data_set[scene_name+'/Ks'] #[T, 3, 3]
        Rs = self.data_set[scene_name+'/Rs'] #[T, 3, 3]
        Ts = self.data_set[scene_name+'/Ts'] #[T, 3]
        

        taget_sample, source_sample = generate_samples(views, flows, pose_maps, Ks, Rs, Ts, source_clusters)
        
        taget_sample['view'] = torch.from_numpy(taget_sample['view'].astype(np.float32)/255).permute(2,0,1) #[3,H,W]
        taget_sample['posemap'] = torch.from_numpy(taget_sample['posemap'].astype(np.float32)) #[6]
        taget_sample['K'] = torch.from_numpy(taget_sample['K'].astype(np.float32)) #[3,3]
        taget_sample['R'] = torch.from_numpy(taget_sample['R'].astype(np.float32)) #[3,3]
        taget_sample['T'] = torch.from_numpy(taget_sample['T'].astype(np.float32)) #[3,1]

        source_sample['view'] = torch.from_numpy(source_sample['view'].astype(np.float32)/255).permute(0,3,1,2) #[num_source, 3,H,W]
        source_sample['flow'] = torch.from_numpy(source_sample['flow'].astype(np.float32)) #[num_source,num_source-1,2,H,W]
        source_sample['posemap'] = torch.from_numpy(source_sample['posemap'].astype(np.float32)) #[num_source,6]
        source_sample['K'] = torch.from_numpy(source_sample['K'].astype(np.float32)) #[num_source,3,3]
        source_sample['R'] = torch.from_numpy(source_sample['R'].astype(np.float32)) #[num_source,3,3]
        source_sample['T'] = torch.from_numpy(source_sample['T'].astype(np.float32)) #[num_source,3,1]

                  
        return [taget_sample, source_sample]
        
    def __len__(self):
        return len(self.scene_list)


class TestDataset(Dataset):
    """
    load test dataset into batches
    """
    def __init__(self, opt):
        super(TestDataset, self).__init__()     
        self.data_set = h5py.File(opt.test_data_path, 'r') 
        self.scene_list = list(self.data_set.keys()) 
        
        
    def __getitem__(self, idx):
        scene_name = self.scene_list[idx]
        views = self.data_set[scene_name+'/views'][:] #[T, H, W, 3]
        views = np.array(views)
        views = views[:,:,:,::-1] # BGR to RGB
        flows = self.data_set[scene_name+'/flows'][:] #[T, num_source, num_source-1, 2, H, W]
        source_clusters = self.data_set[scene_name+'/source_clusters'][:]  #[T, num_source]
        pose_maps = self.data_set[scene_name+'/pose_maps'][:]  #[T, H, W]
        Ks = self.data_set[scene_name+'/Ks'][:]  #[T, 3, 3]
        Rs = self.data_set[scene_name+'/Rs'][:]  #[T, 3, 3]
        Ts = self.data_set[scene_name+'/Ts'][:]  #[T, 3]
        
        
        views = torch.from_numpy(views.astype(np.float32)/255)
        flows = torch.from_numpy(flows.astype(np.float32))
        pose_maps = torch.from_numpy(pose_maps.astype(np.float32))
        Ks= torch.from_numpy(Ks.astype(np.float32))
        Rs= torch.from_numpy(Rs.astype(np.float32))
        Ts= torch.from_numpy(Ts.astype(np.float32))
        
        sample = {'scene_name':scene_name, 
                  'views':views, 
                  'flows':flows, 
                  'source_clusters':source_clusters,
                  'Ks':Ks, 
                  'Rs':Rs, 
                  'Ts':Ts, 
                  'pose_maps':pose_maps}
                  
        return sample
        
    def __len__(self):
        return len(self.scene_list)
