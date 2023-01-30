
import torch
import kornia
import random
import math
import numpy as np
from torch.autograd import Variable
from skimage.measure import compare_ssim

def ComputeQuant(pred_views, gt_views):
    '''
    pred_views: [t,h,w,3]
    gt_views: [t,h,w,3]
    '''
    scene_psnr = []
    scene_ssim =[]
    for i in range(pred_views.shape[0]):
        view_psnr =  comp_psnr(pred_views[i],gt_views[i])
        view_ssim = compare_ssim((pred_views[i]*255.0).astype(np.uint8),
                                 (gt_views[i]*255.0).astype(np.uint8),
                                 gaussian_weights=True,
                                 sigma=1.5,
                                 use_sample_covariance=False,
                                 multichannel=True)
        scene_psnr.append(view_psnr)
        scene_ssim.append(view_ssim)
    return np.mean(scene_psnr), np.mean(scene_ssim)


def crop_image(image, patch_size): 
    '''crop the input image into patches'''
    _,_,h,w=image.shape
    image_stack=[]
    left_top_xy = []
    num_h = 0
    for i in range(0, h-patch_size,patch_size):
        num_h = num_h + 1
        num_w = 0
        for j in range(0, w-patch_size, patch_size):
            num_w = num_w + 1
            image_patch = image[:,:,i:i+patch_size, j:j+patch_size]
            image_stack.append(image_patch)
            left_top_xy.append([j,i])
    image_stack = torch.stack(image_stack) #[n,b,c,patch_size,patch_size] 
    return  image_stack.permute(1,2,3,4,0),left_top_xy,[num_h,num_w] #[b,c,patch_size,patch_size,n] 


def merge_image(image_stack,coordinate):
    '''merge the patches into an intact image'''
    b,c,patch_size,_,_ = image_stack.shape
    image_stack = image_stack.reshape(b,c,patch_size,patch_size,coordinate[0],coordinate[1]) 
    image_stack = image_stack.permute(0,1,4,2,5,3)
    image_merged  = image_stack.reshape(b,c,coordinate[0]*patch_size,coordinate[1]*patch_size)
    return image_merged # [b,c,h_croped,w_croped]
    

def comp_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


#Move tensor(s) to chosen device
def to_device(data, device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    if isinstance(data,(dict)):
        for key, value in data.items():
            data[key] = to_device(value,device)
        return data
    return data.to(device,non_blocking=True)


#Initiate parameters in model 
def weights_init(m):
    '''initialization for layers'''
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    if classname.find('Conv3d') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('ConvTranspose3d') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def setup_seed(seed):
    '''generate random seed'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_samples(views, flows, pose_maps, Ks, Rs, Ts, source_clusters, index = None, mode = 'train'):
    """
    generate input and label samples from the training dataset
    Args:
        views: views dataset / [n, H, W, C]
        flow: optical flow / [n, num_source, numsource-1, 2, H, W]
        pose_maps: pose maps / [n, 6]
        Ks, Rs, Ts: intrinsic and extrinsic parameters / [n, 3, 3], [n, 3, 3], [n, 3] 
        source_clusters: the source indexs for each view / [n, num_source]
    """
    
    if mode == 'train':
        n = views.shape[0] 
        ind_target = random.randrange(n) # index of target view

        # sample target view 
        target_view = views[ind_target] # [H,W,C]
        target_posemap = pose_maps[ind_target] # [H,W]
        target_K = Ks[ind_target] # [3,3]
        target_R = Rs[ind_target] # [3,3]
        target_T = Ts[ind_target][:, np.newaxis] # [3,1]

        # sample source views
        source_cluster = list(source_clusters[ind_target]) # [num_source]
        source_cluster.sort()
        source_views = views[source_cluster] # [num_source, H, W, C]
        source_flows = flows[ind_target] # [num_source, numsource-1, 2, H, W]
        source_posemaps = pose_maps[source_cluster] #[num_source, H, W]
        source_Ks = Ks[source_cluster] # [num_source,3,3]
        source_Rs = Rs[source_cluster] # [num_source,3,3]
        source_Ts = Ts[source_cluster][:, :, np.newaxis] # [num_source,3,1]

    if mode == 'test':
        n = views.shape[1]
        ind_target = index

        # sample target view 
        target_view = views[:, ind_target].permute(0,3,1,2) # [b,C,H,W]
        target_posemap = pose_maps[:, ind_target] #[b,6]
        target_K = Ks[:, ind_target] # [b,3,3]
        target_R = Rs[:, ind_target] # [b,3,3]
        target_T = Ts[:, ind_target].unsqueeze(2) # [b,3,1]

        # sample source views
        source_cluster = list(source_clusters[:,ind_target].squeeze(0)) # [num_source]
        source_cluster.sort()
        source_views = views[:,source_cluster].permute(0,1,4,2,3) # [b,num_source, C, H, W]
        source_flows = flows[:,ind_target] # [b,num_source, numsource-1, 2, H, W]
        source_posemaps = pose_maps[:,source_cluster] #[b,num_source, H, W]
        source_Ks = Ks[:,source_cluster] # [b,num_source,3,3]
        source_Rs = Rs[:,source_cluster] # [b,num_source,3,3]
        source_Ts = Ts[:,source_cluster].unsqueeze(3) # [b,num_source,3,1]

    target_sample = {'view':target_view,
                'posemap':target_posemap,
                'K': target_K,
                'R': target_R, 
                'T': target_T
    }
    source_sample = {'view':source_views,
                'flow':source_flows,
                'posemap':source_posemaps,
                'K': source_Ks,
                'R': source_Rs, 
                'T': source_Ts
    }
    
    return target_sample, source_sample
    

def extract_batch(samples, opt, left_top_xy = None, mode = 'train'):
    """
    Extract batches as input to the network
    """
    # initialization
    target_view = samples[0]['view'] #[b,c,h,w]
    target_posemap = samples[0]['posemap'] #[b,6]
    target_K = samples[0]['K'] #[b,3,3]
    target_R = samples[0]['R'] #[b,3,3]
    target_T = samples[0]['T'] #[b,3,1]

    source_views = samples[1]['view'] #[b,num_source,c,h,w]
    source_flows = samples[1]['flow'] #[b,num_source,num_source-1,2,h,w]
    source_posemaps = samples[1]['posemap'] #[b,num_source,6]
    source_Ks = samples[1]['K'] #[b,num_source,3,3]
    source_Rs = samples[1]['R'] #[b,num_source,3,3]
    source_Ts = samples[1]['T'] #[b,num_source,3,3]

    b,c,H,W = target_view.shape
    num_source = source_views.shape[1]
    patch_size = opt.patch_size
    num_points = opt.patch_size*opt.patch_size 
    band_width = opt.band_width
    depth_range = opt.depth_range
    D = opt.D 
    epipolar_length = opt.epipolar_length

    if mode == 'train':
        ind_h = random.randrange(0, H-patch_size, 8) # index in spatial dimension
        ind_w = random.randrange(0, W-patch_size, 8)
    if mode == 'test':
        ind_h = left_top_xy[1]
        ind_w = left_top_xy[0]

    # generate target patch
    target_patch = target_view[:,:,ind_h:ind_h+patch_size, ind_w:ind_w+patch_size] #[b,c,patch_size,patch_size]

    # generate warped images
    warped_source_views = generate_warped_images(source_views,source_flows) #[b,num_source,num_source-1,3,h,w]
    

    # generate psvs
    psvs = generate_psv(source_views, source_Ks, source_Rs, source_Ts, target_K, target_R, target_T, D, depth_range) #[b,num_source,c,D,h,w]
    psvs = psvs[:,:,:,:,ind_h:ind_h+patch_size, ind_w:ind_w+patch_size] #[b,num_source,c,D,patch_size,patch_size]

    # generate coordinates of target pixels
    XX = Variable(torch.arange(ind_w,ind_w+patch_size).view(1,1,patch_size,1).expand(b,patch_size,-1,-1)).type_as(target_patch) #[b,patch_size, patch_size,1]
    YY = Variable(torch.arange(ind_h,ind_h+patch_size).view(1,patch_size,1,1).expand(b,-1,patch_size,-1)).type_as(target_patch) #[b,patch_size, patch_size,1]
    target_xy = torch.cat((XX,YY), axis=3) #[b,patch_size, patch_size,2]
    target_xy = target_xy.reshape(b,num_points,2) #[b,num_points,2]

    # calculate source neighbors
    neighbor_xy = generate_neighbors(target_xy, epipolar_length, target_K, target_R, target_T, source_Ks, source_Rs, source_Ts) # [b, num_source, num_points, epipolar_length, 2]

    # spatial code
    spa_code = generate_spatial_code(neighbor_xy, target_xy, H, W, band_width) #[b,num_source,2,(2*band_width+1)*epipolar_length,num_points]

    # angular code
    # ang_code = generate_angular_code(source_posemaps,target_posemap,neighbor_xy,ind_h,ind_w,patch_size,band_width) #[b,num_source,1,(2*band_width+1)*epipolar_length,num_points]
    ang_code = generate_angular_code(source_posemaps,target_posemap,band_width,epipolar_length,num_points) #[b,num_source,6,(2*band_width+1)*epipolar_length,num_points]


    target_batch = {'view':target_patch,
                    'posemap':target_posemap
                    }
                    
    source_batch = {'view':source_views,
                    'flow': source_flows,
                    'warped_image': warped_source_views,
                    'posemap':source_posemaps,
                    'psv': psvs,
                    'neighbor_xy':neighbor_xy,
                    'spa_code': spa_code, 
                    'ang_code': ang_code
    }

    return target_batch, source_batch


def generate_psv(source_views,source_Ks,source_Rs,source_Ts,target_K,target_R,target_T, D,depth_range):
    '''
    source_views:[b,num_source,c,h,w]
    source_Ks:[b,num_source,3,3]
    source_Rs:[b,num_source,3,3]
    source_Ts:[b,num_source,3,1]
    target_K:[b,num_source,3,3]
    target_R:[b,num_source,3,3]
    target_T:[b,num_source,3,1]
    D:[1]
    depth_range:[2]
    '''
    b,num_source,c,h,w = source_views.shape
    unit = torch.tensor([0.0,0.0,0.0,1.0],device=torch.device("cuda:0"))
    depth_values = torch.arange(depth_range[0], 
                                depth_range[1], 
                                (depth_range[1]-depth_range[0])/D, 
                                dtype=torch.float32,
                                device=torch.device("cuda:0")).view(1,D).expand(b,D) #[b,D]
    source_proj = torch.cat((source_Ks@torch.cat((source_Rs,
                                                  source_Ts),
                                                  dim=3),
                             unit.view(1,1,1,4).expand(b,num_source,-1,-1)),
                             dim=2) #[b,num_source,4,4]
    target_proj = torch.cat((target_K@torch.cat((target_R,
                                                 target_T),
                                                 dim=2),
                            unit.view(1,1,4).expand(b,-1,-1)),
                            dim=1) #[b,4,4]
    psvs = homo_warping(source_views.reshape(b*num_source,c,h,w), 
                        source_proj.reshape(b*num_source,4,4), 
                        target_proj.unsqueeze(1).expand(-1,num_source,-1,-1).reshape(b*num_source,4,4), 
                        depth_values.unsqueeze(1).expand(-1,num_source,-1).reshape(b*num_source,D)) #[b*num_source,c,D,h,w]
    
    return psvs.reshape(b,num_source,c,D,h,w)
    

def generate_warped_images(source_images, source_flows):
    '''
    source_images: [b,num_source,c,h,w]
    source_flows: [b,num_source,num_source-1,2,h,w]
    '''
    b,num_source,c,h,w = source_images.shape
    warped_source_views = []
    for ind_source1 in range(num_source):
        res_source_indexes = list(range(num_source))
        res_source_indexes.remove(ind_source1)
        for count_source2, ind_source2  in enumerate(res_source_indexes):
            flow = source_flows[:,ind_source1,count_source2] #[b,2,H,W]
            view = source_images[:,ind_source2] #[b,c,H,W]
            warped_image = warping(flow, view) #[b,c,H,W]
            warped_source_views.append(warped_image)
    warped_source_views = torch.stack(warped_source_views,dim=1).reshape(b,num_source,num_source-1,c,h,w) 
    return  warped_source_views #[b,num_source,num_source-1,c,h,w]


def generate_spatial_code(neighbor_xy, target_xy, H, W, band_width):
    '''generate spatial code
    Args:
        neighbor_xy: (x,y) coordinates of epipolar lines \ [b, num_source, num_points, epipolar_length, 2]
        target_xy: reference coordinates \ #[b,num_points,2]
        H, W: size of the source image
        band_width: the width of the band along with the epipolar line
    '''
    b,num_source,num_points,epipolar_length= neighbor_xy.shape[:4]
    XX = Variable(torch.arange(0,W).reshape(1,1,1,W).expand(b,-1,H,-1)).type_as(target_xy) #[b,1,H,W]
    YY = Variable(torch.arange(0,H).reshape(1,1,H,1).expand(b,-1,-1,W)).type_as(target_xy) #[b,1,H,W]
    XY = torch.cat((XX,YY), dim=1) #[b,2,H, W]
    XY = XY.reshape(b,1,2,H,W).expand(-1,num_source,-1,-1,-1) #[b,num_source,2,H,W]
    XY = XY.reshape(b*num_source,2,H,W)
    neighbor_xy = neighbor_xy.reshape(b*num_source,num_points, epipolar_length, 2) #[b*num_source, num_points, epipolar_length, 2]
    source_xy = epipolar_unfold(XY,neighbor_xy,band_width) #[b*num_source,2,(2*band_width+1)*epipolar_length,num_points]
    source_xy = source_xy.reshape(b,num_source,2,(2*band_width+1)*epipolar_length,num_points)
    target_xy = target_xy.permute(0,2,1) #[b,2,num_points]
    target_xy = target_xy.reshape(b,1,2,1,num_points)
    target_xy = target_xy.expand(-1,num_source,-1,(2*band_width+1)*epipolar_length,-1) #[b,num_source,2,(2*band_width+1)*epipolar_length,num_points]
    return source_xy - target_xy #[b,num_source,2,(2*band_width+1)*epipolar_length,num_points]


def generate_angular_code(source_posemaps,target_posemap,band_width,epipolar_length,num_points):
    '''
    source_posemaps: [b,num_source,6]
    target_posemap: [b,6]
    '''
    b,num_source = source_posemaps.shape[:2]
    source_posemaps = source_posemaps.reshape(b,num_source,6,1,1).expand(-1,-1,-1,(2*band_width+1)*epipolar_length,num_points) #[b,num_source,6,(2*band_width+1)*epipolar_length,num_points]
    target_posemap = target_posemap.reshape(b,1,6,1,1).expand(-1,num_source,-1,(2*band_width+1)*epipolar_length,num_points) #[b,num_source,6,(2*band_width+1)*epipolar_length,num_points]
    return source_posemaps - target_posemap


def warping(flow, img_source):
    '''warping the source image to the target one
    Args:
        flow: optical flow of target image [b,2,h,w]
        img_source: source image [b,c,h,w]
    
    Output:
        img_target: warped target image [b,h,w]
    '''
    b,c,h,w = img_source.shape
    # generate grid
    XX = Variable(torch.arange(0,w).view(1,1,w).expand(b,h,w)).type_as(img_source) #[b,h,w]
    YY = Variable(torch.arange(0,h).view(1,h,1).expand(b,h,w)).type_as(img_source)
    grid_w = XX + flow[:,0,:,:]
    grid_h = YY + flow[:,1,:,:]
    grid_w_norm = 2.0 * grid_w / (w-1) -1.0
    grid_h_norm = 2.0 * grid_h / (h-1) -1.0        
    grid = torch.stack((grid_w_norm, grid_h_norm),dim=3) #[b,h,w,2]
    # inverse warp
    img_target = torch.nn.functional.grid_sample(img_source,grid) # [b,c,h,w]
    return img_target


def generate_neighbors(target_xy, epipolar_length, target_K, target_R, target_T, source_Ks, source_Rs, source_Ts):
    '''
    target_xy: [b, num_points, 2]
    epipolar_length: [1]
    target_K, target_R, target_T: intrinsic and extrinsic matrix of target image / [b,3,3], [b,3,3], [b,3,1]
    source_Ks, source_Rs, source_Ts: intrinsic and extrinsic matrix of target image / [b,num_source,3,3], [b,num_source,3,3], [b,num_source,3,1]
    '''
    b,num_points = target_xy.shape[:2]
    num_source = source_Ks.shape[1]
    target_xy = target_xy.unsqueeze(1).expand(-1,num_source,-1,-1) # [b,num_source,num_points, 2]
    target_K = target_K.unsqueeze(1).expand(-1,num_source,-1,-1) # [b,num_source, 3,3]
    target_R = target_R.unsqueeze(1).expand(-1,num_source,-1,-1) # [b,num_source, 3,3]
    target_T = target_T.unsqueeze(1).expand(-1,num_source,-1,-1) # [b,num_source, 3,1]

    target_xy = target_xy.reshape(b*num_source,num_points, 2) # [b*num_source,num_points, 2]
    target_K = target_K.reshape(b*num_source, 3,3) # [b*num_source, 3,3]
    target_R = target_R.reshape(b*num_source, 3,3) # [b*num_source, 3,3]
    target_T = target_T.reshape(b*num_source, 3,1) # [b*num_source, 3,1]

    source_Ks = source_Ks.reshape(b*num_source, 3,3) # [b*num_source, 3,3]
    source_Rs = source_Rs.reshape(b*num_source, 3,3) # [b*num_source, 3,3]
    source_Ts = source_Ts.reshape(b*num_source, 3,1) # [b*num_source, 3,1]

    neighbor_xy = calculate_epipolar_lines(target_xy, epipolar_length, target_K, target_R, target_T, source_Ks, source_Rs, source_Ts) #[b*num_source, num_points, epipolar_length, 2]

    return neighbor_xy.reshape(b,num_source, num_points, epipolar_length, 2)


def calculate_epipolar_lines(target_xy, epipolar_length, target_K, target_R, target_T, source_K, source_R, source_T):
    '''calculate epipolar lines in image_1 corresponding to the given points in  image_2.
    Args:
        target_xy: given points in the target view / [b,num_points,2]
        epipolar_length: the length of the epipolar line
        source_K, source_R, source_T: intrinsic and extrinsic matrix of the source view / [b,3,3], [b,3,3], [b,3,1]
        target_K, target_R, target_T: intrinsic and extrinsic matrix of the target view / [b,3,3], [b,3,3], [b,3,1]
    '''
    b, num_points = target_xy.shape[:2]
    # calculate essential matrix from extrinsic parameters
    essential = kornia.geometry.epipolar.essential_from_Rt(target_R, target_T, source_R, source_T) #[b,3,3]
    # calculate fundemental matrix from essential matrix and intrinsic parameters
    fundamental = kornia.geometry.epipolar.fundamental_from_essential(essential, target_K, source_K) #[b,3,3]
    # calculate epipolar lines: ax+by+c = 0
    epipolar_lines = kornia.geometry.epipolar.compute_correspond_epilines(target_xy, fundamental) #[b,num_points,3]
    # calculate coordinates of pixels on epipolar lines
    epipolar_lines = epipolar_lines.reshape(b*num_points,1,3).expand(-1,epipolar_length,-1) #[b*num_points,epipolar_length,3]
    epipolar_x = torch.arange(-epipolar_length/2,epipolar_length/2).type_as(target_xy).reshape(1,epipolar_length).expand(b*num_points,-1) + target_xy.reshape(-1,2)[:,0:1].expand(-1,epipolar_length) #[b*num_points,epipolar_length]
    epipolar_y = -(epipolar_lines[:,:,0] * epipolar_x + epipolar_lines[:,:,2]) / (epipolar_lines[:,:,1] + 1e-6) #[b*num_points,epipolar_length]
    epipolar_xy = torch.stack([epipolar_x.reshape(b, num_points, epipolar_length, 1),epipolar_y.reshape(b, num_points, epipolar_length, 1)],dim=3) #[b, num_points, epipolar_length, 2]
    return  epipolar_xy


def calculate_psnr(img1, img2):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 1]"""

    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def epipolar_unfold(img, epipolar_xy, band_width):
    '''unfold an image/feature map along with epipolar line given the y coordinates of the epipolar-line points.
    Args:
        img: the image/feature map to be unfolded / [b,c,h,w]
        epipolar_xy: the xy coordinates of points on epipolar lines in 'num_source' source views/ [b, num_points, epipolar_length, 2]
        band_width: the width of the band along with the epipolar line
    Output:
        unfolded_img: the unfolded image/feature map
    '''
    h,w = img.shape[2:]

    epipolar_xy = torch.round(epipolar_xy)

    epipolar_x_norm = 2.0*epipolar_xy[:,:,:,0:1]/(w-1)-1.0 #[b, num_points, epipolar_length, 1]
    epipolar_y_norm = 2.0*epipolar_xy[:,:,:,1:2]/(h-1)-1.0 #[b, num_points, epipolar_length, 1]
    grid = torch.cat((epipolar_x_norm,epipolar_y_norm),dim=-1) #[b, num_points, epipolar_length, 2]

    for i in range(band_width):
        # up
        epipolar_yi_up = epipolar_xy[:,:,:,1:2] - (i+1)
        epipolar_yi_up_norm = 2.0*epipolar_yi_up/(h-1)-1.0
        grid_up = torch.cat((epipolar_x_norm,epipolar_yi_up_norm),dim=-1) #[b, num_points, epipolar_length, 2]
        # down
        epipolar_yi_down = epipolar_xy[:,:,:,1:2] + (i+1)
        epipolar_yi_down_norm = 2.0*epipolar_yi_down/(h-1)-1.0
        grid_down = torch.cat((epipolar_x_norm,epipolar_yi_down_norm),dim=-1) #[b, num_points, epipolar_length, 2]
        # append
        grid = torch.cat((grid,grid_up,grid_down),dim=2)
    
    unfolded_img = torch.nn.functional.grid_sample(img,grid,padding_mode='reflection') #[b,c,num_points,(2*band_width+1)*epipolar_length]
    unfolded_img = unfolded_img.permute(0,1,3,2) #[b,c,(2*band_width+1)*epipolar_length,num_points]
    return unfolded_img


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] (based on reference view)
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]
    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    warped_src_fea = torch.nn.functional.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    return warped_src_fea


def l1_loss(refined_pred,blended_pred,interpolated_pred, gt):
    criterion = torch.nn.L1Loss()
    refined_loss = criterion(refined_pred,gt) 
    blended_loss = criterion(blended_pred,gt) 
    interpolated_loss = 0
    for i in range(interpolated_pred.shape[1]):
        interpolated_loss = interpolated_loss + criterion(interpolated_pred[:,i],gt)
    pred_loss = refined_loss + blended_loss + interpolated_loss
    return pred_loss,refined_loss,blended_loss,interpolated_loss


def Gradient(pred):
    D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy


def WeightSmoothLoss(weight):
    '''
    weight: [b,2,r_patch_size,r_patch_size,range_disp]
    '''
    b,_,r_patch_size,r_patch_size,range_disp = weight.shape
    weight = weight.permute(0,1,4,2,3).reshape(b,-1,r_patch_size,r_patch_size) # [b,2*range_disp,r_patch_size,r_patch_size]
    dx, dy = Gradient(weight)
    return dx.abs().mean() + dy.abs().mean()