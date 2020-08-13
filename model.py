import argparse
import time
import torch
import numpy as np
import torch.optim as optim
from networks import Resnet50_md, Discriminator
from loss import HcGANLoss, MonodepthLoss
from torch.utils.data import DataLoader, ConcatDataset
from dataset import KITTI, prepare_dataloader
from os import path, makedirs
from utils import warp_left, warp_right, to_device, scale_pyramid, adjust_learning_rate
import numpy as np
from itertools import chain

class Model:

    def __init__(self, batch_size=4, input_channels=3, use_multiple_gpu=False,
                       learning_rate=1e-4,
                       model_path='model', device='cuda:0', mode='train', train_dataset_dir='data_scene_flow/training', 
                       val_dataset_dir='data_scene_flow/testing', num_workers=4, do_augmentation=True,
                       output_directory='outputs',
                       input_height=256, input_width=512, augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2]):

        self.batch_size = batch_size
        self.input_channels = input_channels
        self.model_path = model_path
        self.device = device
        self.use_multiple_gpu = use_multiple_gpu

        self.g_LL = Resnet50_md(self.input_channels).to(self.device)
        self.d_R = Discriminator(self.input_channels).to(self.device)

        if self.use_multiple_gpu:
            self.g_LL = torch.nn.DataParallel(self.g_LL)
            self.d_R = torch.nn.DataParallel(self.d_R)

        self.learning_rate=learning_rate
        self.mode = mode
        self.input_height = input_height
        self.input_width = input_width
        self.augment_parameters = augment_parameters
        self.train_dataset_dir = train_dataset_dir
        self.val_dataset_dir = val_dataset_dir
        self.g_best_val_loss = float('inf')
        self.num_workers = num_workers
        self.do_augmentation = do_augmentation

        if self.mode == 'train':
            self.criterion_GAN = HcGANLoss().to(self.device)
            self.criterion_mono = MonodepthLoss()
            
            self.optimizer = optim.Adam(
                chain(
                    self.g_LL.parameters()
                ),
                lr=self.learning_rate
            )
            self.val_n_img, self.val_loader = prepare_dataloader(self.val_dataset_dir, self.mode, self.augment_parameters,
                                                False, self.batch_size,
                                                (self.input_height, self.input_width),
                                                self.num_workers, shuffle=False)
        else:
            self.augment_parameters = None
            self.do_augmentation = False

        self.n_img, self.loader = prepare_dataloader(self.train_dataset_dir, self.mode,
                                                    self.augment_parameters,
                                                    self.do_augmentation, self.batch_size,
                                                    (self.input_height, self.input_width),
                                                    self.num_workers)
        self.output_directory = output_directory
        if 'cuda' in self.device:
            torch.cuda.synchronize()


    def compute_d_loss(self):
        d_RR = self.d_R(self.images_R[0])
        d_RR_est = self.d_R(self.images_R_est[0])
        d_loss_R_fake = self.criterion_GAN(d_RR_est, label=False)
        d_loss_R_real = self.criterion_GAN(d_RR, label=True)
        d_loss_R = -0.5*(d_loss_R_real + d_loss_R_fake)
        d_loss = 0.001*d_loss_R
        return d_loss

    
    def compute_g_loss(self):
        g_loss_mono = self.criterion_mono(self.disps_RL, self.images_L, self.images_R)
        d_RR_est = self.d_R(self.images_R_est[0])
        d_loss_R_fake = self.criterion_GAN(d_RR_est, label=False)
        g_loss = g_loss_mono + 0.001*d_loss_R_fake
        return g_loss   


    def compute(self, left, right):
        self.images_L = scale_pyramid(left)
        self.images_R = scale_pyramid(right)
        self.disps_RL = self.g_LL(left)

        self.disps_RLR_est = [d[:, 1, :, :].unsqueeze(1) for d in self.disps_RL]
        self.disps_RLL_est = [d[:, 0, :, :].unsqueeze(1) for d in self.disps_RL]

        self.images_R_est = [warp_right(self.images_L[i], self.disps_RLR_est[i]) for i in range(4)]
        self.images_L_est = [warp_right(self.images_L[i], self.disps_RLR_est[i]) for i in range(4)]
        

    def train(self, epoch):

        adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
        
        c_time = time.time()
        g_running_loss = 0.0
        d_running_loss = 0.0
        
        self.d_R.train()

        for data in self.loader:
            data = to_device(data, self.device)
            left, right= data['left_image'], data['right_image']

            self.optimizer.zero_grad()
            self.compute(left, right)
            d_loss = self.compute_d_loss()
            d_loss.backward()
            d_running_loss += d_loss.item()
            self.optimizer.step()

        self.g_LL.train()

        for data in self.loader:
            data = to_device(data, self.device)
            left, right= data['left_image'], data['right_image']

            self.optimizer.zero_grad()
            self.compute(left, right)
            g_loss = self.compute_g_loss()
            g_loss.backward()
            g_running_loss += g_loss.item()
            self.optimizer.step()

        g_running_loss /= self.n_img / self.batch_size
        d_running_loss /= self.n_img / self.batch_size
        
        self.save()
        print(
            'Epoch: {}'.format(str(epoch).rjust(3, ' ')),
            'G: {:.3f}'.format(g_running_loss),
            'D: {:.3f}'.format(d_running_loss),
            'Time: {:.2f}s'.format(time.time() - c_time)
        )


    def path_for(self, fn):
        return path.join(self.model_path, fn)


    def save(self, best=False):
        if not path.exists(self.model_path):
            makedirs(self.model_path, exist_ok=True)
        if best:
            torch.save(self.g_LL.state_dict(), self.path_for('gll.nn.best'))
            torch.save(self.d_R.state_dict(), self.path_for('dr.nn.best'))
        else:
            torch.save(self.g_LL.state_dict(), self.path_for('gll.nn'))
            torch.save(self.d_R.state_dict(), self.path_for('dr.nn'))


    def load(self, best=False):
        print('load', 'best', best)
        if best:
            self.g_LL.load_state_dict(torch.load(self.path_for('gll.nn.best')))
            self.d_R.load_state_dict(torch.load(self.path_for('dr.nn.best')))
        else:

            self.g_LL.load_state_dict(torch.load(self.path_for('gll.nn')))
            self.d_R.load_state_dict(torch.load(self.path_for('dr.nn')))
    

    def test(self):

        c_time = time.time()

        self.g_LL.eval()

        g_running_val_loss = 0.0
        d_running_val_loss = 0.0

        disparities_R = np.zeros((self.n_img, self.input_height, self.input_width), dtype=np.float32)
        images_L = np.zeros((self.n_img, 3, self.input_height, self.input_width), dtype=np.float32)
        images_R = np.zeros((self.n_img, 3, self.input_height, self.input_width), dtype=np.float32)
        images_est_R = np.zeros((self.n_img, 3, self.input_height, self.input_width), dtype=np.float32)

        ref_img = None

        RMSE = 0.0
        AbsRel = 0.0
        SqrRel = 0.0

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                
                data = to_device(data, self.device)
                left, right= data['left_image'], data['right_image']

                self.compute(left, right)
                g_loss = self.compute_g_loss()
                g_running_val_loss += g_loss.item()

                DR = self.disps_RLR_est[0].cpu().numpy()[:, 0, :, :]
                ndata, _, _ = DR.shape
                disparities_R[i*self.batch_size:i*self.batch_size+ndata] = DR
                images_est_R[i*self.batch_size:i*self.batch_size+ndata] = warp_right(left, self.disps_RLR_est[0]).cpu().numpy()
                images_L[i*self.batch_size:i*self.batch_size+ndata] = left.cpu().numpy()
                images_R[i*self.batch_size:i*self.batch_size+ndata] = right.cpu().numpy()

                RMSE += torch.sqrt(torch.mean((left - warp_right(left, self.disps_RLR_est[0]))**2, dim=[2,3])).sum(dim=[0,1])
                AbsRel += torch.mean(torch.abs(left - warp_right(left, self.disps_RLR_est[0])) / left, dim=[2,3]).sum(dim=[0,1])
                SqrRel += torch.mean((left - warp_right(left, self.disps_RLR_est[0])**2) / left, dim=[2,3]).sum(dim=[0,1])

            RMSE /= self.val_n_img / self.batch_size
            AbsRel /= self.val_n_img / self.batch_size
            SqrRel /= self.val_n_img / self.batch_size
            print('RMSE', RMSE.item())
            print('AbsRel', AbsRel.item())
            print('SqrRel', SqrRel.item())


            g_running_val_loss /= self.val_n_img / self.batch_size
            d_running_val_loss /= self.val_n_img / self.batch_size

            model_saved = '[*]'
            if g_running_val_loss < self.g_best_val_loss:
                self.save(True)
                self.g_best_val_loss = g_running_val_loss
                model_saved = '[S]'
            print(
                '      Test',
                'G: {:.3f}({:.3f})'.format(g_running_val_loss, self.g_best_val_loss),
                'D: {:.3f}'.format(d_running_val_loss),
                'Time: {:.2f}s'.format(time.time() - c_time),
                model_saved
            )
        
        return disparities_R, images_L, images_R, images_est_R
