import argparse
import os
from math import log10

from PIL import Image

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_950.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

results = {'Set5': {'psnr': [], 'ssim': []}, 'VOC': {'psnr': [], 'ssim': []},'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

model = Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

test_set = TestDatasetFromFolder('/mnt/spinner/mlsp_project/data/test/', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = '/mnt/spinner/mlsp_project/outputs/SRF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

for image_name, lr_image, hr_restore_img, hr_image in test_bar:
    image_name = image_name[0]
    with torch.no_grad():
        lr_image = Variable(lr_image)
        hr_image = Variable(hr_image)
    
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        ###################################
        # BILINEAR, BICUBIC INTERPOLATION #
        ###################################
        lr_im = Image.fromarray(np.reshape(np.transpose(np.uint8(255*lr_image.cpu().numpy()), (0,2,3,1)), [lr_image.shape[2],lr_image.shape[3],3]))
        sr_bilinear = np.transpose(np.array(lr_im.resize((lr_image.shape[2]*UPSCALE_FACTOR, lr_image.shape[3]*UPSCALE_FACTOR), Image.BILINEAR)), (2,0,1) )[np.newaxis,:]
        sr_bicubic = np.transpose(np.array(lr_im.resize((lr_image.shape[2]*UPSCALE_FACTOR, lr_image.shape[3]*UPSCALE_FACTOR), Image.BICUBIC)), (2,0,1) )[np.newaxis,:]

        sr_image = model(lr_image)
        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
    #     ssim = pytorch_ssim.ssim(sr_image, hr_image).data[0]
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
             display_transform()(sr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                         image_name.split('.')[-1], padding=5)

        # Modification to save images separately
        np.save(out_path + image_name.split('.')[0] + 'hr_restore_BILINEAR_psnr_%.4f_ssim_%.4f' % (psnr, ssim), sr_bilinear)
        np.save(out_path + image_name.split('.')[0] + 'hr_restore_BICUBIC_psnr_%.4f_ssim_%.4f' % (psnr, ssim), sr_bicubic)

        np.save(out_path+image_name.split('.')[0] + 'hr_restore_psnr_%.4f_ssim_%.4f'%(psnr, ssim),hr_restore_img.cpu().numpy())
        np.save(out_path + image_name.split('.')[0] + 'hr_psnr_%.4f_ssim_%.4f' % (psnr, ssim),hr_image.cpu().numpy())
        np.save(out_path + image_name.split('.')[0] + 'lr_psnr_%.4f_ssim_%.4f' % (psnr, ssim),sr_image.cpu().numpy())

        # save psnr\ssim
        results[image_name.split('_')[0]]['psnr'].append(psnr)
        results[image_name.split('_')[0]]['ssim'].append(ssim)

out_path = 'statistics/'
saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')
