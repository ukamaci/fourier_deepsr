import torch
from torch import nn
from torchvision.models.vgg import vgg16

import sys
sys.path.append('/home/berk/Documents/GitHub/fourier_deepsr/')
import metrics                                                              # import own metrics.py for importing FRC

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

        # FRC loss implementation
        self.frc = frc_loss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        # FRC Loss implementation
        frc = self.frc(out_images,target_images)

        # print('\n')
        print('kabuk korelasyonu kaybi', 2e-5 * frc)
        print('resimsel kayip', image_loss)
        # print('algisal kayip', 0.006 * perception_loss)
        # print('ters kayip', 0.001 * adversarial_loss)
        # print('\n')

        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss + 0* frc


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class frc_loss(nn.Module):
    def __init__(self):
        super(frc_loss, self).__init__()
        self.mse_loss = nn.MSELoss()
    def forward(self,batch1,batch2):
        loss = 0
        for batch in range(batch1.shape[0]):
            for ch in range(batch1.shape[1]):
                im1 = batch1[batch, ch, :, :]
                im2 = batch2[batch, ch, :, :]
                loss_ind = metrics.get_frc_torch(im1, im2)

                # # Look into whole frequencies
                # loss += self.mse_loss(loss_ind,torch.ones(loss_ind.shape))

                # Look into the mid 1/3 subpart
                l = loss_ind.shape[0]
                loss += self.mse_loss(loss_ind[l//3:2*l//3,:], torch.ones(loss_ind[l//3:2*l//3,:].shape))

        return loss

if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
