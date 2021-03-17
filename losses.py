import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.fft as fft
import kornia


class LaplacianLoss(nn.Module):
    def __init__(self, max_depth=5):
        super(LaplacianLoss, self).__init__()
        self.depth = max_depth

    def forward(self, out_images, target_images):
        raise NotImplementedError


class SSIM_FFT_Loss(nn.Module):
    def __init__(self, alpha=0.0001):
        super(SSIM_FFT_Loss, self).__init__()
        self.reconstructionLoss = ssimLoss()
        self.frequencyLoss = FrequencyDomainLoss()

        self.alpha = alpha

    def forward(self, out_images, target_images):

        return (self.reconstructionLoss(out_images, target_images) 
                + self.alpha * self.frequencyLoss(out_images, target_images))


class ThresholdLimitationLoss(nn.Module):
    def __init__(self):
        super(ThresholdLimitationLoss, self).__init__()

    def forward(self, out_image: torch.Tensor):
        clamped = out_image.clamp(0, 1)
        l1 = torch.log(torch.abs(clamped - out_image) + 1)
        return l1.mean()


class FrequencyDomainLoss(nn.Module):
    def __init__(self):
        super(FrequencyDomainLoss, self).__init__()
        self.l1 = nn.MSELoss()

    def forward(self, out_image, target_image):
        x = fft.fft(fft.fft(out_image, dim=3), dim=2)
        y = fft.fft(fft.fft(target_image, dim=3), dim=2)
        return self.l1(x.real, y.real) + self.l1(x.imag, y.imag)


class ssimLoss(nn.Module):
    def __init__(self):
        super(ssimLoss, self).__init__()
        self.ssim = kornia.losses.SSIMLoss(11)

    def forward(self, out_image, target_image):
        return self.ssim(out_image, target_image)


class psnrLoss(nn.Module):
    def __init__(self):
        super(psnrLoss, self).__init__()
        self.psnr = kornia.losses.PSNRLoss(1.0)

    def forward(self, out_image, target_image):
        return self.psnr(out_image, target_image)


class IndianLoss(nn.Module):
    def __init__(self):
        super(IndianLoss, self).__init__()
        self.perception_loss = PerceptionLoss()
        self.L2 = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):

        perception_loss = self.perception_loss(out_images, target_images)
        recons_loss = 0.6 * self.L1(out_images, target_images) + 0.4 * self.L2(out_images, target_images)
        tv_loss = self.tv_loss(out_images)

        loss = recons_loss + 0.006*perception_loss + 2e-8*tv_loss

        return loss


class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()

        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network

    def forward(self, out_images, target_images):
        return nn.MSELoss(self.loss_network(out_images), self.loss_network(target_images))


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
