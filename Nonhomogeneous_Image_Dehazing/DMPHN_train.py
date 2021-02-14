import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import random
import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from datasets import NH_HazeDataset
import time
from loss import CustomLoss_function
from patcher import patch, convert
import yaml
import metrics
from torch.utils.tensorboard import SummaryWriter


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def train(config):

    METHOD=config["METHOD"]
    GPU=config["TRAINING"]["GPU"]
    LEARNING_RATE=config["TRAINING"]["LEARNING_RATE"]
    EPOCHS=config["TRAINING"]["EPOCHS"]
    
    model = nn.DataParallel(models.DoubleMultiPatchExtended().cuda(GPU))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.2)

    if config["PRETRAINED"]["USE_PRETRAINED"]:
        model.load_state_dict(torch.load(os.path.join(config["PRETRAINED"]["PRETRAINED_FOLDER"], "pre_trained_combined.pkl")))

    writer = SummaryWriter()
    glob_id = 0
            
    for epoch in range(config["TRAINING"]["START_EPOCH"], EPOCHS):

        datasets = []

        for dset in config["TRAINSETS"]["DATASETS"].values():
            train_dataset = NH_HazeDataset(
                hazed_image_files = dset["CROPED_DATA_HAZY"],
                dehazed_image_files = dset["CROPED_DATA_FREE"],
                transform = transforms.Compose([
                    transforms.ToTensor()
                ])
            )
            datasets.append(train_dataset)

        train_dataset = torch.utils.data.ConcatDataset(datasets)

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=dset["BATCH_SIZE"],
            shuffle=True, 
            num_workers=dset["NUM_WORKERS"]
        )
        
        start = 0
        for iteration, images in enumerate(train_dataloader):

            custom_loss_fn1 = CustomLoss_function().cuda(GPU)
            custom_loss_fn2 = CustomLoss_function().cuda(GPU)
            
            gt = Variable(images['dehazed_image'] - 0.5).cuda(GPU) 
            images_lv1 = Variable(images['hazed_image'] - 0.5).cuda(GPU)
            middle, dehazed_image = model(images_lv1)

            loss_middle = custom_loss_fn2(middle, gt)
            loss_end = custom_loss_fn2(dehazed_image, gt)
            loss = 0.8 * loss_end + 0.2 * loss_middle
            model.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss_subnet1", loss_middle, glob_id)
            writer.add_scalar("Loss_subnet2", loss_end, glob_id)
            writer.add_scalar("Total loss", loss, glob_id)

            glob_id += 1
            
            if (iteration + 1) % config["TRAINING"]["LOGGING_FREQ"] == 0:
                print("epoch:", epoch, "iteration:", iteration + 1, "loss1: %.4f" % loss_middle.item(), "loss2: %.4f" % loss_end.item())
        
        if epoch % config["TESTING"]["EPOCH_FREQ"] == 0:
            test(config, model, epoch)

        if epoch % config["VALIDATION"]["EPOCH_FREQ"] == 0:
            validate(config, model, writer, epoch)

        if not os.path.exists(config["MODEL_SAVING"]["SAVING_FOLDER"]):
            os.makedirs(config["MODEL_SAVING"]["SAVING_FOLDER"])

        torch.save(model.state_dict(), os.path.join(config["MODEL_SAVING"]["SAVING_FOLDER"], "mutipatch_v2.pkl"))

        scheduler.step()

    writer.close()


def test(config, model, epoch=-1):

    print(f"Testing model after {epoch}th epoch...")

    test_dataset = NH_HazeDataset(
        hazed_image_files = config["TESTING"]["DATA_HAZY"],
        dehazed_image_files = config["TESTING"]["DATA_HAZY"],
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=False
    )

    for iteration, images in enumerate(test_dataloader):
        with torch.no_grad():                                   
            images_lv1 = Variable(images['hazed_image'] - 0.5).cuda(config["TRAINING"]["GPU"])
            _, dehazed_image = model(images_lv1)

            if not os.path.exists(os.path.join(config["TESTING"]["DEHAZED_PATH"], f"epoch{epoch}")):
                os.makedirs(os.path.join(config["TESTING"]["DEHAZED_PATH"], f"epoch{epoch}"))

            torchvision.utils.save_image(dehazed_image.data + 0.5, os.path.join(config["TESTING"]["DEHAZED_PATH"], f"epoch{epoch}", f"{iteration}_e{epoch}.png"))


def validate(config, model, writer, epoch=-1):

    print(f"Validating model after {epoch}th epoch...")

    for name, dset in config["VALIDATION"]["DATASETS"].items():

        valid_dataset = NH_HazeDataset(
            hazed_image_files = dset["DATA_HAZY"],
            dehazed_image_files = dset["DATA_FREE"],
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        )

        n = len(valid_dataset)

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False
        )

        first_ssim = 0
        second_ssim = 0

        first_psnr = 0
        second_psnr = 0

        for iteration, images in enumerate(valid_dataloader):
            with torch.no_grad():
                gt = Variable(images['dehazed_image'] - 0.5).cuda(config["TRAINING"]["GPU"])                        
                images_lv1 = Variable(images['hazed_image'] - 0.5).cuda(config["TRAINING"]["GPU"])
                middle_result, final_result = model(images_lv1)

                middle_result += 0.5
                final_result += 0.5
                gt += 0.5

                first_psnr += metrics.psnr(middle_result, gt)
                second_psnr += metrics.psnr(final_result, gt)

                first_ssim += metrics.ssim(middle_result, gt)
                second_ssim += metrics.ssim(final_result, gt)

        first_ssim = round(100 * first_ssim.item() / n, 1)
        second_ssim = round(100 * second_ssim.item() / n, 1)

        first_psnr = round(first_psnr.item() / n, 1)
        second_psnr = round(second_psnr.item() / n, 1)

        print(f"{name} dataset SSIM score after {epoch}th epoch: {first_ssim}% and {second_ssim}%")
        print(f"{name} dataset PSNR score after {epoch}th epoch: {first_psnr}db and {second_psnr}db")

        writer.add_scalar(f"{name}_dataset_SSIM_subnet1", first_ssim, epoch)
        writer.add_scalar(f"{name}_dataset_SSIM_subnet2", second_ssim, epoch)
        writer.add_scalar(f"{name}_dataset_PSNR_subnet1", first_psnr, epoch)
        writer.add_scalar(f"{name}_dataset_PSNR_subnet2", second_psnr, epoch)


if __name__ == '__main__':
    with open("config/config.yaml") as f:
        config = yaml.load(f)

    for dset in config["TRAINSETS"]["DATASETS"].values():
        if not dset["ALREADY_CROPED"]:
            convert(dset["ORIG_DATA_HAZY"], dset["CROPED_DATA_HAZY"], dset["CROP_TIMES"])
            convert(dset["ORIG_DATA_FREE"], dset["CROPED_DATA_FREE"], dset["CROP_TIMES"])

    train(config)
