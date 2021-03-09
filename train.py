import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import L1Loss, MSELoss
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
from loss import ssimLoss, psnrLoss, CustomLoss_function
from patcher import patch, convert
import yaml
import metrics
from torch.utils.tensorboard import SummaryWriter
import kornia


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

    GPU=config["TRAINING"]["GPU"]
    LEARNING_RATE=config["TRAINING"]["LEARNING_RATE"]
    EPOCHS=config["TRAINING"]["EPOCHS"]

    model = models.HueTrainer("models/monster_saturation_v0.pkl", "models/monster_value_v0.pkl").cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.6)

    if config["PRETRAINED"]["USE_PRETRAINED"]:
        model.load_state_dict(torch.load(config["PRETRAINED"]["PRETRAINED_MODEL"]))

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

        ls = 0
        ls_hue = 0
        ls_sat = 0
        ls_value = 0
        ls_general = 0

        print(f"NEW EPOCH!")
        
        for iteration, images in enumerate(train_dataloader):

            # custom_loss_fn = CustomLoss_function().cuda(GPU)
            # custom_loss_fn1 = nn.MSELoss().cuda(GPU)
            custom_loss_fn = ssimLoss().cuda(GPU)

            # loss_hue = nn.L1Loss().cuda(GPU)
            # loss_sat = nn.L1Loss().cuda(GPU)
            # loss_value = CustomLoss_function().cuda(GPU)
            # loss_general = CustomLoss_function().cuda(GPU)

            gt = Variable(images['dehazed_image']).cuda(GPU) 
            images_lv1 = Variable(images['hazed_image']).cuda(GPU)

            # hue, sat, val = model(images_lv1)

            # hue_nograd = torch.clone(hue).detach()
            # result = kornia.color.hsv_to_rgb(torch.cat([model.xy2hue(hue_nograd), sat, val], 1))

            # gt_hue, gt_sat, gt_val = torch.chunk(kornia.color.rgb_to_hsv(gt), 3, 1)
            # gt_hue =  model.hue2xy(gt_hue)

            # loss = loss_hue(hue, gt_hue) * 0.4 + 0.25 * loss_sat(sat, gt_sat) + 0.25 * loss_value(torch.cat([val, val, val], 1), torch.cat([gt_val, gt_val, gt_val], 1)) + 0.5 * loss_general(result, gt)
            # loss = loss_general(result, gt)
            # loss = CustomLoss_function().cuda()(model(images_lv1), gt)
            loss = custom_loss_fn(gt, model(images_lv1))
            ls += loss

            # with torch.no_grad():
            #     ls_hue += loss_hue(hue, gt_hue)
            #     ls_sat += loss_sat(sat, gt_sat)
            #     ls_value += loss_value(torch.cat([val, val, val], 1), torch.cat([gt_val, gt_val, gt_val], 1))
            #     ls_general += loss_general(result, gt)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss", loss, glob_id)

            glob_id += 1
            
            if (iteration + 1) % config["TRAINING"]["LOGGING_FREQ"] == 0:
                print("#" * 100)
                # for x, name in zip([ls, ls_hue, ls_sat, ls_value, ls_general], ["ALL", "HUE", "SAT", "VAL", "IMG"]):
                for x, name in zip([ls], ["loss"]):
                    print("epoch:", epoch, "iteration:", iteration + 1, f"loss {name}: {round(x.item() / (iteration + 1), 4)}")
        
        if epoch % config["TESTING"]["EPOCH_FREQ"] == 0:
            test(config, model, epoch)

        if epoch % config["VALIDATION"]["EPOCH_FREQ"] == 0:
            validate(config, model, writer, epoch)

        if not os.path.exists(config["MODEL_SAVING"]["SAVING_FOLDER"]):
            os.makedirs(config["MODEL_SAVING"]["SAVING_FOLDER"])

        torch.save(model.state_dict(), os.path.join(config["MODEL_SAVING"]["SAVING_FOLDER"], config["MODEL_SAVING"]["SAVING_NAME"]))

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
            images = Variable(images['hazed_image']).cuda(config["TRAINING"]["GPU"])

            # result = [torch.zeros(images_lv1.shape) for _ in range(5)]
            # result = torch.zeros(images_lv1.shape)
            # l, r = images_lv1.shape[2] // 10, images_lv1.shape[3] // 10

            # for i in range(10):
            #     for j in range(10):
            #         result[:,:, l * i:l * (i + 1), r * j:r * (j+1)] = model(images_lv1[:,:, l * i:l * (i + 1), r * j:r * (j+1)])
                    # rs = model(images_lv1[:,:, l * i:l * (i + 1), r * j:r * (j+1)], test=True)
                    # for k in range(5):
                    #     result[k][:,:, l * i:l * (i + 1), r * j:r * (j+1)] = rs[k]

            # hue, sat, val = model(images)
            # hue = models.Monster.xy2hue(hue)
            # result = kornia.color.hsv_to_rgb(torch.cat([hue, sat, val], 1))

            # hue = kornia.color.hsv_to_rgb(torch.cat([hue, torch.ones(hue.shape).cuda(), torch.ones(hue.shape).cuda()], 1))
            result = model(images)

            if not os.path.exists(os.path.join(config["TESTING"]["DEHAZED_PATH"], f"epoch{epoch}")):
                os.makedirs(os.path.join(config["TESTING"]["DEHAZED_PATH"], f"epoch{epoch}"))

            torchvision.utils.save_image(result.data, os.path.join(config["TESTING"]["DEHAZED_PATH"], f"epoch{epoch}", f"result_{iteration}_e{epoch}.png"))
            # torchvision.utils.save_image(hue.data, os.path.join(config["TESTING"]["DEHAZED_PATH"], f"epoch{epoch}", f"hue_{iteration}_e{epoch}.png"))
            # torchvision.utils.save_image(sat.data, os.path.join(config["TESTING"]["DEHAZED_PATH"], f"epoch{epoch}", f"sat_{iteration}_e{epoch}.png"))
            # torchvision.utils.save_image(val.data, os.path.join(config["TESTING"]["DEHAZED_PATH"], f"epoch{epoch}", f"val_{iteration}_e{epoch}.png"))


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
        first_psnr = 0

        for iteration, images in enumerate(valid_dataloader):
            with torch.no_grad():
                gt = Variable(images['dehazed_image'] - 0.5).cuda(config["TRAINING"]["GPU"])
                images_lv1 = Variable(images['hazed_image'] - 0.5).cuda(config["TRAINING"]["GPU"])

                final_result = torch.zeros(images_lv1.shape).cuda()
                l, r = images_lv1.shape[2] // 10, images_lv1.shape[3] // 10

                for i in range(10):
                    for j in range(10):
                        final_result[:,:, l * i:l * (i + 1), r * j:r * (j+1)] = model(images_lv1[:,:, l * i:l * (i + 1), r * j:r * (j+1)])

                first_psnr += metrics.psnr(final_result, gt)
                first_ssim += metrics.ssim(final_result, gt)

        first_ssim = round(100 * first_ssim.item() / n, 1)
        first_psnr = round(first_psnr.item() / n, 1)

        print(f"{name} dataset SSIM score after {epoch}th epoch: {first_ssim}%")
        print(f"{name} dataset PSNR score after {epoch}th epoch: {first_psnr}db")

        writer.add_scalar(f"{name}_dataset_SSIM_subnet1", first_ssim, epoch)
        writer.add_scalar(f"{name}_dataset_PSNR_subnet1", first_psnr, epoch)


def read_config(path):
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.join(*[str(i) for i in seq])

    yaml.add_constructor('!join', join)

    with open(path) as f:
        return yaml.load(f)


if __name__ == '__main__':
    config = read_config("config/config_train.yaml")

    for dset in config["TRAINSETS"]["DATASETS"].values():
        if not dset["ALREADY_CROPED"]:
            convert(dset["ORIG_DATA_HAZY"], dset["CROPED_DATA_HAZY"], dset["CROP_TIMES"])
            convert(dset["ORIG_DATA_FREE"], dset["CROPED_DATA_FREE"], dset["CROP_TIMES"])

    train(config)
