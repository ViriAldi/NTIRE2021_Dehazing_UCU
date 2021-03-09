import yaml
import os
import torch
import torch.nn as nn
import models
from datasets import NH_HazeDataset
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import metrics
import kornia


def read_config(path):
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.join(*[str(i) for i in seq])

    yaml.add_constructor('!join', join)

    with open(path) as f:
        return yaml.load(f)


def test(model, dataset, name):

    print(f"Testing on {name} dataset...")

    test_dataset = NH_HazeDataset(
        hazed_image_files = dataset["INPUT_FOLDER"],
        dehazed_image_files = dataset["GT"],
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
            image = (images['hazed_image']).cuda()
            gt = (images['dehazed_image']).cuda()

            # hue = model(image)
            # hsv = kornia.color.rgb_to_hsv((images['dehazed_image']).cuda())
            # hsv[:,0,:,:] = hue
            # dehazed_image = kornia.color.hsv_to_rgb(hsv)

            gt_hue, gt_sat, gt_val = torch.chunk(kornia.color.rgb_to_hsv(gt), 3, 1)
            gt_hue = kornia.color.hsv_to_rgb(torch.cat([gt_hue, torch.ones(gt_hue.shape).cuda(), torch.ones(gt_hue.shape).cuda()], 1))

            if not os.path.exists(dataset["OUTPUT_FOLDER"]):
                os.makedirs(dataset["OUTPUT_FOLDER"])

            torchvision.utils.save_image(image.data, os.path.join(dataset["OUTPUT_FOLDER"], f"rs{name}_{iteration}.png"))
            torchvision.utils.save_image(gt_hue.data, os.path.join(dataset["OUTPUT_FOLDER"], f"hue{name}_{iteration}.png"))
            torchvision.utils.save_image(gt_sat.data, os.path.join(dataset["OUTPUT_FOLDER"], f"sat{name}_{iteration}.png"))
            torchvision.utils.save_image(gt_val.data, os.path.join(dataset["OUTPUT_FOLDER"], f"val{name}_{iteration}.png"))


            # torchvision.utils.save_image(dehazed_image.data, os.path.join(dataset["OUTPUT_FOLDER"], f"{name}_{iteration}.png"))

    print("Done")


def validate(model, dataset, name):

    print(f"Validating on {name} dataset...")

    valid_dataset = NH_HazeDataset(
        hazed_image_files = dataset["DATA_HAZY"],
        dehazed_image_files = dataset["DATA_FREE"],
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False
    )

    second_ssim = 0
    second_psnr = 0

    for image in valid_dataloader:
        with torch.no_grad():
            gt = (image['dehazed_image']).cuda()                        
            image = (image['hazed_image']).cuda()

            gt_hue, gt_sat, gt_val = torch.chunk(kornia.color.rgb_to_hsv(gt), 3, 1)
            gt_hue = kornia.color.hsv_to_rgb(torch.cat([gt_hue, torch.ones(gt_hue.shape).cuda(), torch.ones(gt_hue.shape).cuda()], 1))

            # colors = model(image)
            # final_result = colors * torchvision.transforms.functional.rgb_to_grayscale(color, num_output_channels=3) / torchvision.transforms.functional.rgb_to_grayscale(colors, num_output_channels=3)

            second_psnr += metrics.psnr(final_result, gt)
            second_ssim += metrics.ssim(final_result, gt)

    second_ssim = round(100 * second_ssim.item() / len(valid_dataset), 1)
    second_psnr = round(second_psnr.item() / len(valid_dataset), 1)

    print(f"{name} dataset SSIM score {second_ssim}%")
    print(f"{name} dataset PSNR score {second_psnr}db")


def test_and_validate(config, model):

    # for name, dataset in config["VALIDATION"].items():
    #     validate(model, dataset, name)
    
    for name, dataset in config["TESTING"].items():
        test(model, dataset, name)

if __name__ == "__main__":
    config = read_config("config/config_test.yaml")
    model = models.HueEstimator().cuda()
    # model.load_state_dict(torch.load(config["MODEL"]))

    test_and_validate(config, model)
