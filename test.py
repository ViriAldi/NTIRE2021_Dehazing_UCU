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

    model1 = models.MultiPatchSMP().cuda()
    model1.load_state_dict(torch.load("models/SMP_VAL_143E.pkl"))

    model2 = models.SuperHue().cuda()
    model2 = nn.DataParallel(model2)
    model2.load_state_dict(torch.load("models/MIN_SMP.pkl"))

    model3 = models.ColorSegmentator().cuda()
    model3 = nn.DataParallel(model3)
    model3.load_state_dict(torch.load("models/MAX_COLOR.pkl"))

    model4 = models.ColorSegmentator().cuda()
    model4 = nn.DataParallel(model4)
    model4.load_state_dict(torch.load("models/MIN_COLOR.pkl"))

    for iteration, images in enumerate(test_dataloader):
        with torch.no_grad():                                   
            image = (images['hazed_image']).cuda()
            gt = (images['dehazed_image']).cuda()

            # final_result = torch.zeros(image.shape).cuda()
            # x, y = image.shape[2] // 5, image.shape[3] // 5

            # final_result = model(image)

            # colors = model(image)
            # final_result = colors * torchvision.transforms.functional.rgb_to_grayscale(color, num_output_channels=3) / torchvision.transforms.functional.rgb_to_grayscale(colors, num_output_channels=3)

            # for i in range(5):
            #     for j in range(5):
            #         final_result[:,1,i*x: i*x + x, y*j: y*j + y], final_result[:,2,i*x: i*x + x, y*j: y*j + y] = model1(image[:,:,i*x: i*x + x, y*j: y*j + y])

            # final_result[:,0,:,:] = model(image)

            # final_result = kornia.color.hsv_to_rgb(final_result)

            # if not os.path.exists(dataset["OUTPUT_FOLDER"]):
            #     os.makedirs(dataset["OUTPUT_FOLDER"])

            # torchvision.utils.save_image(final_result.data, os.path.join(dataset["OUTPUT_FOLDER"], f"rs{name}_{iteration}.png"))
            # torchvision.utils.save_image(gt_hue.data, os.path.join(dataset["OUTPUT_FOLDER"], f"hue{name}_{iteration}.png"))
            # torchvision.utils.save_image(gt_sat.data, os.path.join(dataset["OUTPUT_FOLDER"], f"sat{name}_{iteration}.png"))
            # torchvision.utils.save_image(gt_val.data, os.path.join(dataset["OUTPUT_FOLDER"], f"val{name}_{iteration}.png"))


            # torchvision.utils.save_image(dehazed_image.data, os.path.join(dataset["OUTPUT_FOLDER"], f"{name}_{iteration}.png"))

            result = torch.zeros(image.shape).cuda()
            result1 = torch.zeros(image.shape).cuda()
            result2 = torch.zeros(image.shape).cuda()
            result3 = torch.zeros(image.shape).cuda()
            result4 = torch.zeros(image.shape).cuda()
            counter = torch.zeros(image.shape).cuda()

            s = 5
            x, y = image.shape[2] // s, image.shape[3] // s
            k = 1
            batch = 16

            lst = []
            lst_result = []
            lst_mid = []
            lst_mxcol = []
            lst_mncol = []

            for i in range(s * k):
                for j in range(s * k):
                    img = image[:,:,i*x // k: i*x // k + x, y*j // k: y*j // k + y]
                    if img.shape[2] != x or img.shape[3] != y:
                        continue
                    lst.append(img)

            stk = torch.cat(lst, 0).split(batch, 0)

            stk_tmp = []
            vl = []
            for inp in stk:
                val = model1(inp)
                vl.append(val)
                stk_tmp.append(torch.cat([inp[:,0:1,:,:], val, inp[:,1:2,:,:], val, inp[:,2:3,:,:], val], 1))

            stk = stk_tmp

            for inp in stk:
                lst_mid.append(model2(inp))
                lst_result.append(model(inp))
                lst_mxcol.append(model3(inp))
                lst_mncol.append(model4(inp))
            
            lsmncl = torch.cat(lst_mncol, 0).split(1, 0)
            lsmxcl = torch.cat(lst_mxcol, 0).split(1, 0)
            lsmd = torch.cat(lst_mid, 0).split(1, 0)
            ls = torch.cat(lst_result, 0).split(1, 0)
            vl = torch.cat(vl, 0).split(1, 0)

            idx = 0
            for i in range(s * k):
                for j in range(s * k):
                    img = image[:,:,i*x // k: i*x // k + x, y*j // k: y*j // k + y]
                    if img.shape[2] != x or img.shape[3] != y:
                        continue
                    result[:,:,i*x // k: i*x // k + x, y*j // k: y*j // k + y] += ls[idx]
                    result1[:,:,i*x // k: i*x // k + x, y*j // k: y*j // k + y] += vl[idx]
                    result2[:,:,i*x // k: i*x // k + x, y*j // k: y*j // k + y] += lsmd[idx]
                    result3[:,:,i*x // k: i*x // k + x, y*j // k: y*j // k + y] += lsmxcl[idx]
                    result4[:,:,i*x // k: i*x // k + x, y*j // k: y*j // k + y] += lsmncl[idx]
                    counter[:,:,i*x // k: i*x // k + x, y*j // k: y*j // k + y] += 1
                    idx += 1

            result = result / counter
            result1 = result1 / counter
            result2 = result2 / counter
            result3 = result3 / counter
            result4 = result4 / counter

            idx = 0

            for i in range(s * k):
                for j in range(s * k):
                    torchvision.utils.save_image(result1[:,:,i*x // k: i*x // k + x, y*j // k: y*j // k + y], "prepared/NTIRE2021_5X5_MAX/" + str(len(test_dataloader) * iteration + idx).rjust(5, "0") + ".png")
                    torchvision.utils.save_image(result[:,:,i*x // k: i*x // k + x, y*j // k: y*j // k + y], "prepared/NTIRE2021_5X5_MID/" + str(len(test_dataloader) * iteration + idx).rjust(5, "0") + ".png")
                    torchvision.utils.save_image(result2[:,:,i*x // k: i*x // k + x, y*j // k: y*j // k + y], "prepared/NTIRE2021_5X5_MIN/" + str(len(test_dataloader) * iteration + idx).rjust(5, "0") + ".png")
                    idx += 1



            # sm = torch.sum(gt, 1, keepdim=True)

            # mn = result2
            # mx = result1
            # md = result

            # # mn = 2 * result - result1
            # # mx = result1
            # # md = sm - mx - mn

            # # print(metrics.ssim(torch.cat([])))

            # print(abs(mn - torch.min(gt, 1)[0]).mean())
            # print(abs(mx - torch.max(gt, 1)[0]).mean())
            # print(abs(md - sm + torch.max(gt, 1)[0] + torch.min(gt, 1)[0]).mean())

            # result = torch.zeros((gt.shape)).cuda()

            # # mx_mask = torch.max(gt, 1, keepdim=True)[1]
            # # gt_max = torch.cat([mx_mask == 0, mx_mask == 1, mx_mask == 2], 1)

            # mx_mask = torch.max(result3, 1, keepdim=True)[1]
            # gt_max = torch.cat([mx_mask == 0, mx_mask == 1, mx_mask == 2], 1)

            # # mn_mask = torch.min(gt, 1, keepdim=True)[1]
            # # gt_min = torch.cat([mn_mask == 0, mn_mask == 1, mn_mask == 2], 1)

            # mn_mask = torch.max(result4, 1, keepdim=True)[1]
            # gt_min = torch.cat([mn_mask == 0, mn_mask == 1, mn_mask == 2], 1)

            # gt_mid = torch.ones(gt.shape).cuda() - gt_max.int() - gt_min.int()

            # result += gt_max * mx
            # result += gt_mid * md
            # result += gt_min * mn

            # print(abs(result[:,0,:,:] - gt[:,0,:,:]).mean())
            # print(abs(result[:,1,:,:] - gt[:,1,:,:]).mean())
            # print(abs(result[:,2,:,:] - gt[:,2,:,:]).mean())

            # print(torch.square(result[:,0,:,:] - gt[:,0,:,:]).mean())
            # print(torch.square(result[:,1,:,:] - gt[:,1,:,:]).mean())
            # print(torch.square(result[:,2,:,:] - gt[:,2,:,:]).mean())

            # print(metrics.ssim(result, gt))
            # print(metrics.psnr(result, gt))

            # if not os.path.exists(dataset["OUTPUT_FOLDER"]):
            #     os.makedirs(dataset["OUTPUT_FOLDER"])

            # torchvision.utils.save_image(result.data, os.path.join(dataset["OUTPUT_FOLDER"], f"{name}_{iteration}_rolf.png"))

            print("dn")

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

            final_result = torch.zeros(image.shape).cuda()
            x, y = image.shape[2:]

            # colors = model(image)
            # final_result = colors * torchvision.transforms.functional.rgb_to_grayscale(color, num_output_channels=3) / torchvision.transforms.functional.rgb_to_grayscale(colors, num_output_channels=3)

            for i in range(10):
                for j in range(10):
                    final_result[:,:,i*x: i*x + i, y*j: y*j + y] = model(image[:,:,i*x: i*x + i, y*j: y*j + y])


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
    model = models.SuperHue().cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(config["MODEL"]))

    test_and_validate(config, model)