import os

import kornia
import torch

import BasicTrainer
import utils


class Trainer(BasicTrainer.BasicTrainer):

    def get_result(self, inp):
        """
        IMPLEMENT THIS METHOD DUE TO CURRENT ARCHITECTURE
        """
        return self.model(inp)

    def step(self, input):
        """
        IMPLEMENT THIS METHOD DUE TO CURRENT ARCHITECTURE
        """

        hazed = input["hazed_image"]
        gt = input["dehazed_image"]

        output = self.get_result(hazed)
        self.loss = self.criterio(output, gt)

        self.model.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        if self.use_tensorboard:
            self.tensorboard_writer.add_scalar("Loss", self.loss, self.iter_idx)
            self.iter_idx += 1

    def test(self):

        print(f"Testing model after {self.epoch}th epoch...")

        with torch.no_grad():

            dataloader = self.make_dataloader(self.cfg["TESTING_DATA"])

            for idx, input in enumerate(dataloader):
                
                hazed = input["hazed_image"]
                gt = input["gt"]

                result = self.get_result(hazed)
                self.save_image(result, idx)

    def validate(self):
        """
        IMPLEMENT THIS METHOD DUE TO CURRENT ARCHITECTURE
        """

        print(f"Validating model after {self.epoch}th epoch...")

        with torch.no_grad():
            for name, dataset in self.cfg["VALIDATON_DATA"].items():

                dataloader = self.make_dataloader(dataset)
                ssim = 0
                psnr = 0

                for input in dataloader:

                    hazed = input["hazed_image"]
                    gt = input["gt"]
                    output = self.get_result(hazed)

                    psnr += metrics.psnr(output, gt)
                    ssim += metrics.ssim(output, gt)

                ssim_avg = round(ssim.item() / len(dataloader), 1)
                psnr_avg = round(100 * psnr.item() / len(dataloader), 1)

                print(f"{name} dataset SSIM score after {self.epoch}th epoch: {self.ssim_avg}%")
                print(f"{name} dataset PSNR score after {self.epoch}th epoch: {self.psnr_avg}db")

                if self.use_tensorboard:
                    self.tensorboard_writer.add_scalar(f"{name}_dataset_SSIM_subnet1", ssim_avg, self.epoch)
                    self.tensorboard_writer.add_scalar(f"{name}_dataset_PSNR_subnet1", psnr_avg, self.epoch)


if __name__ == '__main__':
    cfg = utils.read_config("config/config_train.yaml")
    trainer = Trainer(cfg)
    trainer.train()
