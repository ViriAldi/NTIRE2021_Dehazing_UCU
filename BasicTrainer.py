import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from datasets import MergedDataset, NH_HazeDataset
import losses
import models


class BasicTrainer:

    def __init__(self, cfg: dict):

        self.cfg = cfg
        self.model = getattr(models, cfg["MODEL"]["ARCHITECTURE"])().cuda()
        
        if cfg["MODEL"]["USE_PRETRAINED"]:
            self.model.load_state_dict(torch.load(cfg["MODEL"]["PRETRAINED_MODEL"]))
        if cfg["TRAINING"]["USE_PARALLEL"]:
            self.model = nn.DataParallel(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["TRAINING"]["LEARNING_RATE"])

        self.use_scheduler = cfg["TRAINING"]["USE_SCHEDULER"]
        if self.use_scheduler:
            self.scheduler = StepLR(self.optimizer, 
                                step_size=cfg["TRAINING"]["SCHEDULER"]["STEP"], 
                                gamma=cfg["TRAINING"]["SCHEDULER"]["GAMMA"])

        self.use_tensorboard = cfg["TRAINING"]["USE_TENSORBOARD"]
        if self.use_tensorboard:
            self.tensorboard_writer = None
            self.iter_idx = 0

        self.criterio = getattr(losses, cfg["MODEL"]["LOSS"]["ARCH"])().cuda()

        self.loss = 0
        self.epoch = 1

        self.num_epoch = cfg["TRAINING"]["EPOCHS"]

        self.log_freq = cfg["TRAINING"]["LOGGING_FREQ"]
        self.test_freq = cfg["TRAINING"]["TEST_FREQ"]
        self.valid_freq = cfg["TRAINING"]["VALID_FREQ"]


    def run_epoch(self):

        train_dataloader = self.make_dataloader(self.cfg["TRAINING_DATA"])
                
        for idx, image in enumerate(train_dataloader):

            self.step(image)
            if (idx + 1) % self.log_freq == 0:
                self.log(idx)
        
        if self.epoch % self.test_freq == 0:
            self.test()

        if self.epoch % self.valid_freq == 0:
            self.validate()

        self.make_checkpoint()
        if self.use_scheduler:
            self.scheduler.step()
        self.epoch += 1

    def train(self):
        if self.use_tensorboard:
            self.tensorboard_writer = SummaryWriter()

        for _ in range(self.num_epoch):
            self.run_epoch()

        if self.use_tensorboard:
            self.tensorboard_writer.close()

    def make_checkpoint(self):

        folder = self.cfg["MODEL"]["SAVING_FOLDER"]
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(self.model.state_dict(), 
                    os.path.join(folder, f"{self.cfg['MODEL']['NAME']}_E{self.epoch}_checkpoint.pkl"))

    @staticmethod
    def make_dataloader(cfg):
        datasets_sequential = []
        for dset in cfg["DATASETS"].values():
            datasets_parallel = []
            for dset_parallel in dset.values():
                _dataset = NH_HazeDataset(
                    hazed_image_files = dset_parallel["DATA_HAZY"],
                    dehazed_image_files = dset_parallel["DATA_FREE"],
                    transform = transforms.Compose([
                        transforms.ToTensor()
                    ])
                )
                datasets_parallel.append(_dataset)
            datasets_sequential.append(MergedDataset(datasets_parallel))
        dataset = torch.utils.data.ConcatDataset(datasets_sequential)

        dataloader = DataLoader(
            dataset, 
            batch_size=cfg["BATCH_SIZE"],
            shuffle=cfg["SHUFFLE"], 
            num_workers=cfg["NUM_WORKERS"]
        )

        return dataloader

    def step(self, input):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError
