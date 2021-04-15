from comet_ml import Experiment, ExistingExperiment
from dataset import SlidingWindowDataset
from e3d_lstm import E3DLSTM
from functools import lru_cache
from torch.utils.data import DataLoader
from utils import h5_virtual_file, window, weights_init
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from utils import psnr, ssim, upload_images
import argparse
import torchvision.transforms as transforms
from math import log10

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=True,
    help="path to folder for saving checkpoints",
)
parser.add_argument(
    "--checkpoint", type=str, help="path of checkpoint for pretrained model"
)
parser.add_argument(
    "--train_continue", action="store_true", help="resuming from checkpoint."
)
parser.add_argument(
    "-it",
    "--init_type",
    default="",
    type=str,
    help="the name of an initialization method: normal | xavier | kaiming | orthogonal",
)

parser.add_argument(
    "--epochs", type=int, default=200, help="number of epochs to train. Default: 200."
)
parser.add_argument(
    "-tbs",
    "--train_batch_size",
    type=int,
    default=20,  # for supervised learning use 384, for SimCLR use 220, for CURL use 110
    help="batch size for training. Default: 6.",
)
parser.add_argument(
    "-nw", "--num_workers", default=4, type=int, help="number of CPU you get"
)
parser.add_argument(
    "-vbs",
    "--validation_batch_size",
    type=int,
    default=20,
    help="batch size for validation. Default: 10.",
)
parser.add_argument(
    "-ilr",
    "--init_learning_rate",
    type=float,
    default=0.001,
    help="set initial learning rate. Default: 0.0001.",
)
parser.add_argument(
    "--milestones",
    type=list,
    default=[100, 150, 200, 250, 300, 350, 400, 450, 500],
    help="Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]",
)
parser.add_argument(
    "--progress_iter",
    type=int,
    default=100,
    help="frequency of reporting progress and validation. N: after every N iterations. Default: 100.",
)
parser.add_argument(
    "--logimagefreq", type=int, default=1, help="frequency of logging image.",
)
parser.add_argument(
    "--checkpoint_epoch",
    type=int,
    default=5,
    help="checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 5.",
)
parser.add_argument(
    "-wp", "--workspace", default="tianyu-z", type=str, help="comet-ml workspace"
)
parser.add_argument(
    "-dh", "--data_h", default=128, type=int, help="H of the data shape"
)
parser.add_argument(
    "-dw", "--data_w", default=128, type=int, help="W of the data shape"
)
parser.add_argument(
    "-pn", "--projectname", default="E3D", type=str, help="comet-ml project name",
)
parser.add_argument(
    "--nocomet", action="store_true", help="not using comet_ml logging."
)
parser.add_argument(
    "--cometid", type=str, default="", help="the comet id to resume exps",
)
parser.add_argument(
    "-rs",
    "--randomseed",
    type=int,
    default=2021,
    help="batch size for validation. Default: 10.",
)
parser.add_argument(
    "-vs",
    "--validationsample",
    type=int,
    default=300,
    help="batch size for validation. Default: 10.",
)
args = parser.parse_args()

random_seed = args.randomseed
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TaxiBJTrainer(nn.Module):
    def __init__(self, args=args):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.float
        self.args = args
        # TODO make all configurable
        self.num_epoch = args.epochs
        self.batch_size = args.train_batch_size

        self.input_time_window = 4
        self.output_time_horizon = 1
        self.temporal_stride = 1
        self.temporal_frames = 1
        self.time_steps = (
            self.input_time_window - self.temporal_frames + 1
        ) // self.temporal_stride

        # Initiate the network
        # CxT×H×W
        input_shape = (1, self.temporal_frames, 128, 128)
        output_shape = (1, self.output_time_horizon, 128, 128)

        self.tau = 1
        hidden_size = 64
        kernel = (1, 5, 5)
        lstm_layers = 4

        self.encoder = E3DLSTM(
            input_shape, hidden_size, lstm_layers, kernel, self.tau
        ).type(dtype)
        self.decoder = nn.Conv3d(
            hidden_size * self.time_steps, output_shape[0], kernel, padding=(0, 2, 2)
        ).type(dtype)

        if self.args.train_continue:
            if not self.args.nocomet and self.args.cometid != "":
                self.comet_exp = ExistingExperiment(
                    previous_experiment=self.args.cometid
                )
            elif not self.args.nocomet and self.args.cometid == "":
                self.comet_exp = Experiment(
                    workspace=self.args.workspace, project_name=self.args.projectname
                )
            else:
                self.comet_exp = None
            self.ckpt_dict = torch.load(self.args.checkpoint)
            self.load_state_dict(self.ckpt_dict["state_dict"])
            self.to(self.device)
            params = self.parameters(recurse=True)
            self.optimizer = torch.optim.Adam(
                params, lr=self.args.init_learning_rate, weight_decay=0
            )
            self.optimizer.load_state_dict(self.ckpt_dict["opt_state_dict"])
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.args.milestones, gamma=0.1
            )
        else:
            # start logging info in comet-ml
            if not self.args.nocomet:
                self.comet_exp = Experiment(
                    workspace=self.args.workspace, project_name=self.args.projectname
                )
                # self.comet_exp.log_parameters(flatten_opts(self.args))
            else:
                self.comet_exp = None
            self.ckpt_dict = {
                "trainLoss": {},
                "valLoss": {},
                "valPSNR": {},
                "valSSIM": {},
                "epoch": -1,
                "detail": "End to end E3D",
                "trainBatchSz": self.args.train_batch_size,
            }
            self.to(self.device)
            params = self.parameters(recurse=True)
            self.optimizer = torch.optim.Adam(
                params, lr=self.args.init_learning_rate, weight_decay=0
            )
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.args.milestones, gamma=0.1
            )

        # Setup optimizer

        # TODO learning rate scheduler
        # Weight decay stands for L2 regularization

        self.apply(weights_init())

    def forward(self, input_seq):
        return self.decoder(self.encoder(input_seq))

    def loss(self, input_seq, target):
        output = self(input_seq)

        l2_loss = F.mse_loss(output * 255, target * 255)
        l1_loss = F.l1_loss(output * 255, target * 255)
        # psnr_ = psnr(output, target)
        psnr_ = 10 * log10(255 / l2_loss)
        ssim_ = ssim(output, target)
        return l1_loss, l2_loss, output, psnr_, ssim_

    # @property
    # @lru_cache(maxsize=1)
    # def data(self):
    #     taxibj_dir = "./data/TaxiBJ/"
    #     # TODO make configurable
    #     f = h5_virtual_file(
    #         [
    #             f"{taxibj_dir}BJ13_M32x32_T30_InOut.h5",
    #             f"{taxibj_dir}BJ14_M32x32_T30_InOut.h5",
    #             f"{taxibj_dir}BJ15_M32x32_T30_InOut.h5",
    #             f"{taxibj_dir}BJ16_M32x32_T30_InOut.h5",
    #         ]
    #     )
    #     return f.get("data")

    @lru_cache(maxsize=1)
    def data(self, isTrain):
        if isTrain:
            path = "/miniscratch/tyz/datasets/CloudCast/200MB/pkls/train.pkl"
        else:
            path = "/miniscratch/tyz/datasets/CloudCast/200MB/pkls/test.pkl"
        # TODO make configurable
        with open(path, "rb") as file:
            f = pickle.load(file)

        return np.moveaxis(f, -1, 0)[:, np.newaxis, :, :]

    def get_trainloader(self, raw_data, shuffle=True):
        # NOTE note we do simple transformation, only approx within [0,1]
        dataset = SlidingWindowDataset(
            raw_data,
            self.input_time_window,
            self.output_time_horizon,
            lambda t: (t - t.min()) / (t.max() - t.min()),
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def validate(self, val_dataloader):
        self.eval()

        sum_l1_loss = 0
        sum_l2_loss = 0
        psnrs = 0
        ssims = 0
        image_log = []
        num_log = 0
        with torch.no_grad():
            for i, (input, target) in enumerate(val_dataloader):
                if i > self.args.validationsample:
                    break
                frames_seq = []

                for indices in window(
                    range(self.input_time_window),
                    self.temporal_frames,
                    self.temporal_stride,
                ):
                    # batch x channels x time x window x height
                    frames_seq.append(input[:, :, indices[0] : indices[-1] + 1])
                input = torch.stack(frames_seq, dim=0).to(self.device)
                target = target.to(self.device)

                l1_loss, l2_loss, output, psnr_, ssim_ = self.loss(input, target)
                sum_l1_loss += l1_loss
                sum_l2_loss += l2_loss
                # print("output", output.shape)
                psnrs += psnr_
                ssims += ssim_
                if num_log < 10:
                    image_log.append(target.cpu()[0].repeat(1, 3, 1, 1))
                    image_log.append(output.cpu()[0].repeat(1, 3, 1, 1))
                    num_log += 1
        print(f"Validation L1:{sum_l1_loss / (i + 1)}; L2: {sum_l2_loss / (i + 1)}")
        return (
            sum_l1_loss / (i + 1),
            sum_l2_loss / (i + 1),
            psnrs / (i + 1),
            ssims / (i + 1),
            image_log,
        )

    def resume_train(self, ckpt_path="./taxibj_trainer.pt", resume=False):
        # 2 weeks / 30min time step = 672
        train_dataloader = self.get_trainloader(self.data(True))
        val_dataloader = self.get_trainloader(self.data(False), False)

        best_loss = 999999
        best_psnr = -1
        best_ssim = -1
        for epoch in range(self.ckpt_dict["epoch"] + 1, args.epochs):
            trainloss = 0
            for i, (input, target) in enumerate(train_dataloader):
                frames_seq = []

                for indices in window(
                    range(self.input_time_window),
                    self.temporal_frames,
                    self.temporal_stride,
                ):
                    # batch x channels x time x window x height
                    frames_seq.append(input[:, :, indices[0] : indices[-1] + 1])

                input = torch.stack(frames_seq, dim=0).to(self.device)
                target = target.to(self.device)
                self.train()
                self.optimizer.zero_grad()
                l1_loss, l2_loss, _, _, _ = self.loss(input, target)
                loss = l1_loss + l2_loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                trainloss += loss.item()
                if i % 10 == 0:
                    print(
                        "Epoch: {}/{}, step: {}/{}, mse: {}".format(
                            epoch, args.epochs, i, len(train_dataloader), l2_loss
                        )
                    )

            l1loss, l2loss, psnr_, ssim_, image_log = self.validate(val_dataloader)
            upload_images(
                image_log,
                epoch=epoch,
                exp=self.comet_exp,
                im_per_row=2,
                rows_per_log=int(len(image_log) / 2),
            )
            self.ckpt_dict["trainLoss"][epoch] = trainloss
            self.ckpt_dict["valLoss"][epoch] = l1loss.item() + l2loss.item()
            self.ckpt_dict["valPSNR"][epoch] = psnr_
            self.ckpt_dict["valSSIM"][epoch] = ssim_
            self.ckpt_dict["epoch"] = epoch
            self.ckpt_dict["state_dict"] = self.state_dict()
            self.ckpt_dict["opt_state_dict"] = self.optimizer.state_dict()
            self.comet_exp.log_metric(
                "trainLoss", self.ckpt_dict["trainLoss"][epoch], epoch=epoch
            )
            self.comet_exp.log_metric(
                "valLoss", self.ckpt_dict["valLoss"][epoch], epoch=epoch
            )
            self.comet_exp.log_metric(
                "valPSNR", self.ckpt_dict["valPSNR"][epoch], epoch=epoch
            )
            self.comet_exp.log_metric(
                "valSSIM", self.ckpt_dict["valSSIM"][epoch], epoch=epoch
            )
            torch.save(self.ckpt_dict, self.args.checkpoint_dir + "/latest.pth")
            if best_ssim < self.ckpt_dict["valSSIM"][epoch]:
                best_ssim = self.ckpt_dict["valSSIM"][epoch]
                torch.save(
                    self.ckpt_dict, self.args.checkpoint_dir + "/bestvalSSIM.pth"
                )
                print("Best SSIM found at {}".format(best_ssim))
            elif best_psnr < self.ckpt_dict["valPSNR"][epoch]:
                best_psnr = self.ckpt_dict["valPSNR"][epoch]
                torch.save(
                    self.ckpt_dict, self.args.checkpoint_dir + "/bestvalPSNR.pth"
                )
                print("Best SSIM found at {}".format(best_psnr))
            elif best_loss > self.ckpt_dict["valLoss"][epoch]:
                best_loss = self.ckpt_dict["valLoss"][epoch]
                torch.save(self.ckpt_dict, self.args.checkpoint_dir + "/bestloss.pth")
                print("Best SSIM found at {}".format(best_loss))
            epoch += 1


if __name__ == "__main__":
    trainer = TaxiBJTrainer()
    trainer.resume_train(ckpt_path=args.checkpoint, resume=args.train_continue)
