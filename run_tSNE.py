import argparse
import os
import os.path as osp
from datetime import timedelta
import time
import sys
import random

import easydict
import numpy as np
import yaml

import torch
import torch.utils.data as data
from torch.backends import cudnn

from tSNE import t_SNE
from model.make_model_clip import build_model

start_epoch = best_mAP = 0

def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False
        # , pin_memory=True
    )
    return cluster_loader


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    cudnn.benchmark = True

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("==> Load unlabeled dataset")


    model = build_model(args, 395, 395)

    checkpoint = torch.load(args.resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.to("cuda")

    t_SNE(args, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Contrastive learning on unsupervised Cross re-ID")
    args_main = parser.parse_args()

    args = yaml.load(open('config/config_sysu.yaml'), Loader=yaml.FullLoader)
    args = easydict.EasyDict(args)
    main(args)
