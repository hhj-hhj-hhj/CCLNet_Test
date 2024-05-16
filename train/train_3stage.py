import argparse
import collections
import os
import os.path as osp
import shutil
from datetime import timedelta
import time
import sys
import random

import easydict
import numpy as np
import yaml
from sklearn.cluster import DBSCAN

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.cuda.amp import autocast
import torchvision.transforms as transforms
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
from torch.cuda import amp

from util.eval_metrics import extract_features_clip
from util.faiss_rerank import compute_jaccard_distance
from util.loss.supcontrast import SupConLoss
from util.utils import AverageMeter
from model.make_model_clip import load_clip_to_cpu
from util.optim.scheduler_p2w import cosine_lr
from model.img2text import IMG2TEXT
from util.make_optimizer import make_optimizer_3stage

from ClusterContrast.cm import ClusterMemory
from data.data_manager import process_query_sysu, process_gallery_sysu
from data.data_manager import process_test_regdb
from data.dataloader import SYSUData_Stage2, RegDBData_Stage2, IterLoader, TestData
from util.eval import tester
from util.utils import IdentitySampler_nosk, GenIdx
from model.img2text import get_loss_img2text

from util.make_optimizer import make_optimizer_2stage, make_optimizer_2stage_later
from util.optim.lr_scheduler import WarmupMultiStepLR


def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False
        # , pin_memory=True
    )
    return cluster_loader

def do_train_stage3(args,
                    model,
                    img2text,
                    clip_model):
    best_acc = 0
    device = 'cuda'
    epochs = args.stage3_maxepochs
    start_time = time.monotonic()

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.dataset == 'sysu':
        transform_train_rgb = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.ToTensor(),
            normalizer,
            transforms.RandomErasing(p=0.5)
        ])
        transform_train_ir = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.ToTensor(),
            normalizer,
            transforms.RandomErasing(p=0.5),
        ])
    else:
        transform_train_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomGrayscale(p=0.5),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
        transforms.RandomErasing(p=0.5),
        ])
        transform_train_ir = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor(),
            normalizer,
            transforms.RandomErasing(p=0.5),
        ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
    ])

    batch = args.stage3_ims_per_batch
    epochs = args.stage3_epochs
    num_classes_rgb = model.num_classes_rgb
    num_classes_ir = model.num_classes_ir

    model.eval()
    img2text.train()

    model.to(device)
    img2text.to(device)


    scaler = amp.GradScaler()
    # losses_i2t_rgb2rgb = AverageMeter()
    # losses_i2t_rgb2ir = AverageMeter()
    # losses_i2t_ir2rgb = AverageMeter()
    # losses_i2t_ir2ir = AverageMeter()
    losses_rgb = AverageMeter()
    losses_ir = AverageMeter()

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_img.to(device)
    loss_txt.to(device)

    end = end = time.time()
    if args.dataset == 'sysu':
        trainset = SYSUData_Stage2(args.data_path, transform_train_rgb, transform_train_ir)
    else:
        trainset = RegDBData_Stage2(args.data_path, args.trial, transform_train_rgb, transform_train_ir)
    print("New Dataset Information---- ")
    print("  ----------------------------")
    print("  subset   | # ids | # images")
    print("  ----------------------------")
    print("  visible  | {:5d} | {:8d}".format(len(np.unique(trainset.train_color_label)),
                                              len(trainset.train_color_image)))
    print("  thermal  | {:5d} | {:8d}".format(len(np.unique(trainset.train_thermal_label)),
                                              len(trainset.train_thermal_image)))
    print("  ----------------------------")
    print("Data loading time:\t {:.3f}".format(time.time() - end))

    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    sampler = IdentitySampler_nosk(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos,
                                   args.num_instances, args.batch_size)

    trainset.cIndex = sampler.index1
    trainset.tIndex = sampler.index2

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size * args.num_instances, sampler=sampler,
                                  num_workers=args.workers,
                                  drop_last=True)

    num_batches_per_epoch = len(trainloader)

    optimizer_3stage = make_optimizer_3stage(args, img2text)
    scheduler_3stage = cosine_lr(optimizer_3stage, args.lr,
                                 args.stage3_warmup_epoch * num_batches_per_epoch,
                                 args.stage3_maxepochs * num_batches_per_epoch)

    for epoch in range(1, epochs + 1):
        # losses_i2t_ir2ir.reset()
        # losses_i2t_ir2rgb.reset()
        # losses_i2t_rgb2ir.reset()
        # losses_i2t_rgb2rgb.reset()

        losses_rgb.reset()
        losses_ir.reset()


        for n_iter, (img1, img2, label1, label2) in enumerate(trainloader):
            img1 = img1.to(device)
            img2 = img2.to(device)

            label1 = label1.to(device)
            label2 = label2.to(device)

            step = (epoch - 1) * num_batches_per_epoch + n_iter
            scheduler_3stage(step)
            optimizer_3stage.zero_grad()

            with autocast():
                loss_rgb = get_loss_img2text(model,img2text,img1,loss_img,loss_txt,clip_model,1,"A person photo of")
                loss_ir = get_loss_img2text(model,img2text,img2,loss_img,loss_txt,clip_model,2,"A person photo of")

                loss = (loss_rgb + loss_ir) / 2
                scaler.scale(loss).backward()
                scaler.step(optimizer_3stage)
            scaler.update()


            losses_rgb.update(loss_rgb.item())
            losses_ir.update(loss_ir.item())

            if n_iter % args.print_freq == 0:
                print(
                    "Epoch[{}] Iteration[{}/{}] Loss_rgb_ir: ({:.6f}) ({:.6f}) Base Lr: {:.2e}".format(epoch, n_iter + 1,
                                                                                    num_batches_per_epoch,
                                                                                    losses_rgb.avg,
                                                                                    losses_ir.avg,
                                                                                    optimizer_3stage.param_groups[0]['lr'])
                )
        if epoch % args.eval_step == 0 or (epoch == args.stage3_maxepochs):
            torch.save(img2text.state_dict(),
                       os.path.join(args.model_path, args.logs_file + '_img2text_{}.pth'.format(epoch)))
