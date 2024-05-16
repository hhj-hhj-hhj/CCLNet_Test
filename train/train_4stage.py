import os
from datetime import timedelta
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.cuda import amp
from util.utils import AverageMeter


from data.data_manager import process_query_sysu, process_gallery_sysu
from data.data_manager import process_test_regdb
from data.dataloader import SYSUData_Stage2, RegDBData_Stage2, IterLoader, TestData
from util.eval import tester
from util.utils import IdentitySampler_nosk_stage4, GenIdx
from model.img2text import get_loss_img2text, get_text_features



def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False
        # , pin_memory=True
    )
    return cluster_loader

def do_train_stage4(args,
                    model,
                    img2text,
                    clip_model,
                    optimizer,
                    scheduler):
    best_acc = 0
    device = 'cuda'
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

    epochs = args.stage4_maxepochs

    model.to(device)
    img2text.to(device)

    model.train()
    img2text.eval()

    scaler = amp.GradScaler()
    losses_i2t_rgb2rgb = AverageMeter()
    losses_i2t_rgb2ir = AverageMeter()
    losses_i2t_ir2rgb = AverageMeter()
    losses_i2t_ir2ir = AverageMeter()

    loss_i2t = nn.CrossEntropyLoss()

    end = time.time()
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

    sampler = IdentitySampler_nosk_stage4(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos,
                                   args.stage4_num_instances, args.stage4_batch_size)

    trainset.cIndex = sampler.index1
    trainset.tIndex = sampler.index2

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size * args.num_instances, sampler=sampler,
                                  num_workers=args.workers,
                                  drop_last=True)

    num_batches_per_epoch = len(trainloader)


    for epoch in range(1, epochs + 1):
        # end = time.time()
        losses_i2t_rgb2rgb.reset()
        losses_i2t_rgb2ir.reset()
        losses_i2t_ir2rgb.reset()
        losses_i2t_ir2ir.reset()

        scheduler.step()

        for n_iter, (img_rgb, img_ir, label_rgb, label_ir) in enumerate(trainloader):
            optimizer.zero_grad()

            img_rgb = img_rgb.to(device)
            img_ir = img_ir.to(device)

            with amp.autocast(enabled=True):

                # logit_scale = clip_model.logit_scale.exp()
                # logit_scale = logit_scale.mean()

                score_rgb, feat_rgb, image_features_rgb, score_ir, feat_ir, image_features_ir = model(x1=img_rgb, x2=img_ir, modal=0)

                with torch.no_grad():
                    token_features_rgb = img2text(image_features_rgb)
                    token_features_ir = img2text(image_features_ir)

                    text_features_rgb2rgb = get_text_features(token_features_rgb, clip_model, clip_model.dtype, "A visible photo of")
                    text_features_rgb2ir = get_text_features(token_features_rgb, clip_model, clip_model.dtype, "An infrared photo of")
                    text_features_ir2rgb = get_text_features(token_features_ir, clip_model, clip_model.dtype, "A visible photo of")
                    text_features_ir2ir = get_text_features(token_features_ir, clip_model, clip_model.dtype, "An infrared photo of")

                image_features_rgb = image_features_rgb / image_features_rgb.norm(dim=-1, keepdim=True)
                image_features_ir = image_features_ir / image_features_ir.norm(dim=-1, keepdim=True)

                text_features_rgb2rgb = text_features_rgb2rgb / text_features_rgb2rgb.norm(dim=-1, keepdim=True)
                text_features_rgb2ir = text_features_rgb2ir / text_features_rgb2ir.norm(dim=-1, keepdim=True)
                text_features_ir2rgb = text_features_ir2rgb / text_features_ir2rgb.norm(dim=-1, keepdim=True)
                text_features_ir2ir = text_features_ir2ir / text_features_ir2ir.norm(dim=-1, keepdim=True)


                ground_truth = torch.arange(len(image_features_rgb)).long()
                ground_truth = ground_truth.to(device)

                logits_rgb2rgb = image_features_rgb @ text_features_rgb2rgb.t()
                logits_ir2rgb = image_features_rgb @ text_features_ir2rgb.t()
                logits_rgb2ir = image_features_ir @ text_features_rgb2ir.t()
                logits_ir2ir = image_features_ir @ text_features_ir2ir.t()

                loss_rgb2rgb = loss_i2t(logits_rgb2rgb, ground_truth)
                loss_rgb2ir = loss_i2t(logits_rgb2ir, ground_truth)
                loss_ir2rgb = loss_i2t(logits_ir2rgb, ground_truth)
                loss_ir2ir = loss_i2t(logits_ir2ir, ground_truth)

                loss = (loss_rgb2rgb + loss_rgb2ir + loss_ir2rgb + loss_ir2ir) / 4


            scaler.scale(loss)
            scaler.step(optimizer)
            scaler.update()

            losses_i2t_rgb2rgb.update(loss_rgb2rgb.item())
            losses_i2t_rgb2ir.update(loss_rgb2ir.item())
            losses_i2t_ir2rgb.update(loss_ir2rgb.item())
            losses_i2t_ir2ir.update(loss_ir2ir.item())

            if n_iter % args.print_freq == 0:
                print("Epoch[{}] Iteration[{}/{}], Loss_rgb2rgb_rgb2ir_ir2rgb_ir2ir: ({:.3f}) ({:.3f}) ({:.3f}) ({:.3f}), Base Lr: {:.2e}"
                 .format(epoch, (n_iter + 1), len(trainloader), losses_i2t_rgb2rgb.avg, losses_i2t_rgb2ir.avg,
                         losses_i2t_ir2rgb.avg, losses_i2t_ir2ir.avg,scheduler.get_lr()[0]))

        if epoch % args.eval_step == 0 or (epoch == args.stage4_maxepochs):
            print("start test")
            # if True:
            if args.dataset == 'sysu':
                print('Test Epoch: {}'.format(epoch))
                test_mode = [1, 2]
                query_img, query_label, query_cam = process_query_sysu(args.data_path, mode=args.mode)
                queryset = TestData(query_img, query_label, transform=transform_test,
                                    img_size=(args.img_w, args.img_h))
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False,
                                               num_workers=args.workers)

                for trial in range(10):
                    # print('-------test trial {}-------'.format(trial))
                    gall_img, gall_label, gall_cam = process_gallery_sysu(args.data_path, mode=args.mode,
                                                                          trial=trial)
                    gallset = TestData(gall_img, gall_label, transform=transform_test,
                                       img_size=(args.img_w, args.img_h))
                    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False,
                                                  num_workers=args.workers)

                    cmc, mAP, mINP = tester(args, epoch, model, test_mode, gall_label, gall_loader, query_label,
                                            query_loader,
                                            feat_dim=3072,
                                            query_cam=query_cam, gall_cam=gall_cam)

                    if trial == 0:
                        all_cmc = cmc
                        all_mAP = mAP
                        all_mINP = mINP
                    else:
                        all_cmc = all_cmc + cmc
                        all_mAP = all_mAP + mAP
                        all_mINP = all_mINP + mINP

                cmc = all_cmc / 10
                mAP = all_mAP / 10
                mINP = all_mINP / 10
                print(
                    "Performance[ALL]: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
                        cmc[0], cmc[4],
                        cmc[9], cmc[19],
                        mAP, mINP))

                if cmc[0] > best_acc:
                    best_acc = cmc[0]
                    best_epoch = epoch
                    best_mAP = mAP
                    best_mINP = mINP
                    state = {
                        "state_dict": model.state_dict(),
                        "cmc": cmc,
                        "mAP": mAP,
                        "mINP": mINP,
                        "epoch": epoch,
                    }
                    torch.save(state, os.path.join(args.model_path, args.logs_file + "_stage4_V1.pth"))
                print("Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}".format(best_epoch, best_acc,
                                                                                            best_mAP, best_mINP))
            elif args.dataset == 'regdb':
                print('Test Epoch: {}'.format(epoch))

                query_img, query_label = process_test_regdb(img_dir=args.data_path, trial=args.trial,
                                                            modal='visible')
                gall_img, gall_label = process_test_regdb(img_dir=args.data_path, trial=args.trial, modal='thermal')

                test_mode = [2, 1]
                gallset = TestData(gall_img, gall_label, transform=transform_test,
                                   img_size=(args.img_w, args.img_h))
                queryset = TestData(query_img, query_label, transform=transform_test,
                                    img_size=(args.img_w, args.img_h))

                # testing data loader
                gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers)
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False,
                                               num_workers=args.workers)

                cmc, mAP, mINP = tester(args, epoch, model, test_mode, gall_label, gall_loader, query_label,
                                        query_loader,
                                        feat_dim=3072)

                print(
                    "Performance[ALL]: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
                        cmc[0], cmc[4],
                        cmc[9], cmc[19],
                        mAP, mINP))
                if cmc[0] > best_acc:
                    best_acc = cmc[0]
                    best_epoch = epoch
                    best_mAP = mAP
                    best_mINP = mINP
                    state = {
                        "state_dict": model.state_dict(),
                        "cmc": cmc,
                        "mAP": mAP,
                        "mINP": mINP,
                        "epoch": epoch,
                    }
                    torch.save(state, os.path.join(args.model_path, args.logs_file + "_stage4_regdb.pth"))
                print("Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}".format(best_epoch, best_acc,
                                                                                            best_mAP, best_mINP))
            else:
                print('please input correct dataset!!')

        torch.cuda.empty_cache()

    end_time = time.monotonic()
    print('Stage4 running time: ', timedelta(seconds=end_time - start_time))
