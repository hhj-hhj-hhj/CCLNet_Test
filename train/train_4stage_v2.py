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
from util.loss.softmax_loss import CrossEntropyLabelSmooth



def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False
        # , pin_memory=True
    )
    return cluster_loader

def do_train_stage4_v2(args,
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

    batch = args.stage4_ims_per_batch
    num_classes_rgb = model.num_classes_rgb
    num_classes_ir = model.num_classes_ir
    i_ter_rgb = num_classes_rgb // batch
    i_ter_ir = num_classes_ir // batch
    left_rgb = num_classes_rgb-batch* (num_classes_rgb//batch)
    left_ir = num_classes_ir-batch* (num_classes_ir//batch)
    if left_rgb != 0 :
        i_ter_rgb = i_ter_rgb+1
    if left_ir != 0 :
        i_ter_ir = i_ter_ir+1
    text_features_rgb = []
    text_features_ir = []
    with torch.no_grad():
        for i in range(i_ter_rgb):
            if i+1 != i_ter_rgb:
                l_list_rgb = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list_rgb = torch.arange(i*batch, num_classes_rgb)
            # with amp.autocast(enabled=True):
            text_feature_rgb = model(get_text = True, label = l_list_rgb, modal=1)
            text_features_rgb.append(text_feature_rgb.cpu())
        text_features_rgb = torch.cat(text_features_rgb, 0).cuda()
    with torch.no_grad():
        for i in range(i_ter_ir):
            if i+1 != i_ter_ir:
                l_list_ir = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list_ir = torch.arange(i*batch, num_classes_ir)
            # with amp.autocast(enabled=True):
            text_feature_ir = model(get_text = True, label = l_list_ir, modal=2)
            text_features_ir.append(text_feature_ir.cpu())
        text_features_ir = torch.cat(text_features_ir, 0).cuda()

    del text_feature_rgb,text_feature_ir

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


    # loss_i2t = nn.CrossEntropyLoss()

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

    loss_fn_rgb = CrossEntropyLabelSmooth(num_classes=len(trainset.train_color_label))
    loss_fn_ir = CrossEntropyLabelSmooth(num_classes=len(trainset.train_thermal_label))


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

            label_rgb = label_rgb.to(device, dtype=torch.int64)
            label_ir = label_ir.to(device, dtype=torch.int64)

            with amp.autocast(enabled=True):

                # logit_scale = clip_model.logit_scale.exp()
                # logit_scale = logit_scale.mean()

                score_rgb, feat_rgb, image_features_rgb, score_ir, feat_ir, image_features_ir = model(x1=img_rgb, x2=img_ir, modal=0)

                with torch.no_grad():
                # for param in img2text.parameters():
                #     param.requires_grad = False
                    token_features_rgb = img2text(image_features_rgb)
                    token_features_ir = img2text(image_features_ir)

                text_features_rgb2rgb = get_text_features(token_features_rgb, clip_model, clip_model.dtype, "A visible photo of")
                text_features_rgb2ir = get_text_features(token_features_rgb, clip_model, clip_model.dtype, "An infrared photo of")
                text_features_ir2rgb = get_text_features(token_features_ir, clip_model, clip_model.dtype, "A visible photo of")
                text_features_ir2ir = get_text_features(token_features_ir, clip_model, clip_model.dtype, "An infrared photo of")

                # image_features_rgb = image_features_rgb / image_features_rgb.norm(dim=-1, keepdim=True)
                # image_features_ir = image_features_ir / image_features_ir.norm(dim=-1, keepdim=True)
                #
                # text_features_rgb2rgb = text_features_rgb2rgb / text_features_rgb2rgb.norm(dim=-1, keepdim=True)
                # text_features_rgb2ir = text_features_rgb2ir / text_features_rgb2ir.norm(dim=-1, keepdim=True)
                # text_features_ir2rgb = text_features_ir2rgb / text_features_ir2rgb.norm(dim=-1, keepdim=True)
                # text_features_ir2ir = text_features_ir2ir / text_features_ir2ir.norm(dim=-1, keepdim=True)


                # ground_truth = torch.arange(len(image_features_rgb)).long()
                # ground_truth = ground_truth.to(device)

                logits_rgb2rgb = text_features_rgb2rgb @ text_features_rgb.t()
                logits_ir2rgb = text_features_ir2rgb @ text_features_rgb.t()
                logits_rgb2ir = text_features_rgb2ir @ text_features_ir.t()
                logits_ir2ir = text_features_ir2ir @ text_features_ir.t()

                loss_rgb2rgb = loss_fn_rgb(logits_rgb2rgb, label_rgb)
                loss_rgb2ir = loss_fn_ir(logits_rgb2ir, label_ir)
                loss_ir2rgb = loss_fn_rgb(logits_ir2rgb, label_rgb)
                loss_ir2ir = loss_fn_ir(logits_ir2ir, label_ir)

                loss = (loss_rgb2rgb + loss_rgb2ir + loss_ir2rgb + loss_ir2ir) / 4

            print(f"loss_rgb2rgb: {loss_rgb2rgb.item()}, \nloss_rgb2ir: {loss_rgb2ir.item()}, \nloss_ir2rgb: {loss_ir2rgb.item()}, \nloss_ir2ir: {loss_ir2ir.item()}")
            print(f"loss: {loss.item()}")
            scaler.scale(loss).backward()
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
