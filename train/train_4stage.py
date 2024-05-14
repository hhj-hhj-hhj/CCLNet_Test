import logging
import os
import torch
import torch.nn as nn
from util.utils import AverageMeter
# from util.loss. import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from util.loss.supcontrast import SupConLoss
from model.img2text import get_text_features, get_loss_img2text, get_loss
from model.make_model_clip import load_clip_to_cpu

def do_train_stage4(cfg,
                    model,
                    img2text,
                    center_criterion,
                    train_loader_stage4,
                    val_loader,
                    optimizer,
                    optimizer_center,
                    scheduler,
                    loss_fn,
                    num_query, local_rank):
    log_period = cfg.SOLVER.STAGE4.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE4.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE4.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE4.MAX_EPOCHS

    logger = logging.getLogger("transreid_VI.train")
    logger.info('start training stage4')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        img2text.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            img2text = nn.DataParallel(img2text)
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    loss_rgb = nn.CrossEntropyLoss()
    loss_ir = nn.CrossEntropyLoss()

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()


    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        # evaluator.reset()

        scheduler.step()

        model.train()
        img2text.eval()


        clip_model = load_clip_to_cpu(model.model_name, model.h_resolution, model.w_resolution,
                                      model.vision_stride_size)
        clip_model.to("cuda")

        for n_iter, (img_rgb, img_ir, label_rgb, label_ir) in enumerate(train_loader_stage4):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img_rgb = img_rgb.to(device)
            img_ir = img_ir.to(device)
            label_rgb = label_rgb.to(device, dtype=torch.int64)
            label_ir = label_ir.to(device, dtype=torch.int64)

            with amp.autocast(enabled=True):
                score_rgb, feat_rgb, image_features_rgb = model(x=img_rgb, label=label_rgb)
                score_ir, feat_ir, image_features_ir = model(x=img_ir, label=label_ir)
                with torch.no_grad():
                    token_features_rgb = img2text(image_features_rgb)
                    token_features_ir = img2text(image_features_ir)

                    text_features_rgb = get_text_features(token_features_rgb, clip_model, clip_model.dtype, "An infrared photo of")
                    text_features_ir = get_text_features(token_features_ir, clip_model, clip_model.dtype, "A visible photo of")

                image_features_rgb = image_features_rgb / image_features_rgb.norm(dim=-1, keepdim=True)
                image_features_ir = image_features_ir / image_features_ir.norm(dim=-1, keepdim=True)

                text_features_rgb = text_features_rgb / text_features_rgb.norm(dim=-1, keepdim=True)
                text_features_ir = text_features_ir / text_features_ir.norm(dim=-1, keepdim=True)


                # logit_scale = clip_model.logit_scale.exp()
                # logit_scale = logit_scale.mean()

                ground_truth_rgb = torch.arange(len(image_features_rgb)).long()
                ground_truth_rgb = ground_truth_rgb.cuda(device, non_blocking=True)

                ground_truth_ir = torch.arange(len(image_features_ir)).long()
                ground_truth_ir = ground_truth_ir.cuda(device, non_blocking=True)

                # logits_rgb2ir = logit_scale * image_features_rgb @ text_features_ir.t()
                # logits_ir2rgb = logit_scale * image_features_ir @ text_features_rgb.t()

                logits_rgb2ir = image_features_rgb @ text_features_ir.t()
                logits_ir2rgb = image_features_ir @ text_features_rgb.t()

                loss_ir2rgb = loss_rgb(logits_ir2rgb, ground_truth_rgb)
                loss_rgb2ir = loss_ir(logits_rgb2ir, ground_truth_ir)

                loss = (loss_ir2rgb + loss_rgb2ir) / 2

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc_rgb2ir = (logits_rgb2ir.max(1)[1] == label_ir).float().mean()
            acc_ir2rgb = (logits_ir2rgb.max(1)[1] == label_rgb).float().mean()

            acc = (acc_rgb2ir + acc_ir2rgb) / 2

            loss_meter.update(loss.item(), img_rgb.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage4),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))



        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader_stage4.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}_VI_stage4.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}_VI_stage4.pth'.format(epoch)))
