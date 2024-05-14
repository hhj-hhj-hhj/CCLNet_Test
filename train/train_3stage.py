import logging
import os
import time
import torch
import torch.nn as nn
from util.utils import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from util.loss.supcontrast import SupConLoss
from model.img2text import get_text_features, get_loss_img2text, get_loss
from model.make_model_clip import load_clip_to_cpu


def do_train_stage3(cfg,
                    model,
                    img2text,
                    center_criterion,
                    train_loader_stage2_all,
                    optimizer,
                    # optimizer_center,
                    scheduler,
                    loss_fn,
                    local_rank):
    log_period = cfg.SOLVER.STAGE3.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE3.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE3.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    model.eval()
    img2text.train()
    epochs = cfg.SOLVER.STAGE3.MAX_EPOCHS

    logger = logging.getLogger("transreid_VI.train")
    logger.info('start training stage3')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        img2text.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            img2text = nn.DataParallel(img2text)
        #     num_classes = model.module.num_classes
        # else:
        #     num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img.to(device)
    loss_txt.to(device)

    clip_model = load_clip_to_cpu(model.model_name, model.h_resolution, model.w_resolution, model.vision_stride_size)
    clip_model.to("cuda")

    # train
    from datetime import timedelta
    all_start_time = time.monotonic()

    batch = cfg.SOLVER.STAGE3.IMS_PER_BATCH

    num_batches_per_epoch = len(train_loader_stage2_all)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        # evaluator.reset()

        # scheduler.step()

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2_all):
            # img = img.to(device)
            # target = vid.to(device)
            # target_cam = target_cam.to(device)
            # target_view = target_view.to(device)

            # data_time = time.time() - start_time

            step = epoch * num_batches_per_epoch + n_iter
            scheduler(step)
            optimizer.zero_grad()

            img = img.cuda(device, non_blocking=True)

            with amp.autocast(enabled=True):
                total_loss = get_loss_img2text(model, img2text, img, loss_img, loss_txt, clip_model)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

            if (n_iter + 1) % log_period == 0:
                loss_meter.update(total_loss.item())
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.6f} Base Lr: {:.2e}".format(epoch, n_iter + 1,
                                                                                    num_batches_per_epoch,
                                                                                    total_loss.item(),
                                                                                    optimizer.param_groups[0]['lr'])
                )
        end_time = time.time()
        time_per_batch = (end_time - start_time) / num_batches_per_epoch

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader_stage2_all.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(img2text.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_img2text_{}_VI.pth'.format(epoch)))
            else:
                torch.save(img2text.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_img2text_{}_VI.pth'.format(epoch)))

            # batch_time = time.time() - start_time
            #
            # timestep = epoch * num_batches_per_epoch + n_iter
            # log_data = {
            #     "loss": total_loss.item(),
            #     "data_time": data_time,
            #     "batch_time": batch_time,
            #     "lr": optimizer.param_groups[0]["lr"]
            # }





