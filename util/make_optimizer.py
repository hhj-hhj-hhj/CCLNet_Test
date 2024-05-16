import torch
from torch import optim

def make_optimizer_1stage(args, model):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "prompt_learner" in key:
            lr = args.stage1_baselr
            weight_decay = args.stage1_weight_decay
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

    optimizer = getattr(torch.optim, 'Adam')(params)
    return optimizer


def make_optimizer_2stage(args, model_net):
    params = []
    keys = []
    for key, value in model_net.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = args.stage2_baselr
        weight_decay = args.stage2_weight_decay
        if "bias" in key:
            lr = args.stage2_baselr * args.stage2_bias_lr_factor
            weight_decay = args.stage2_weight_decay_bias

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    optimizer_net = getattr(torch.optim, 'Adam')(params)
    return optimizer_net

def make_optimizer_4stage(args, model_net):
    params = []
    keys = []
    for key, value in model_net.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = args.stage4_baselr
        weight_decay = args.stage4_weight_decay
        if "bias" in key:
            lr = args.stage4_baselr * args.stage4_bias_lr_factor
            weight_decay = args.stage4_weight_decay_bias

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    optimizer_net = getattr(torch.optim, 'Adam')(params)
    return optimizer_net

def make_optimizer_2stage_later(args, model_net):
    params = []
    keys = []
    for key, value in model_net.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = args.stage2_baselr * args.stage2_laterlr_factor
        weight_decay = args.stage2_weight_decay
        if "bias" in key:
            lr = args.stage2_baselr * args.stage2_bias_lr_factor
            weight_decay = args.stage2_weight_decay_bias

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    optimizer_net = getattr(torch.optim, 'Adam')(params)
    return optimizer_net


def make_optimizer_3stage(args, img2text):
    exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n: not exclude(n)
    named_parameters = list(img2text.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    return optimizer