# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F

from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss


def make_loss(args, num_classes):

    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    print("label smooth on, numclasses:", num_classes)
    triplet = TripletLoss(args.margin)

    def loss_func(score, feat, i2tscore, target):

        # mask = target!= ignore_index
        # i2tscore = i2tscore[mask]
        # target = target[mask]

        # if mask.sum() == 0:
        #     return torch.tensor([0.0]).cuda()

        if isinstance(score, list):
            ID_LOSS = [xent(scor, target) for scor in score[0:]]
            ID_LOSS = sum(ID_LOSS)
        else:
            ID_LOSS = xent(score, target)
        if isinstance(feat, list):
            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
            TRI_LOSS = sum(TRI_LOSS)
        else:
            TRI_LOSS = triplet(feat, target)[0]

        I2TLOSS = xent(i2tscore, target)

        loss = args.i2t_loss_weight * I2TLOSS + args.id_loss_weight * ID_LOSS + args.triplet_loss_weight * TRI_LOSS
        return loss

    return loss_func


# def make_loss(num_classes, ignore_index=-1):
#
#     xent = CrossEntropyLabelSmooth(num_classes=num_classes)
#     print("label smooth on, numclasses:", num_classes)
#
#     def loss_func(i2tscore, target):
#
#         mask = target!= ignore_index
#         i2tscore = i2tscore[mask]
#         target = target[mask]
#
#         if mask.sum() == 0:
#             return torch.tensor([0.0]).cuda()
#
#         I2TLOSS = xent(i2tscore, target)
#
#         return I2TLOSS
#
#     return loss_func