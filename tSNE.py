import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# torch.backends.cuda.max_split_size_mb = 2750

def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False
        # , pin_memory=True
    )
    return cluster_loader


def t_SNE(args,
        model,
        ):

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


    batch = args.stage2_ims_per_batch
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

    text_features_rgb = text_features_rgb.cpu().numpy()
    text_features_ir = text_features_ir.cpu().numpy()

    # new_rgb = [text_features_rgb[:100] for _ in range(3)]
    # new_rgb = np.concatenate(new_rgb, axis=0)
    # new_ir = [text_features_ir[:100] for _ in range(3)]
    # new_ir = np.concatenate(new_ir, axis=0)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne_rgb = tsne.fit_transform(text_features_rgb)
    X_tsne_ir = tsne.fit_transform(text_features_ir)

    # X_tsne_rgb = tsne.fit_transform(new_rgb)
    # X_tsne_ir = tsne.fit_transform(new_ir)

    # 将两种文本特征的tSNE结果画在一张图上
    # plt.figure(figsize=(10, 5))
    # plt.scatter(X_tsne_rgb[:, 0], X_tsne_rgb[:, 1], c='r', label='rgb', marker='*')
    # plt.scatter(X_tsne_ir[:, 0], X_tsne_ir[:, 1], c='b', label='ir')
    # plt.legend()
    # plt.show()

    # 创建一个颜色列表，长度与特征点数量相同
    colors = plt.cm.rainbow(np.linspace(0, 1, len(text_features_rgb)))

    plt.figure(figsize=(10, 5))

    # 对于rgb特征，使用五角星形状，颜色从颜色列表中获取
    plt.scatter(X_tsne_rgb[:, 0], X_tsne_rgb[:, 1], c=colors, label='rgb', marker='*')

    # 对于ir特征，使用默认的圆形形状，颜色从颜色列表中获取
    plt.scatter(X_tsne_ir[:, 0], X_tsne_ir[:, 1], c=colors, label='ir')

    plt.legend()
    plt.show()
    print('t-SNE finished!')