import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.manifold import TSNE

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (11 + gamma * iter_num / max_iter) ** (-power)
    # decay = (1 + gamma) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs, shuffle=False, num_workers=args.worker,
                                      drop_last=False)
    dsets["target_te"] = ImageList(txt_tar, transform=image_test())
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item() / np.log(all_label.size()[0])

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int), :]
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, predict, mean_ent
    else:
        return accuracy * 100, mean_ent, predict, mean_ent


def draw_confusion_matrix(label_true, label_pred, label_name, normlize, title="Confusion Matrix", pdf_save_path=None, dpi=300):
    cm = confusion_matrix(label_true, label_pred)
    if normlize:
        row_sums = np.sum(cm, axis=1)
        cm = cm / row_sums[:, np.newaxis]

    plt.imshow(cm, cmap='Oranges')
    # plt.axis('off')
    plt.tick_params(left=False, pad=0.01, gridOn=False)
    plt.tick_params(bottom=False, pad=0.01, gridOn=False)

    # plt.title(title)
    # plt.rcParams['font.sans-serif'] = ['KaiTi']
    # plt.xlabel("Predicted label", fontsize='10', verticalalignment='baseline',)
    plt.xlabel("Predicted label", fontdict={'family': 'Times New Roman', 'size': 10})
    # plt.ylabel("Truth label", fontsize='10', verticalalignment='baseline')
    plt.ylabel("Truth label", fontdict={'family': 'Times New Roman', 'size': 10})
    plt.yticks(range(label_name.__len__()), label_name, rotation=45, fontproperties='Times New Roman', size=8)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45, fontproperties='Times New Roman', size=8)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.xticks([])
    # plt.axis('off')

    plt.tight_layout()

    cb = plt.colorbar()
    plt.clim(0, 1)
    cb.outline.set_visible(False)

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)
            value = float(format('%.2f' % cm[j, i]))
            if value < 0.01:
                plt.text(i, j, '', fontsize='5.0', style="normal", weight="normal",
                         verticalalignment='center', horizontalalignment='center', color=color)
            else:
                plt.text(i, j, value, fontsize='5.0', style="normal", weight="normal",
                         verticalalignment='center', horizontalalignment='center', color=color)

    minor_locator = AutoMinorLocator(2)
    plt.gca().yaxis.set_minor_locator(minor_locator)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', c='white', linewidth=0.75)

    plt.tick_params(left=False, pad=0.01, gridOn=False)
    plt.tick_params(bottom=False, pad=0.01, gridOn=False)
    # plt.grid(c='white')
    # plt.show()
    if not pdf_save_path is None:
        # plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi, pad_inches=0.0)


def draw_target_cm(args):
    dset_loaders = data_load(args)
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    print('testing......')
    acc_s_te, acc_list, pry, mean_ent = cal_acc(dset_loaders['test'], netF, netB, netC, True)
    log_str = 'Task: {}, Accuracy = {:.2f}%'.format(args.name, acc_s_te,) + '\n' + acc_list
    print(log_str + '\n')

    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders['test'])
        for i in range(len(iter_test)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            feature = netB(netF(inputs))
            output = netC(feature)
            if start_test:
                all_feature = feature.float().cpu()
                all_output = output.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_feature = torch.cat((all_feature, feature.float().cpu()), 0)
                all_output = torch.cat((all_output, output.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)

        # print('drawing t-SNE......')
        # feature_bank = all_feature.numpy()
        # label_bank = all_label.numpy()
        # tsne = TSNE(n_components=2, random_state=33)
        # output = tsne.fit_transform(feature_bank)
        # # fig, ax = plt.subplots(figsize=(10, 10))
        # # ax.spines['top'].set_visible(False)
        # # ax.spines['right'].set_visible(False)
        # # ax.spines['bottom'].set_visible(False)
        # # ax.spines['left'].set_visible(False)
        # plt.subplots(figsize=(10, 10))
        # hex = ["#1D57FB", "#B332D6", "#D70303", "#1AD7F6", "#787D88", "#A0B050",
        #        "#FAA00E", "#A08530", "#429C3E", "#C2473E", "#E3764F", "#915001"]
        # for i in range(12):
        #     index = (label_bank == i)
        #     plt.scatter(output[index, 0], output[index, 1], s=5, c=hex[i], marker='o', cmap=plt.cm.Spectral)
        #
        # plt.legend(['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck'])
        # plt.xticks([])
        # plt.yticks([])
        # # plt.savefig('VISDA-C_t-SNE', bbox_inches='tight', dpi=300, pad_inches=0.0)
        # plt.savefig('VISDA-C_t-SNE', bbox_inches='tight', dpi=300)

        print('drawing confusion matrix......')
        labels_name = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
                       'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)
        predict = torch.squeeze(predict).float()

        draw_confusion_matrix(label_true=all_label,
                              label_pred=predict,
                              label_name=labels_name,
                              normlize=True,
                              title="Confusion Matrix",
                              pdf_save_path='VISDA-C_confusion_matrix',
                              dpi=300)
        return


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DINE')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'image-clef', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet18, resnet50, resnext50")
    parser.add_argument('--net_src', type=str, default='resnet50',
                        help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])

    args = parser.parse_args()
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = './data/'
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        # args.output_dir = osp.join(args.output, args.net_src + '_' + args.net, str(args.seed), args.da, args.dset,
        #                            names[args.s][0].upper() + names[args.t][0].upper(), '1')
        # args.output_dir = osp.join(args.output, args.net_src + '_' + args.net, str(args.seed), args.da, args.dset,
        #                            names[args.s][0].upper() + names[args.t][0].upper(), '10')
        # args.output_dir = osp.join(args.output, args.net_src + '_' + args.net, str(args.seed), args.da, args.dset,
        #                            names[args.s][0].upper() + names[args.t][0].upper(), '20')
        args.output_dir = osp.join(args.output, args.net_src + '_' + args.net, str(args.seed), args.da, args.dset,
                                   names[args.s][0].upper() + names[args.t][0].upper(), '30')

        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(osp.join(args.output_dir, 'log_finetune.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()

        draw_target_cm(args)