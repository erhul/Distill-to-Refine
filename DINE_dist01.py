import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx_
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
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


def double_image_train(resize_size=256, crop_size=224, alexnet=False):
    train_transforms = []
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    for i in range(args.aug_nums):
        train_transform = transforms.Compose(
            [
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        )
        train_transforms.append(train_transform)
    return train_transforms


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    count = np.zeros(args.class_num)
    tr_txt = []
    te_txt = []
    for i in range(len(txt_src)):
        line = txt_src[i]
        reci = line.strip().split(' ')
        if count[int(reci[1])] < 3:
            count[int(reci[1])] += 1
            te_txt.append(line)
        else:
            tr_txt.append(line)

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["target"] = ImageList_idx_(txt_tar, transform_list=double_image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=False)
    dsets["target_te"] = ImageList(txt_tar, transform=image_test())
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs, shuffle=False,
                                           num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

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
            if netB is None:
                outputs = netC(netF(inputs))
            else:
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
    ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
    mean_ent = ent / np.log(all_label.size()[0])

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int), :]
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, mean_ent
    else:
        return accuracy * 100, mean_ent


def train_source_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()
    netC = network.feat_classifier_simpl(class_num=args.class_num, feat_dim=netF.in_features).cuda()

    param_group = []
    learning_rate = args.lr_src
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netF(inputs_source))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)(outputs_source,
                                                                                           labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, None, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netC.train()

    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netC


def test_target_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()
    netC = network.feat_classifier_simpl(class_num=args.class_num, feat_dim=netF.in_features).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders['test'], netF, None, netC, False)
    log_str = '\nTask: {}, Accuracy = {:.2f}%'.format(args.name, acc)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')


def copy_target_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()
    netC = network.feat_classifier_simpl(class_num=args.class_num, feat_dim=netF.in_features).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    source_model = nn.Sequential(netF, netC).cuda()
    source_model.eval()

    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net, pretrain=True).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    ent_best = 1.0
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // 10
    iter_num = 0

    model = nn.Sequential(netF, netB, netC).cuda()
    model.eval()

    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders["target_te"])
        for i in range(len(dset_loaders["target_te"])):
            data = iter_test.next()
            inputs, labels = data[0], data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = source_model(inputs)
            outputs = nn.Softmax(dim=1)(outputs)
            _, src_idx = torch.sort(outputs, 1, descending=True)
            if args.topk > 0:
                topk = np.min([args.topk, args.class_num])
                for i in range(outputs.size()[0]):
                    outputs[i, src_idx[i, topk:]] = (1.0 - outputs[i, src_idx[i, :topk]].sum()) / (
                                outputs.size()[1] - topk)

            if start_test:
                all_output = outputs.float()
                all_label = labels
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        print(accuracy)
        mem_P = all_output.detach()

    model.train()

    pre_data = None
    pre_out = None
    pre_idx = None
    while iter_num < max_iter:

        if args.ema < 1.0 and iter_num > 0 and iter_num % interval_iter == 0:
            model.eval()
            start_test = True
            with torch.no_grad():
                iter_test = iter(dset_loaders["target_te"])
                for i in range(len(dset_loaders["target_te"])):
                    data = iter_test.next()
                    inputs, labels = data[0], data[1]
                    labels = labels.cuda()
                    inputs = inputs.cuda()
                    outputs = model(inputs)
                    outputs = nn.Softmax(dim=1)(outputs)
                    if start_test:
                        all_output = outputs.float()
                        all_label = labels
                        start_test = False
                    else:
                        all_output = torch.cat((all_output, outputs.float()), 0)
                        all_label = torch.cat((all_label, labels), 0)
                mem_P = mem_P * args.ema + all_output.detach() * (1 - args.ema)
                _, predict = torch.max(mem_P, 1)
                accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
                print(accuracy)
            model.train()

        try:
            inputs_target, y, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, y, tar_idx = iter_target.next()

        if inputs_target.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, power=1.5)
        # decay_kl = (1 + 10 * iter_num / max_iter) ** (-0.75)
        # decay_kl = 1
        decay_kl = 1 if iter_num / interval_iter <= 1 else 0
        # decay_kl = [1 if math.floor(iter_num / interval_iter) % 2 == 0 else 0]
        inputs_target = inputs_target.cuda()
        with torch.no_grad():
            # outputs_target_by_source = mem_P[tar_idx, :]
            # _, src_idx = torch.sort(outputs_target_by_source, 1, descending=True)
            if iter_num == 1:
                outputs_target_by_source = mem_P[tar_idx, :]
                _, src_idx = torch.sort(outputs_target_by_source, 1, descending=True)
            else:
                pre_outputs_target_by_source = mem_P[pre_idx, :]
                outputs_target_by_source = mem_P[tar_idx, :]

        # outputs_target = model(inputs_target)
        outputs_target = model(inputs_target[:, 0, ...])
        # outputs_target = netF(inputs_target[:, 0, ...])
        # outputs_target = netB(outputs_target)
        # outputs_target = netC(outputs_target)

        if pre_data != None:
            out_pre = model(pre_data[:, 1, ...])
            current_batch_inputs = torch.cat((pre_data[:, 1, ...], inputs_target[:, 0, ...]), dim=0)
            current_batch_outputs = torch.cat((out_pre, outputs_target), dim=0)
            current_batch_outputs = torch.nn.Softmax(dim=1)(current_batch_outputs)
            kl_loss = nn.KLDivLoss(reduction='batchmean')(current_batch_outputs.log(),
                                                          torch.cat(
                                                              (pre_outputs_target_by_source, outputs_target_by_source),
                                                              dim=0)) * decay_kl
            optimizer.zero_grad()

            entropy_loss = torch.mean(loss.Entropy(current_batch_outputs))
            msoftmax = current_batch_outputs.mean(dim=0)
            gentropy_loss = torch.sum(- msoftmax * torch.log(msoftmax + 1e-5))
            mi_loss = entropy_loss - gentropy_loss

            decay_dml = min(1.0, iter_num / interval_iter)
            # decay_dml = 1 if iter_num / interval_iter > 1 else 0
            # decay_dml = 1 if math.floor(iter_num / interval_iter) % 2 == 1 else 0
            dml_loss = (
                    F.kl_div(
                        F.log_softmax(out_pre / args.T, dim=1),
                        F.softmax(pre_out.detach() / args.T, dim=1),  # detach
                        reduction="batchmean",
                    ) * args.T * args.T * decay_dml
            )
            total_loss = kl_loss + mi_loss + args.dml_loss_factor * dml_loss

            total_loss.backward()

            # if args.mix > 0:
            #     alpha = 0.3
            #     lam = np.random.beta(alpha, alpha)
            #     index = torch.randperm(current_batch_inputs.size()[0]).cuda()
            #     mixed_input = lam * current_batch_inputs + (1 - lam) * current_batch_inputs[index, :]
            #     mixed_output = (lam * current_batch_outputs + (1 - lam) * current_batch_outputs[index, :]).detach()
            #
            #     update_batch_stats(model, False)
            #     outputs_target_m = model(mixed_input)
            #     update_batch_stats(model, True)
            #     outputs_target_m = torch.nn.Softmax(dim=1)(outputs_target_m)
            #     classifier_loss = args.mix * nn.KLDivLoss(reduction='batchmean')(outputs_target_m.log(), mixed_output)
            #     classifier_loss.backward()
            optimizer.step()
        else:
            outputs_target = torch.nn.Softmax(dim=1)(outputs_target)
            kl_loss = nn.KLDivLoss(reduction='batchmean')(outputs_target.log(), outputs_target_by_source) * decay_kl
            optimizer.zero_grad()

            entropy_loss = torch.mean(loss.Entropy(outputs_target))
            msoftmax = outputs_target.mean(dim=0)
            gentropy_loss = torch.sum(- msoftmax * torch.log(msoftmax + 1e-5))
            mi_loss = entropy_loss - gentropy_loss
            total_loss = kl_loss + mi_loss

            total_loss.backward()

            # if args.mix > 0:
            #     alpha = 0.3
            #     lam = np.random.beta(alpha, alpha)
            #     current_batch_inputs = inputs_target[:, 0, ...]
            #     index = torch.randperm(current_batch_inputs.size()[0]).cuda()
            #     mixed_input = lam * current_batch_inputs + (1 - lam) * current_batch_inputs[index, :]
            #     mixed_output = (lam * outputs_target + (1 - lam) * outputs_target[index, :]).detach()
            #
            #     update_batch_stats(model, False)
            #     outputs_target_m = model(mixed_input)
            #     update_batch_stats(model, True)
            #     outputs_target_m = torch.nn.Softmax(dim=1)(outputs_target_m)
            #     classifier_loss = args.mix * nn.KLDivLoss(reduction='batchmean')(outputs_target_m.log(), mixed_output)
            #     classifier_loss.backward()
            optimizer.step()

        pre_data = inputs_target
        pre_out = outputs_target
        pre_idx = tar_idx

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            acc_s_te, mean_ent = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%, Ent = {:.4f}'.format(args.name, iter_num, max_iter,
                                                                                      acc_s_te, mean_ent)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            model.train()

    torch.save(netF.state_dict(), osp.join(args.output_dir, "source_F.pt"))
    torch.save(netB.state_dict(), osp.join(args.output_dir, "source_B.pt"))
    torch.save(netC.state_dict(), osp.join(args.output_dir, "source_C.pt"))


def update_batch_stats(model, flag):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.update_batch_stats = flag


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
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'image-clef', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50',
                        help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--lr_src', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net_src', type=str, default='resnet50',
                        help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--output_src', type=str, default='san')

    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--topk', type=int, default=1)

    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--ema', type=float, default=0.6)
    parser.add_argument('--mix', type=float, default=1.0)

    # leh
    parser.add_argument("--aug_nums", type=int, default=2)
    parser.add_argument("--T", type=float, default=3.0)
    parser.add_argument("--dml_loss_factor", type=float, default=1.0, help="DML loss weight factor")

    args = parser.parse_args()
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    args.output_dir_src = osp.join(args.output_src, args.net_src, str(args.seed), 'uda', args.dset,
                                   names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        # os.mkdir(args.output_dir_src)
        os.makedirs(args.output_dir_src)

    if not args.distill:
        print(args.output_dir_src + '/source_F.pt')
        args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_source_simp(args)

        args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
        for i in range(len(names)):
            if i == args.s:
                continue
            args.t = i
            args.name = names[args.s][0].upper() + names[args.t][0].upper()
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

            test_target_simp(args)

    if args.distill:
        for i in range(len(names)):
            if i == args.s:
                continue
            args.t = i
            args.name = names[args.s][0].upper() + names[args.t][0].upper()

            args.output_dir = osp.join(args.output, args.net_src + '_' + args.net, str(args.seed), args.da, args.dset,
                                       names[args.s][0].upper() + names[args.t][0].upper())
            if not osp.exists(args.output_dir):
                os.system('mkdir -p ' + args.output_dir)
            if not osp.exists(args.output_dir):
                # os.mkdir(args.output_dir)
                os.makedirs(args.output_dir)

            args.out_file = open(osp.join(args.output_dir, 'log_tar.txt'), 'w')
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

            # test_target_simp(args)
            copy_target_simp(args)