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
from data_list import ImageList, ImageList_idx, ImageList_aug_idx
import random
from loss import CrossEntropyLabelSmooth
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from PIL import ImageFilter


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p + 1e-5), dim=axis)


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
    return optimizer, decay


def lr_scheduler_(optimizer, decay):
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
        weak_transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        train_transforms.append(weak_transform)

    for i in range(args.aug_nums):
        strong_augmentation = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomResizedCrop(size=crop_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomSolarize(threshold=128, p=0.5),
            # transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        train_transforms.append(strong_augmentation)
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

    dsets["target"] = ImageList_aug_idx(txt_tar, transform_list=double_image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=False)
    dsets["target_te"] = ImageList_idx(txt_tar, transform=image_test())
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs, shuffle=False,
                                           num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=False,
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


def get_distances(X, Y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances


@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank):
    pred_probs = []
    pred_probs_all = []

    for feats in features.split(64):
        distances = get_distances(feats, features_bank)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)

    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    # First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard


def refine_predictions(features, banks):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pred_labels, probs, pred_labels_all, pred_labels_hard = soft_k_nearest_neighbors(features, feature_bank, probs_bank)

    return pred_labels, probs, pred_labels_all, pred_labels_hard


def eval_and_label_dataset(epoch, netF, netB, netC, target_te_loader):
    print("Evaluating Dataset!")
    logits, indices, gt_labels = [], [], []
    features = []

    start_test = True
    with torch.no_grad():
        iter_target_te = iter(target_te_loader)
        for i in range(len(target_te_loader)):
            data = iter_target_te.next()
            inputs = data[0].cuda()
            targets = data[1].cuda()
            idxs = data[2].cuda()

            feats = netB(netF(inputs))
            logits_cls = netC(feats)
            if start_test:
                features = feats.float()
                logits = logits_cls.float()
                gt_labels = targets.float()
                indices = idxs.float()
                start_test = False
            else:
                features = torch.cat((features, feats.float()), 0)
                gt_labels = torch.cat((gt_labels, targets.float()), 0)
                logits = torch.cat((logits, logits_cls.float()), 0)
                indices = torch.cat((indices, idxs.float()), 0)

        probs = F.softmax(logits, dim=1)
        rand_idxs = torch.randperm(len(features)).cuda()
        banks = {
            "features": features[rand_idxs][: 16384],
            "probs": probs[rand_idxs][: 16384],
            "ptr": 0,
        }
        # refine predicted labels
        pred_labels, pred_probs, _, _ = refine_predictions(features, banks)
        acc = 100. * accuracy_score(gt_labels.to('cpu'), pred_labels.to('cpu'))
        print("\n| Test epoch  #%d\t Accuracy: %.2f%%\n" % (epoch, acc))

        return acc, banks, gt_labels, pred_labels, pred_probs


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

    max_iter = args.max_epoch * len(dset_loaders["target"])
    dist_iter = args.dist_epoch * len(dset_loaders["target"])
    interval_iter = dist_iter // 10
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
        str = 'initial accurary = {:.2f}%'.format(accuracy * 100)
        print(str)
        mem_P = all_output.detach()

    model.train()

    pre_data = None
    pre_out = None
    pre_idx = None
    pred_labels = None
    pred_probs = None
    decay = None
    while iter_num < max_iter:
        e = int(iter_num / len(dset_loaders["target"]))

        if iter_num >= dist_iter and iter_num % len(dset_loaders["target"]) == 0:
            model.eval()
            _, _, _, pred_labels, pred_probs = eval_and_label_dataset(e, netF, netB, netC, dset_loaders["target_te"])
            model.train()
        try:
            inputs_target, y, tar_idx, inputs_target_aug1, inputs_target_aug2 = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, y, tar_idx, inputs_target_aug1, inputs_target_aug2 = iter_target.next()

        if inputs_target.size(0) == 1:
            continue

        iter_num += 1
        if iter_num <= dist_iter:
            _, decay = lr_scheduler(optimizer, iter_num=iter_num, max_iter=50 * len(dset_loaders["target"]), power=1.5)
        else:
            lr_scheduler_(optimizer, decay)

        inputs_target = inputs_target.cuda()
        tar_idx = tar_idx.cuda()
        y = y.cuda()
        inputs_target_aug1 = inputs_target_aug1.cuda()
        inputs_target_aug2 = inputs_target_aug2.cuda()

        with torch.no_grad():
            if iter_num == 1:
                outputs_target_by_source = mem_P[tar_idx, :]
            else:
                pre_outputs_target_by_source = mem_P[pre_idx, :]
                outputs_target_by_source = mem_P[tar_idx, :]

        if pre_data != None:
            if iter_num <= dist_iter:
                outputs_target = model(inputs_target[:, 0, ...])
                out_pre = model(pre_data[:, 1, ...])
                current_batch_outputs = torch.cat((out_pre, outputs_target), dim=0)
                current_batch_outputs = torch.nn.Softmax(dim=1)(current_batch_outputs)

                decay_kl = 1 if iter_num / interval_iter <= 1 else 0
                kl_loss = nn.KLDivLoss(reduction='batchmean')(current_batch_outputs.log(),
                                                              torch.cat((pre_outputs_target_by_source,
                                                                         outputs_target_by_source), dim=0)) * decay_kl
                optimizer.zero_grad()
                decay_dml = 1 if iter_num / interval_iter > 1 else (iter_num - 1) / interval_iter
                dml_loss = (
                        F.kl_div(
                            F.log_softmax(out_pre / args.T, dim=1),
                            F.softmax(pre_out.detach() / args.T, dim=1),  # detach
                            reduction="batchmean",
                        ) * args.T * args.T * decay_dml
                )

                entropy_loss = torch.mean(loss.Entropy(current_batch_outputs))
                msoftmax = current_batch_outputs.mean(dim=0)
                gentropy_loss = torch.sum(- msoftmax * torch.log(msoftmax + 1e-5))
                mi_loss = entropy_loss - gentropy_loss

                total_loss = kl_loss + dml_loss + mi_loss
                total_loss.backward()

                if args.mix > 0:
                    alpha = 0.3
                    lam = np.random.beta(alpha, alpha)
                    current_batch_inputs = torch.cat((pre_data[:, 1, ...], inputs_target[:, 0, ...]), dim=0)
                    index = torch.randperm(current_batch_inputs.size()[0]).cuda()
                    mixed_input = lam * current_batch_inputs + (1 - lam) * current_batch_inputs[index, :]
                    mixed_output = (lam * current_batch_outputs + (1 - lam) * current_batch_outputs[index, :]).detach()

                    update_batch_stats(model, False)
                    outputs_target_m = model(mixed_input)
                    update_batch_stats(model, True)
                    outputs_target_m = torch.nn.Softmax(dim=1)(outputs_target_m)
                    mix_loss = args.mix * nn.KLDivLoss(reduction='batchmean')(outputs_target_m.log(), mixed_output)
                    mix_loss.backward()
                optimizer.step()
            else:
                outputs_target = None
                logits_q = model(inputs_target_aug1)
                logits_k = model(inputs_target_aug2)
                current_batch_outputs = torch.cat((logits_q, logits_k), dim=0)
                current_batch_outputs = torch.nn.Softmax(dim=1)(current_batch_outputs)

                with torch.no_grad():
                    pseudo_labels_w = pred_labels[tar_idx]
                    probs_w = pred_probs[tar_idx]
                    # CE weights
                    max_entropy = torch.log2(torch.tensor(args.class_num))
                    w = entropy(probs_w)
                    w = w / max_entropy
                    w = torch.exp(-w).unsqueeze(1)

                    targets_onehot_noise = F.one_hot(pseudo_labels_w, args.class_num).float().cuda()
                    q_prob = F.softmax(logits_q.detach(), dim=1)
                    k_prob = F.softmax(logits_k.detach(), dim=1)

                    def comb(p1, p2, lam):
                        return (1 - lam) * p1 + lam * p2

                    targets_corrected1 = comb(k_prob, targets_onehot_noise, w)
                    targets_corrected2 = comb(q_prob, targets_onehot_noise, w)

                def CE(logits, targets):
                    return - (targets * F.log_softmax(logits, dim=1)).sum(-1).mean()

                cls_loss = CE(logits_q, targets_corrected1) + CE(logits_k, targets_corrected2)
                optimizer.zero_grad()

                entropy_loss = torch.mean(loss.Entropy(current_batch_outputs))
                msoftmax = current_batch_outputs.mean(dim=0)
                gentropy_loss = torch.sum(- msoftmax * torch.log(msoftmax + 1e-5))
                mi_loss = entropy_loss - gentropy_loss

                total_loss = cls_loss + mi_loss
                total_loss.backward()

                if args.mix > 0:
                    alpha = 0.3
                    lam = np.random.beta(alpha, alpha)
                    current_batch_inputs = torch.cat((inputs_target_aug1, inputs_target_aug2), dim=0)
                    index = torch.randperm(current_batch_inputs.size()[0]).cuda()
                    mixed_input = lam * current_batch_inputs + (1 - lam) * current_batch_inputs[index, :]
                    mixed_output = (lam * current_batch_outputs + (1 - lam) * current_batch_outputs[index, :]).detach()

                    update_batch_stats(model, False)
                    outputs_target_m = model(mixed_input)
                    update_batch_stats(model, True)
                    outputs_target_m = torch.nn.Softmax(dim=1)(outputs_target_m)
                    mix_loss = args.mix * nn.KLDivLoss(reduction='batchmean')(outputs_target_m.log(), mixed_output)
                    mix_loss.backward()
                optimizer.step()
        else:
            outputs_target = model(inputs_target[:, 0, ...])
            current_batch_outputs = torch.nn.Softmax(dim=1)(outputs_target)
            kl_loss = nn.KLDivLoss(reduction='batchmean')(current_batch_outputs.log(), outputs_target_by_source)
            optimizer.zero_grad()

            entropy_loss = torch.mean(loss.Entropy(current_batch_outputs))
            msoftmax = current_batch_outputs.mean(dim=0)
            gentropy_loss = torch.sum(- msoftmax * torch.log(msoftmax + 1e-5))
            mi_loss = entropy_loss - gentropy_loss

            total_loss = kl_loss + mi_loss
            total_loss.backward()

            if args.mix > 0:
                alpha = 0.3
                lam = np.random.beta(alpha, alpha)
                current_batch_inputs = inputs_target[:, 0, ...]
                index = torch.randperm(current_batch_inputs.size()[0]).cuda()
                mixed_input = lam * current_batch_inputs + (1 - lam) * current_batch_inputs[index, :]
                mixed_output = (lam * current_batch_outputs + (1 - lam) * current_batch_outputs[index, :]).detach()

                update_batch_stats(model, False)
                outputs_target_m = model(mixed_input)
                update_batch_stats(model, True)
                outputs_target_m = torch.nn.Softmax(dim=1)(outputs_target_m)
                mix_loss = args.mix * nn.KLDivLoss(reduction='batchmean')(outputs_target_m.log(), mixed_output)
                mix_loss.backward()
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
    parser.add_argument('--max_epoch', type=int, default=50, help="max iterations")
    parser.add_argument('--dist_epoch', type=int, default=30, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='VISDA-C',
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

    parser.add_argument("--aug_nums", type=int, default=2)
    parser.add_argument("--T", type=float, default=3.0)
    parser.add_argument("--dml_loss_factor", type=float, default=1.0, help="DML loss weight factor")
    parser.add_argument('--num_neighbors', default=10, type=int)

    args = parser.parse_args()
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10

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
            os.makedirs(args.output_dir)

        args.out_file = open(osp.join(args.output_dir, 'log_tar.txt'), 'w')
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        test_target_simp(args)
        copy_target_simp(args)