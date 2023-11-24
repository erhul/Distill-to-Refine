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
from data_list import ImageList, ImageList_idx, ImageList_noisy_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import collections.abc as container_abcs
from torch._six import string_classes
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.infonce import InstanceLoss
from PIL import ImageFilter


def mixup(input, alpha=1.0):
    bs = input.size(0)
    randind = torch.randperm(bs).to(input.device)
    # beta = torch.distributions.beta.Beta(alpha, alpha)
    # lam = beta.sample([bs]).to(input.device)
    import numpy as np
    lam = np.random.beta(alpha, alpha)
    lam = torch.ones_like(randind).float() * lam
    lam = torch.max(lam, 1. - lam)
    lam_expanded = lam.view([-1] + [1] * (input.dim() - 1))
    input = lam_expanded * input + (1. - lam_expanded) * input[randind]
    return input, randind, lam.unsqueeze(1)


def extract_features(netF, netB, netC, netP, loader):
    features = torch.zeros(len(loader.dataset), args.feat_dim).cuda()
    all_labels = torch.zeros(len(loader.dataset)).cuda()
    cluster_labels = torch.zeros(len(loader.dataset), args.num_cluster).cuda()

    netF.eval()
    netB.eval()
    netC.eval()
    netP.eval()

    local_features = []
    local_labels = []
    local_cluster_labels = []
    with torch.no_grad():
        iter_loader = iter(loader)
        for i in range(len(loader)):
            data = iter_loader.next()
            images, labels = convert_to_cuda(data)
            # images = data[0]
            # labels = data[1]
            # images = images.cuda()
            # labels = labels.cuda()
            local_labels.append(labels)

            x = netF(images)
            local_cluster_labels.append(F.softmax(netC(netB(x)), dim=1))
            local_features.append(F.normalize(netP(x), dim=1))
    local_features = torch.cat(local_features, dim=0)
    local_labels = torch.cat(local_labels, dim=0)
    local_cluster_labels = torch.cat(local_cluster_labels, dim=0)

    indices = torch.Tensor(list(iter(loader.sampler))).long().cuda()

    features.index_add_(0, indices, local_features)
    all_labels.index_add_(0, indices, local_labels.float())
    cluster_labels.index_add_(0, indices, local_cluster_labels.float())

    labels = all_labels.long()
    return features, cluster_labels, labels

def hist(assignments, is_clean, labels, n_iter, args, sample_type='context_assignments_hist'):
    fig, ax = plt.subplots()
    ax.hist(assignments[is_clean, labels[is_clean].long()].cpu().numpy(), label='clean', bins=100, alpha=0.5)
    ax.hist(assignments[~is_clean, labels[~is_clean].long()].cpu().numpy(), label='noisy', bins=100, alpha=0.5)
    ax.legend()
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    fname = f'{str(n_iter).zfill(5)}-{sample_type}.png'
    img.save(osp.join(args.output_dir, fname))
    plt.close()

def psedo_labeling(netF, netB, netC, netP, target_te_loader, test_loader, labels, prototypes, context_assignments,
                   confidences, n_iter, args):
    print('Generating the psedo-labels')
    features, cluster_labels, _ = extract_features(netF, netB, netC, netP, target_te_loader)
    confidence, context_assignments = correct_labels(cluster_labels, labels, features,
                                                     prototypes, context_assignments, confidences, args)
    evaluate(netF, netB, netC, netP, test_loader, features, confidence, cluster_labels, labels, context_assignments,
             n_iter, args)

def evaluate(netF, netB, netC, netP, loader, features, confidence, cluster_labels, labels, context_assignments, n_iter, args):
    # opt = self.opt
    clean_labels = torch.Tensor(loader.dataset.targets).cuda().long()

    is_clean = clean_labels.cpu().numpy() == labels.cpu().numpy()
    # hist(context_assignments, is_clean, labels, n_iter, args)
    train_acc = (torch.argmax(cluster_labels, dim=1) == clean_labels).float().mean()
    test_features, test_cluster_labels, test_labels = extract_features(netF, netB, netC, netP, loader)
    test_acc = (test_labels == torch.argmax(test_cluster_labels, dim=1)).float().mean()

    from utils.knn_monitor import knn_predict
    knn_labels = knn_predict(test_features, features, clean_labels,
                             classes=args.num_cluster, knn_k=200, knn_t=0.1)[:, 0]
    # a = torch.argmax(test_cluster_labels, dim=1)
    # print(str(torch.unique(torch.argmax(test_cluster_labels, dim=1), return_counts=True)))

    knn_acc = (test_labels == knn_labels).float().mean()

    estimated_noise_ratio = (confidence > 0.5).float().mean().item()
    args.scale1 = estimated_noise_ratio
    args.scale2 = estimated_noise_ratio

    noise_accuracy = ((confidence > 0.5) == (clean_labels == labels)).float().mean()
    from sklearn.metrics import roc_auc_score
    context_noise_auc = roc_auc_score(is_clean, confidence.cpu().numpy())
    print(
        'n_iter: {}, estimated_noise_ratio: {}, noise_accuracy:{:.2f}, context_noise_auc:{:.2f}, train_acc:{:.4f}, test_acc:{:.4f}, knn_acc:{:.4f}'
        .format(n_iter, estimated_noise_ratio, noise_accuracy.item(), context_noise_auc, train_acc.item(),
                test_acc.item(), knn_acc.item()))


def correct_labels(cluster_labels, labels, features, prototypes, context_assignment, confidences, args):
    centers = F.normalize(cluster_labels.T.mm(features), dim=1)
    context_assignments_logits = features.mm(centers.T) / args.temp
    context_assignments = F.softmax(context_assignments_logits, dim=1)
    losses = - context_assignments[torch.arange(labels.size(0)).long(), labels.long()]
    losses = losses.cpu().numpy()[:, np.newaxis]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    # labels = labels.cpu().numpy()

    from sklearn.mixture import GaussianMixture
    confidence = np.zeros((losses.shape[0],))
    gm = GaussianMixture(n_components=2, random_state=0).fit(losses)
    pdf = gm.predict_proba(losses)
    confidence = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]
    # confidence = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmax(gm.means_)]
    confidence = torch.from_numpy(confidence).float().cuda()

    prototypes.copy_(centers)
    context_assignment.copy_(context_assignments.float())
    confidences.copy_(confidence.float())

    return confidence, context_assignments


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

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def image_train(resize_size=256, crop_size=224, alexnet=False):
    train_transforms = []
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')

    weak_transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    train_transforms.append(weak_transform)

    for i in range(args.aug_nums):
        train_transform = transforms.Compose([
            # transforms.Resize((resize_size, resize_size)),
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
        train_transforms.append(train_transform)
    return train_transforms

# def image_train(resize_size=256, crop_size=224, alexnet=False):
#   if not alexnet:
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225])
#   else:
#     normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
#   return  transforms.Compose([
#         transforms.Resize((resize_size, resize_size)),
#         transforms.RandomCrop(crop_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize
#     ])

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
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    # dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dsets["target"] = ImageList_noisy_idx(txt_tar, transform_list=image_train())
    dsets["target_te"] = ImageList(txt_tar, transform=image_test())
    dsets["test"] = ImageList(txt_test, transform=image_test())
    return dsets


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


def generate_pseudo_labels(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_loader = iter(loader)
        for i in range(len(loader)):
            data = iter_loader.next()
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
        noise_ratio = 1 - torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        print(f'possible noise ratio {noise_ratio}')

    return predict

def convert_to_cuda(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data
        return data.cuda()
    elif isinstance(data, container_abcs.Mapping):
        return {key: convert_to_cuda(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(convert_to_cuda(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [convert_to_cuda(d) for d in data]
    else:
        return data

def comb(p1, p2, lam):
    return (1 - lam) * p1 + lam * p2

def CE(logits, targets):
    return - (targets * F.log_softmax(logits, dim=1)).sum(-1).mean()

def create_projector(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, in_dim),
                         nn.BatchNorm1d(in_dim),
                         nn.ReLU(inplace=True),
                         nn.Linear(in_dim, out_dim),
                         nn.BatchNorm1d(out_dim),
                         )

def create_classifier(in_dim, out_dim, dropout=0.25):
    return nn.Sequential(nn.Linear(in_dim, in_dim),
                         nn.BatchNorm1d(in_dim),
                         nn.ReLU(inplace=True),
                         nn.Dropout(p=dropout),
                         nn.Linear(in_dim, out_dim),
                         nn.BatchNorm1d(out_dim)
                         )

def train_target(args):
    dset_loaders = {}
    dsets = data_load(args)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.worker, drop_last=False)
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=args.batch_size * 3, shuffle=True,
                                      num_workers=args.worker, drop_last=False)

    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    # netC = create_classifier(netF.in_features, args.class_num).cuda()
    netP = create_projector(netF.in_features, args.feat_dim).cuda()

    modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netP.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // 10
    # interval_iter = len(dset_loaders["target"])
    iter_num = 0

    netF.eval()
    netB.eval()
    netC.eval()
    netP.eval()
    # torch.set_printoptions(threshold=3000)
    pseudo_labels = generate_pseudo_labels(dset_loaders['target_te'], netF, netB, netC)
    assert len(dsets["target"].targets) == len(pseudo_labels)
    dset_loaders["target"].dataset.targets = pseudo_labels.tolist()
    detect_class_num = len(np.unique(pseudo_labels[pseudo_labels >= 0]))
    num_samples = len(pseudo_labels)
    pseudo_labels = pseudo_labels.cuda()
    prototypes = torch.randn(args.num_cluster, args.feat_dim).cuda()
    prototypes = F.normalize(prototypes, dim=1)
    confidences = torch.zeros(num_samples).cuda()
    context_assignments = torch.zeros(num_samples, args.num_cluster)

    acc_s_te, _, pry, mean_ent = cal_acc(dset_loaders['test'], netF, netB, netC, False)
    log_str = 'Task: {}, Iter:{}/{}; Accuracy={:.2f}%, Ent={:.3f}'.format(args.name, iter_num, max_iter, acc_s_te, mean_ent)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    netF.train()
    netB.train()
    netC.train()
    netP.train()

    psedo_labeling(netF, netB, netC, netP, dset_loaders['target_te'], dset_loaders['test'],
                   pseudo_labels, prototypes, context_assignments, confidences, iter_num, args)

    old_pry = 0
    while iter_num < max_iter:
        try:
            inputs = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs = iter_test.next()

        # if inputs_test.size(0) == 1:
        #     continue

        # inputs_test = inputs_test.cuda()
        inputs, indices = convert_to_cuda(inputs)

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, power=0.75)

        images, _ = inputs
        im_w, im_q, im_k = images

        im_mix, mix_randind, mix_lam = mixup(torch.cat([im_w, im_q, im_k]))
        # compute query features
        x_q = netF(torch.cat([im_w, im_mix, im_q, im_k]))
        q = netP(x_q)
        q_logits = netC(netB(x_q))

        q = nn.functional.normalize(q, dim=1)
        q_w, q_mix, q1, q2 = q.split([im_w.size(0), im_mix.size(0), im_q.size(0), im_k.size(0)])
        w_logits, mix_logits, q_logits1, q_logits2 = q_logits.split(
            [im_w.size(0), im_mix.size(0), im_q.size(0), im_k.size(0)])

        contrastive_loss = InstanceLoss(args.temp)(q1, q2)

        with torch.no_grad():
            labels = pseudo_labels[indices].long()
            batch_confidences = confidences[indices].unsqueeze(1)

            targets_onehot_noise = F.one_hot(labels, args.num_cluster).float().cuda()
            # w_prob = F.softmax(w_logits.detach(), dim=1)
            q_prob1 = F.softmax(q_logits1.detach(), dim=1)
            q_prob2 = F.softmax(q_logits2.detach(), dim=1)

            targets_corrected1 = comb(q_prob2, targets_onehot_noise, batch_confidences)
            targets_corrected2 = comb(q_prob1, targets_onehot_noise, batch_confidences)
            a = (targets_corrected1 + targets_corrected2) * 0.5
            targets_mix_corrected = comb((q_prob1 + q_prob2) * 0.5, targets_onehot_noise, batch_confidences)
            targets_mix_corrected = targets_mix_corrected.repeat((q_mix.size(0) // q_logits1.size(0), 1))
            targets_mix_corrected = comb(targets_mix_corrected[mix_randind], targets_mix_corrected, mix_lam)

            # targets_mix_noise = targets_onehot_noise.repeat((q_mix.size(0) // q_logits1.size(0), 1))
            # targets_mix_noise = comb(targets_mix_noise[mix_randind], targets_mix_noise, mix_lam)

        align_logits = q_mix.mm(prototypes.T) / args.temp

        cls_loss1 = CE(q_logits1, targets_corrected1) + CE(q_logits2, targets_corrected2)
        cls_loss2 = CE(mix_logits, targets_mix_corrected)
        align_loss = CE(align_logits, targets_mix_corrected)

        pred_softmax = F.softmax(torch.cat([q_logits1, q_logits2, mix_logits]), dim=1)
        # pred_softmax = F.softmax(torch.cat([q_logits1, q_logits2]), dim=1)
        ent_loss = - (pred_softmax * F.log_softmax(torch.cat([q_logits1, q_logits2, mix_logits]), dim=1)).sum(dim=1).mean()
        # ent_loss = - (pred_softmax * F.log_softmax(torch.cat([q_logits1, q_logits2]), dim=1)).sum(dim=1).mean()
        prob_mean = pred_softmax.mean(dim=0)
        ne_loss = (prob_mean * prob_mean.log()).sum()

        optimizer.zero_grad()
        loss = contrastive_loss + \
               args.cls_loss_weight * (cls_loss1 + cls_loss2) + \
               args.ent_loss_weight * ent_loss + \
               args.ne_loss_weight * ne_loss + \
               args.align_loss_weight * align_loss
        # loss = contrastive_loss + \
        #        args.cls_loss_weight * cls_loss1 + \
        #        args.ent_loss_weight * ent_loss + \
        #        args.ne_loss_weight * ne_loss
        loss.backward()
        optimizer.step()
        print(
            'Task: {}, iter: {}/{}; contrastive_loss:{:.2f}, cls_loss1:{:.2f}, cls_loss2:{:.2f}, ent_loss:{:.2f}, ne_loss:{:.2f}, align_loss:{:.2f}'
            .format(args.name, iter_num, max_iter, contrastive_loss.item(), cls_loss1.item(), cls_loss2.item(),
                    ent_loss.item(), ne_loss.item(), align_loss.item()))

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            psedo_labeling(netF, netB, netC, netP, dset_loaders['target_te'], dset_loaders['test'],
                           pseudo_labels, prototypes, context_assignments, confidences, iter_num, args)
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_te, _, pry, mean_ent = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy={:.2f}%, Ent={:.3f}'.format(args.name, iter_num, max_iter,
                                                                                  acc_s_te, mean_ent)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()

            if torch.abs(pry - old_pry).sum() == 0:
                break
            else:
                old_pry = pry.clone()

    return netF, netB, netC


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

    parser.add_argument("--aug_nums", type=int, default=2)
    parser.add_argument('--temp', type=float, default=0.25, help='temp for contrastive loss')
    parser.add_argument('--feat_dim', type=int, default=256, help='projection feat_dim')
    parser.add_argument('--num_cluster', type=int, default=65, help='num_cluster')
    parser.add_argument('--cls_loss_weight', type=float, default=1.0, help='cls_loss_weight')
    parser.add_argument('--align_loss_weight', type=float, default=1.0, help='align_loss_weight')
    parser.add_argument('--ent_loss_weight', type=float, default=1.0, help='ent_loss_weight')
    parser.add_argument('--ne_loss_weight', type=float, default=1.0, help='ne_loss_weight')

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
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        args.output_dir = osp.join(args.output, args.net_src + '_' + args.net, str(args.seed), args.da, args.dset,
                                   names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(osp.join(args.output_dir, 'log_finetune.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()

        train_target(args)