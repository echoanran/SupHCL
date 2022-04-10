#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
import os
import math

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad, gradcheck

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

from tools.utils import funcs, losses

from tensorboardX import SummaryWriter

import torchvision.models as models


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def smooth_one_hot(true_label,
                   smoothing=0.0,
                   hard_upper_bound=False,
                   upper_bound=None):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    num_class = true_label.size(-1)
    upper_bound = torch.tensor(upper_bound).to(true_label.device)

    with torch.no_grad():
        if not hard_upper_bound:
            true_dist = (1.0 - smoothing) * true_label.data.clone()
            true_dist += smoothing / (num_class - 1)
        else:
            true_dist = true_label.data.clone()
            true_dist = true_dist * upper_bound.repeat(true_label.shape[0], 1)

    return true_dist


class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition (Supervised Contrastive Loss)
    """
    def init_environment(self):

        super().init_environment()
        self.best_f1 = np.zeros(self.arg.model_args['num_class'])
        self.best_acc = np.zeros(self.arg.model_args['num_class'])
        self.best_aver_f1 = 0
        self.best_aver_acc = 0

        self.prototype = dict()
        self.prototype_update = dict()

        torch.manual_seed(self.arg.seed)
        torch.cuda.manual_seed_all(self.arg.seed)
        torch.cuda.manual_seed(self.arg.seed)
        np.random.seed(self.arg.seed)

    def load_data(self):

        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
            self.arg.test_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()

        dataset_train = Feeder(**self.arg.train_feeder_args)
        dataset_val = Feeder(**self.arg.test_feeder_args)

        self.sampler_train = torch.utils.data.RandomSampler(dataset_train)
        self.sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=dataset_train,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker *
                torchlight.ngpu(self.arg.device),
                drop_last=True)
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=dataset_val,
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker *
                torchlight.ngpu(self.arg.device),
                drop_last=False)

    def load_model(self):

        self.train_logger = SummaryWriter(log_dir=os.path.join(
            self.arg.work_dir, 'train'),
                                          comment='train')
        self.validation_logger = SummaryWriter(log_dir=os.path.join(
            self.arg.work_dir, 'validation'),
                                               comment='validation')

        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))

        update_dict = {}
        model_dict = self.model.state_dict()
        if self.arg.pretrain and self.arg.model_args['backbone'] in [
                'resnet18'
        ]:
            pretrained_dict = models.resnet18(pretrained=True).state_dict()
            for k, v in pretrained_dict.items():
                if "layer1" in k:
                    update_dict[k.replace("layer1", "encoder.4", 1)] = v
                elif "layer2" in k:
                    update_dict[k.replace("layer2", "encoder.5", 1)] = v
                elif "layer3" in k:
                    update_dict[k.replace("layer3", "encoder.6", 1)] = v
                elif "layer4" in k and self.arg.model_args[
                        'backbone'] == 'resnet18':
                    update_dict[k.replace("layer4", "encoder.7", 1)] = v
                elif k == 'conv1.weight':
                    update_dict['encoder.0.weight'] = v
                elif k == 'bn1.weight':
                    update_dict['encoder.1.weight'] = v
                elif k == 'bn1.bias':
                    update_dict['encoder.1.bias'] = v
                elif k == 'bn1.running_mean':
                    update_dict['encoder.1.running_mean'] = v
                elif k == 'bn1.running_var':
                    update_dict['encoder.1.running_var'] = v
                elif k == 'bn1.num_batches_tracked':
                    update_dict['encoder.1.num_batches_tracked'] = v

        print('updated params:{}'.format(len(update_dict)))
        model_dict.update(update_dict)
        self.model.load_state_dict(model_dict)

        if isinstance(self.arg.loss, list):
            self.loss = dict()
            if 'URC' in self.arg.loss:
                self.loss['URC'] = losses.URCLoss(**self.arg.loss_args['URC'])
            if 'BRC' in self.arg.loss:
                self.loss['BRC'] = losses.BRCLoss(**self.arg.loss_args['BRC'])
            if 'MRC' in self.arg.loss:
                self.loss['MRC'] = losses.MRCLoss(**self.arg.loss_args['MRC'])
            if 'dice' in self.arg.loss:
                self.loss['dice'] = losses.WeightedDiceLoss(
                    **self.arg.loss_args['dice'])
            if 'bce' in self.arg.loss:
                self.loss['bce'] = losses.WeightedBCELoss(
                    **self.arg.loss_args['bce'])
        else:
            raise ValueError()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.arg.base_lr,
                                       momentum=0.9,
                                       nesterov=self.arg.nesterov,
                                       weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.arg.base_lr,
                                        weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(),
                                           lr=self.arg.base_lr,
                                           alpha=0.9,
                                           momentum=0,
                                           weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.scheduler == 'cosine':
            eta_min = self.arg.base_lr * (
                self.arg.scheduler_args['lr_decay_rate']**3)
            self.lr = eta_min + (self.arg.base_lr - eta_min) * (1 + math.cos(
                math.pi * self.meta_info['epoch'] / self.arg.num_epoch)) / 2
        elif self.arg.scheduler == 'step':
            steps = np.sum(self.meta_info['epoch'] > np.array(
                self.arg.scheduler_args['lr_decay_epochs']))
            if steps > 0:
                self.lr = self.arg.base_lr * (
                    self.arg.scheduler_args['lr_decay_rate']**steps)
        elif self.arg.scheduler == 'constant':
            self.lr = self.arg.base_lr
        else:
            raise ValueError('Invalid learning rate schedule {}'.format(
                self.args.scheduler))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        return self.lr

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = dict()
        loss_dict = dict()
        result_frag = []
        label_frag = []

        print("training dataloader length: ", len(loader))

        for idx, (_, label, image, prototype_label,
                  prototype_id) in enumerate(loader):

            if len(image.size()) > 5:
                data = data.squeeze(0)
                label = label.squeeze(0)
                image = image.squeeze(0)

            # get data
            label = label.float().to(self.dev)
            image = image.float().to(self.dev)

            # forward
            feature, output, backbone_feature = self.model(image)

            backbone_feature = backbone_feature.view(
                -1, backbone_feature.shape[-1])

            prototype_label = prototype_label.view(-1,
                                                   prototype_label.shape[-1])
            prototype_id = prototype_id.view(-1)
            prototype_id = prototype_id.data.numpy()

            # collect prototype feature
            for pidx, proto_id in enumerate(prototype_id):
                if proto_id == -1:
                    continue
                if proto_id not in self.prototype_update:
                    self.prototype_update[proto_id] = []
                    self.prototype_update[proto_id].append(
                        backbone_feature[pidx].squeeze().data.cpu().numpy())
                else:
                    self.prototype_update[proto_id].append(
                        backbone_feature[pidx].squeeze().data.cpu().numpy())

            result_frag.append(output.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())

            ctr_flag = True
            prototype_flag = False
            if self.meta_info['epoch'] >= self.arg.prototype_enable_epoch:
                prototype_flag = True

            loss = torch.tensor(0).float().to(self.dev)
            for k, v in self.loss.items():
                if self.meta_info['epoch'] < self.arg.prototype_enable_epoch:
                    if k == 'MRC':
                        continue
                    loss_dict[k] = self.loss[k](output, label, feature,
                                                ctr_flag, prototype_flag)
                else:
                    loss_dict[k] = self.loss[k](output, label, feature,
                                                ctr_flag, prototype_flag,
                                                backbone_feature,
                                                self.K_prototype,
                                                prototype_label, prototype_id)
                loss += loss_dict[k]
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()

            if 'total' not in loss_value:
                loss_value['total'] = []
            loss_value['total'].append(self.iter_info['loss'])
            for k, v in self.loss.items():
                if self.meta_info[
                        'epoch'] < self.arg.prototype_enable_epoch and k == 'MRC':
                    continue
                self.iter_info[k] = loss_dict[k].data.item()
                if k not in loss_value:
                    loss_value[k] = []
                loss_value[k].append(self.iter_info[k])

            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value['total'])
        for k, v in self.loss.items():
            if self.meta_info[
                    'epoch'] < self.arg.prototype_enable_epoch and k == 'MRC':
                continue
            self.epoch_info['mean_' + k] = np.mean(loss_value[k])

        self.show_epoch_info()
        self.io.print_timer()

        self.train_logger.add_scalar('loss', self.epoch_info['mean_loss'],
                                     self.meta_info['epoch'])
        for k, v in self.loss.items():
            if self.meta_info[
                    'epoch'] < self.arg.prototype_enable_epoch and k == 'MRC':
                continue
            self.train_logger.add_scalar(k, self.epoch_info['mean_' + k],
                                         self.meta_info['epoch'])

        # update prototype
        for k, v in self.prototype_update.items():
            self.prototype[k] = np.stack(self.prototype_update[k])

        items = sorted(self.prototype.items(), key=lambda d: d[0])
        for item in items:
            self.prototype[item[0]] = item[1]

        N_prototype = len(self.prototype.keys())
        self.K_prototype = torch.zeros(
            (N_prototype, backbone_feature.shape[-1])).to(image.device)
        for idx, (k, v) in enumerate(self.prototype.items()):
            self.K_prototype[idx] = torch.tensor(np.mean(v, axis=0))
        self.K_prototype = F.normalize(self.K_prototype, dim=1)

        self.prototype_update = dict()

        # visualize loss and metrics
        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)

        f1_score, accuracy, train_f1, train_acc = funcs.record_metrics(
            self.result, self.label, self.epoch_info['mean_loss'],
            self.arg.model_args['num_class'], self.arg.work_dir, 'train')

        self.train_logger.add_scalar('train-acc', train_acc,
                                     self.meta_info['epoch'])
        self.train_logger.add_scalar('train-F1', train_f1,
                                     self.meta_info['epoch'])

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        result_frag = []
        loss_value = dict()
        loss_dict = dict()
        label_frag = []

        print("validation dataloader length: ", len(loader))
        for idx, (_, label, image, prototype_label,
                  prototype_id) in enumerate(loader):

            if len(image.size()) > 5:
                data = data.squeeze(0)
                label = label.squeeze(0)
                image = image.squeeze(0)

            # get data
            data = data.float().to(self.dev)
            label = label.float().to(self.dev)
            image = image.float().to(self.dev)

            # inference
            with torch.no_grad():
                feature, output, backbone_feature = self.model(image)

            backbone_feature = backbone_feature.view(
                -1, backbone_feature.shape[-1])
            prototype_label = prototype_label.view(-1,
                                                   prototype_label.shape[-1])
            prototype_id = prototype_id.view(-1)
            prototype_id = prototype_id.data.numpy()

            result_frag.append(output.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())

            loss = torch.tensor(0).float().to(self.dev)
            for k, v in self.loss.items():
                if self.meta_info['epoch'] < self.arg.prototype_enable_epoch:
                    if k == 'MRC':
                        continue
                    loss_dict[k] = self.loss[k](output, label, feature, True,
                                                True)
                else:
                    loss_dict[k] = self.loss[k](output, label, feature, True,
                                                True, backbone_feature,
                                                self.K_prototype,
                                                prototype_label, prototype_id)
                loss += loss_dict[k]

            if 'total' not in loss_value:
                loss_value['total'] = []
            loss_value['total'].append(loss.item())
            for k, v in self.loss.items():
                if self.meta_info[
                        'epoch'] < self.arg.prototype_enable_epoch and k == 'MRC':
                    continue
                self.iter_info[k] = loss_dict[k].data.item()
                if k not in loss_value:
                    loss_value[k] = []
                loss_value[k].append(self.iter_info[k])

        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)

        self.epoch_info['mean_loss'] = np.mean(loss_value['total'])
        for k, v in self.loss.items():
            if self.meta_info[
                    'epoch'] < self.arg.prototype_enable_epoch and k == 'MRC':
                continue
            self.epoch_info['mean_' + k] = np.mean(loss_value[k])

        self.show_epoch_info()

        self.validation_logger.add_scalar('loss', self.epoch_info['mean_loss'],
                                          self.meta_info['epoch'])
        for k, v in self.loss.items():
            if self.meta_info[
                    'epoch'] < self.arg.prototype_enable_epoch and k == 'MRC':
                continue
            self.validation_logger.add_scalar(k, self.epoch_info['mean_' + k],
                                              self.meta_info['epoch'])

        # compute f1 score
        f1_score, accuracy, val_f1, val_acc = funcs.record_metrics(
            self.result, self.label, self.epoch_info['mean_loss'],
            self.arg.model_args['num_class'], self.arg.work_dir, 'val')
        if self.best_aver_f1 < val_f1:
            self.best_aver_f1 = val_f1
            self.best_f1 = f1_score
            torch.save(self.model.state_dict(),
                       os.path.join(self.arg.work_dir, 'final_model.pt'))

            state = {
                'model': self.model.state_dict(),
                'K_prototype': self.K_prototype,
            }
            torch.save(state,
                       os.path.join(self.arg.work_dir, 'final_proto_model.pt'))

        self.validation_logger.add_scalar('val-acc', val_acc,
                                          self.meta_info['epoch'])
        self.validation_logger.add_scalar('val-F1', val_f1,
                                          self.meta_info['epoch'])

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Supervised Contrastive Loss with Resnet')

        # region arguments yapf: disable
        # optim
        parser.add_argument('--base_lr', type=float,
                            default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD',
                            help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool,
                            default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float,
                            default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--smoothing', type=float,
                            default=0.0, help='label smoothing rate')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--pretrain', type=str2bool, default=True,
                            help='load pretrained weights on ImageNet or not')
        parser.add_argument('--prototype_enable_epoch', type=int, default=0,
                            help='which epoch to enable prototype loss')

        # loss
        parser.add_argument('--loss', default=None,
                            help='the loss will be used')
        parser.add_argument('--loss_args', action=DictAction,
                            default=dict(), help='the arguments of loss')

        # scheduler
        parser.add_argument('--scheduler', default='constant',
                            help='the scheduler will be used')
        parser.add_argument('--scheduler_args', action=DictAction,
                            default=dict(), help='the arguments of scheduler')

        # endregion yapf: enable

        return parser
