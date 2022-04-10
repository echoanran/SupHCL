from locale import normalize
import torch
from torch.cuda import device_count
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random
import os
import numpy as np
from scipy import stats

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WeightedBCELoss(nn.Module):
    def __init__(self, weights, lambda_clf=1, size_average=True):
        super(WeightedBCELoss, self).__init__()
        self.size_average = size_average
        self.weights = torch.tensor(weights)
        self.lambda_clf = lambda_clf

    def forward(self, outputs, targets, *args):
        num_class = outputs.size()[-1]
        outputs = outputs.view(-1, num_class)
        targets = targets.view(-1, num_class)

        N, num_class = outputs.size()
        loss_buff = 0
        for i in range(num_class):
            target = targets[:, i]
            output = outputs[:, i]
            loss_au = torch.sum(-(
                (1.0 - self.weights[i]) * target * torch.log(
                    (output + 0.05) / 1.05) + self.weights[i] *
                (1.0 - target) * torch.log((1.05 - output) / 1.05)))
            loss_buff += (1.0 - self.weights[i]) * loss_au
        return self.lambda_clf * loss_buff / (num_class * N)


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=0.0, p=1, reduction='sum'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[
            0], "predict & target batch size don't match"

        num = torch.sum(torch.mul(predict, target)) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class WeightedDiceLoss(nn.Module):
    def __init__(self,
                 weights=None,
                 ignore_index=None,
                 lambda_clf=1,
                 **kwargs):
        super(WeightedDiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weights = torch.tensor(weights)
        self.ignore_index = ignore_index
        self.lambda_clf = lambda_clf

    def forward(self, outputs, targets, *args):
        num_class = outputs.size()[-1]
        outputs = outputs.view(-1, num_class)
        targets = targets.view(-1, num_class)

        N, num_class = outputs.size()

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0

        for i in range(num_class):
            if i != self.ignore_index:
                dice_loss = dice(outputs[:, i], targets[:, i])
                if self.weights is not None:
                    assert self.weights.shape[0] == targets.shape[1], \
                        'Expect weights shape [{}], get[{}]'.format(
                            targets.shape[1], self.weights.shape[0])
                    dice_loss *= 1 - self.weights[i]
                total_loss += dice_loss

        return self.lambda_clf * (total_loss / targets.shape[1])


class BRCLoss(nn.Module):
    def __init__(self,
                 temperature=0.1,
                 base_temperature=1,
                 lambda_ctr=1,
                 weights=None,
                 adjacent_matrix=None,
                 pcc_threshold=0.5,
                 is_weights=True,
                 is_stable=False,
                 **kwargs):
        super(BRCLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.lambda_ctr = lambda_ctr
        self.weights = weights
        if self.weights is not None:
            self.weights = 1 - torch.tensor(weights)
            self.weights = self.weights.float().to(device)

        self.adjacent_matrix = adjacent_matrix
        self.pcc_threshold = pcc_threshold

        self.is_weights = is_weights
        self.is_stable = is_stable

        if self.adjacent_matrix is not None:
            self.graph = torch.tensor(self.adjacent_matrix)
            if self.is_weights:
                self.graph[self.graph > self.pcc_threshold] = 1
                self.graph[self.graph <= self.pcc_threshold] = 0
            else:
                self.graph[self.graph <= self.pcc_threshold] = 0

    def forward(self, outputs, targets, features, ctr_enable=True, *args):

        device = (torch.device('cuda')
                  if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        n_views = features.shape[1]
        num_class = outputs.shape[-1]

        if self.adjacent_matrix is None:
            mask = torch.eye(num_class, dtype=torch.float32).to(device)
        else:
            mask = self.graph.float().to(device)
            neg_mask = 1 - mask

        ctr_loss = []
        if not ctr_enable:
            return torch.tensor(0).float().to(device)

        for i in range(batch_size):
            sample_feature = features[i, :, :].view(n_views, num_class, -1)

            sample_feature = F.normalize(sample_feature.view(
                n_views * num_class, -1),
                                         dim=1)

            vector_dot_product = torch.div(
                torch.matmul(sample_feature, sample_feature.T),
                self.temperature)

            # for numerical stability
            if self.is_stable:
                logits_max, _ = torch.max(vector_dot_product,
                                          dim=1,
                                          keepdim=True)
                vector_dot_product = vector_dot_product - logits_max.detach()

            mask = mask.repeat(n_views, n_views)
            neg_mask = neg_mask.repeat(n_views, n_views)

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask), 1,
                torch.arange(n_views * num_class).view(-1, 1).to(device), 0)
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(vector_dot_product).to(device) * logits_mask

            log_prob = vector_dot_product.to(device) - torch.log(
                exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

            # loss for one AU
            graph_ctr_loss = -(self.temperature /
                               self.base_temperature) * mean_log_prob_pos

            ctr_loss.append(graph_ctr_loss.view(n_views, num_class).mean())

        ctr_loss = torch.stack(ctr_loss)
        if self.weights is not None:
            ctr_loss = ctr_loss * self.weights

        ctr_loss = torch.sum(ctr_loss)
        ctr_loss /= batch_size

        return self.lambda_ctr * ctr_loss


class URCLoss(nn.Module):
    def __init__(self,
                 temperature=0.1,
                 base_temperature=1,
                 feature_dim=256,
                 lambda_ctr=1,
                 class_weights=None,
                 instance_weights=None,
                 is_stable=False,
                 **kwargs):
        super(URCLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

        self.feature_dim = feature_dim
        self.lambda_ctr = lambda_ctr
        self.class_weights = class_weights
        self.instance_weights = instance_weights

        if self.class_weights is not None:
            self.class_weights = 1 - torch.tensor(class_weights)
            self.class_weights = self.class_weights.float().to(device)

        if self.instance_weights is not None:
            self.instance_weights = 1 - torch.tensor(instance_weights)
            self.instance_weights = self.instance_weights.float().to(device)

        self.is_stable = is_stable

    def forward(self, outputs, targets, features, ctr_enable=True, *args):
        """
            Contrastive loss for each AU feature.
        Args:
            features: features, output of the network extract_feature
                     hidden vector of shape [batch_size, num_views, feature_dim * num_classes].
            targets: labels, provide supervision, ground truth of shape [batch_size, num_views, num_classes].
        Returns:
            contrastive loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        n_views = features.shape[1]
        num_class = outputs.shape[-1]

        targets = targets.contiguous().view(batch_size, n_views, -1)

        ctr_loss = []
        if not ctr_enable:
            return torch.tensor(0).float().to(device)

        # calculate supervised contrastive loss for each AU
        for i in range(num_class):
            au_feature = features[:, :, i * self.feature_dim:(i + 1) *
                                  self.feature_dim]

            au_feature = torch.cat(torch.unbind(au_feature, dim=1), dim=0)

            au_feature = F.normalize(au_feature, dim=1)

            au_label = targets[:, :, i]
            au_label = torch.cat(torch.unbind(au_label, dim=1), dim=0)
            au_label = au_label.contiguous().view(-1, 1)

            mask = torch.eq(au_label, au_label.T).float().to(device)

            vector_dot_product = torch.div(
                torch.matmul(au_feature, au_feature.T), self.temperature)

            # for numerical stability
            if self.is_stable:
                logits_max, _ = torch.max(vector_dot_product,
                                          dim=1,
                                          keepdim=True)
                vector_dot_product = vector_dot_product - logits_max.detach()

            logits_mask = torch.scatter(
                torch.ones_like(mask), 1,
                torch.arange(batch_size * n_views).view(-1, 1).to(device), 0)
            mask = mask * logits_mask

            if self.instance_weights is not None:
                pos_neg_mask = torch.zeros(au_label.shape[0]).to(device)
                pos_neg_mask[au_label.squeeze() ==
                             1] = self.instance_weights[i]
                pos_neg_mask[au_label.squeeze() ==
                             0] = 1 - self.instance_weights[i]
                pos_neg_mask = pos_neg_mask.repeat(au_label.shape[0], 1)
                mask *= pos_neg_mask

            # compute log_prob
            exp_logits = torch.exp(vector_dot_product).to(device) * logits_mask
            log_prob = vector_dot_product.to(device) - torch.log(
                exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

            # loss for one AU
            au_ctr_loss = -(self.temperature /
                            self.base_temperature) * mean_log_prob_pos

            ctr_loss.append(au_ctr_loss.view(n_views, batch_size).mean())

        ctr_loss = torch.stack(ctr_loss)
        if self.class_weights is not None:
            ctr_loss = ctr_loss * self.class_weights

        ctr_loss = torch.sum(ctr_loss)
        ctr_loss /= num_class

        return self.lambda_ctr * ctr_loss




class MRCLoss(nn.Module):
    def __init__(self,
                 is_stable=True,
                 temperature=0.1,
                 base_temperature=1,
                 normalize=True,
                 num_class=12,
                 lambda_prototype=1,
                 prototype_weights=None,
                 is_reverse=False,
                 size_average=True):
        super(MRCLoss, self).__init__()

        self.is_stable = is_stable
        self.size_average = size_average
        self.lambda_prototype = lambda_prototype
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.normalize = normalize

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.L2norm = nn.MSELoss(size_average=False)

        self.is_reverse = is_reverse
        if prototype_weights is None:
            self.prototype_weights = torch.ones(num_class).to(device)
        else:
            if self.is_reverse:
                self.prototype_weights = 1 - torch.tensor(
                    prototype_weights).to(device)
            else:
                self.prototype_weights = torch.tensor(prototype_weights).to(
                    device)

    def forward(self,
                outputs,
                targets,
                features=None,
                ctr_enable=True,
                prototype_enable=True,
                backbone_features=None,
                prototype_features=None,
                prototype_labels=None,
                prototype_ids=None,
                *args):

        backbone_features = F.normalize(backbone_features, dim=1)

        num_class = outputs.size()[-1]
        outputs = outputs.view(-1, num_class)
        targets = targets.view(-1, num_class)

        N, num_class = outputs.size()
        num_prototype = len(prototype_features)

        ctr_loss = []
        for idx in range(N):
            sample_feature = backbone_features[idx]
            prototype_label = prototype_labels[idx][:num_prototype]

            vector_dot_product = torch.div(
                torch.matmul(sample_feature, prototype_features.T),
                self.temperature)

            # for numerical stability
            if self.is_stable:
                logits_max = torch.max(vector_dot_product)
                vector_dot_product = vector_dot_product - logits_max.detach()

            mask = prototype_label.to(device)
            mask = mask * self.prototype_weights

            # compute log_prob
            exp_logits = torch.exp(vector_dot_product).to(device) * (1 - mask)
            log_prob = vector_dot_product.to(device) - torch.log(
                exp_logits.sum())

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum()

            # loss for one AU
            au_ctr_loss = -(self.temperature /
                            self.base_temperature) * mean_log_prob_pos

            ctr_loss.append(au_ctr_loss.mean())

        ctr_loss = torch.stack(ctr_loss)
        ctr_loss = torch.sum(ctr_loss)
        ctr_loss /= N

        return self.lambda_prototype * ctr_loss