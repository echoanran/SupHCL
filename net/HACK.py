import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(SpatialAttention, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size - 1) // 2

        self.grp1_conv1k = nn.Conv2d(self.in_channels,
                                     self.in_channels // 2,
                                     (1, self.kernel_size),
                                     padding=(0, pad))
        self.grp1_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.grp1_convk1 = nn.Conv2d(self.in_channels // 2,
                                     1, (self.kernel_size, 1),
                                     padding=(pad, 0))
        self.grp1_bn2 = nn.BatchNorm2d(1)

        self.grp2_convk1 = nn.Conv2d(self.in_channels,
                                     self.in_channels // 2,
                                     (self.kernel_size, 1),
                                     padding=(pad, 0))
        self.grp2_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.grp2_conv1k = nn.Conv2d(self.in_channels // 2,
                                     1, (1, self.kernel_size),
                                     padding=(0, pad))
        self.grp2_bn2 = nn.BatchNorm2d(1)

    def forward(self, input_):
        # Generate Group 1 Features
        grp1_feats = self.grp1_conv1k(input_)
        grp1_feats = F.relu(self.grp1_bn1(grp1_feats))
        grp1_feats = self.grp1_convk1(grp1_feats)
        grp1_feats = F.relu(self.grp1_bn2(grp1_feats))

        # Generate Group 2 features
        grp2_feats = self.grp2_convk1(input_)
        grp2_feats = F.relu(self.grp2_bn1(grp2_feats))
        grp2_feats = self.grp2_conv1k(grp2_feats)
        grp2_feats = F.relu(self.grp2_bn2(grp2_feats))

        added_feats = torch.sigmoid(torch.add(grp1_feats, grp2_feats))
        added_feats = added_feats.expand_as(input_).clone()

        return added_feats


class Model(nn.Module):
    r"""The proposed H A C K

    Args:
        num_class (int): Number of classes for the multi-label classification task
        backbone (str): 'resnet18' as default
        dropout (int): for all the model
        **kwargs (optional): Other parameters

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, num_class)` 
          
    """
    def __init__(self,
                 num_class,
                 backbone='resnet18',
                 dropout=0.3,
                 pooling=False,
                 d_clf=256,
                 d_contrast=256,
                 **kwargs):
        super().__init__()
        self.num_class = num_class
        self.backbone = backbone
        self.dropout = dropout
        self.pooling = pooling

        self.d_clf = d_clf
        self.d_contrast = d_contrast

        if self.backbone == 'resnet18':
            self.encoder = nn.Sequential(
                *list(models.resnet18(pretrained=False).children())
                [:-2],  # [N, 512, image_size // (2^4), _]
            )
            self.output_channel = 512
            self.output_size = 8
        else:
            raise ValueError

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.pooling == False:
            self.spa_attn = nn.ModuleList([
                SpatialAttention(self.output_channel, kernel_size=3)
                for _ in range(num_class)
            ])
            self.projector = nn.ModuleList([
                nn.Linear(
                    self.output_channel * self.output_size * self.output_size,
                    self.d_clf) for _ in range(num_class)
            ])
            self.contrast_projector = nn.ModuleList([
                nn.Linear(
                    self.output_channel * self.output_size * self.output_size,
                    self.d_contrast) for _ in range(num_class)
            ])
            self.final = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.d_clf, 1),
                    nn.Sigmoid(),
                ) for _ in range(num_class)
            ])
        else:
            self.spa_attn = nn.ModuleList([
                SpatialAttention(self.output_channel, kernel_size=3)
                for _ in range(num_class)
            ])
            self.projector = nn.ModuleList([
                nn.Linear(self.output_channel, self.d_clf)
                for _ in range(num_class)
            ])
            self.contrast_projector = nn.ModuleList([
                nn.Linear(self.output_channel, self.d_contrast)
                for _ in range(num_class)
            ])
            self.final = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.d_clf, 1),
                    nn.Sigmoid(),
                ) for _ in range(num_class)
            ])

    def forward(self, image, subject_infos=None):
        '''
        image: [N, C, H, W]
        '''
        N, T, C, H, W = image.shape
        x = image.view(-1, image.shape[2], image.shape[3], image.shape[4])

        x = self.encoder(x)

        backbone_feature = self.avgpool(x).view(x.shape[0], x.shape[1])

        clf_feats = []
        contrast_feats = []

        for idx in range(self.num_class):
            attn_weight = self.spa_attn[idx](x)
            feat_w_attn = torch.mul(x, attn_weight)

            if self.pooling == False:
                feat_w_attn = feat_w_attn.view(feat_w_attn.shape[0], -1)
            else:
                feat_w_attn = self.avgpool(feat_w_attn)
                feat_w_attn = feat_w_attn.view(feat_w_attn.shape[0], -1)

            clf_feat = self.projector[idx](feat_w_attn)
            contrast_feat = self.contrast_projector[idx](feat_w_attn)

            clf_feats.append(clf_feat)
            contrast_feats.append(contrast_feat)

        # contrast
        feature = torch.stack(contrast_feats, dim=-1).view(x.shape[0], -1)

        feature = feature.view(N, T, -1)

        # clf
        x = torch.stack(clf_feats, dim=-1)

        cls_outputs = []
        for idx in range(self.num_class):
            cls_outputs.append(self.final[idx](x[:, :, idx]))
        output = torch.stack(cls_outputs, dim=-1).squeeze(1)

        output = output.view(N, T, -1)

        return feature, output, backbone_feature

    def visualize(self, image, subject_infos=None):
        '''
        image for cnn: [N, C, H, W] if single
                        [N, T, C, H, W] if sequential model (time_model is set)
        '''
        N, T, C, H, W = image.shape
        x = image.view(-1, image.shape[2], image.shape[3], image.shape[4])

        x = self.encoder(x)

        backbone_feature = self.avgpool(x).view(x.shape[0], x.shape[1])

        clf_feats = []
        contrast_feats = []
        attn_weights = []
        proto_feat = []

        proto_feat.append(backbone_feature)

        for idx in range(self.num_class):
            attn_weight = self.spa_attn[idx](x)
            feat_w_attn = torch.mul(x, attn_weight)

            if self.pooling == False:
                feat_w_attn = feat_w_attn.view(feat_w_attn.shape[0], -1)
            else:
                feat_w_attn = self.avgpool(feat_w_attn)
                feat_w_attn = feat_w_attn.view(feat_w_attn.shape[0], -1)

            clf_feat = self.projector[idx](feat_w_attn)
            contrast_feat = self.contrast_projector[idx](feat_w_attn)

            attn_weights.append(attn_weight)
            clf_feats.append(clf_feat)
            contrast_feats.append(contrast_feat)

        # proto
        proto_feat = torch.stack(proto_feat, dim=-1)

        # contrast
        feature = torch.stack(contrast_feats, dim=-1).view(x.shape[0], -1)

        feature = feature.view(N, T, -1)

        # clf
        x = torch.stack(clf_feats, dim=-1)

        cls_outputs = []
        for idx in range(self.num_class):
            cls_outputs.append(self.final[idx](x[:, :, idx]))
        output = torch.stack(cls_outputs, dim=-1).squeeze(1)

        output = output.view(N, T, -1)

        attn = torch.stack(attn_weights, dim=-1)

        return feature, output, attn, proto_feat