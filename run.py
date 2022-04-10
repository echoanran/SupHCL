import sys
import numpy as np
import os
import json
import argparse
from ruamel import yaml
from au_lib.meta_utils import compute_class_frequency, compute_label_frequency

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--processor_name',
                        type=str,
                        default='train-HACK',
                        help='processor name')
    parser.add_argument('-c',
                        '--config_dir',
                        type=str,
                        default='./config/HACK',
                        help='config dir name')
    parser.add_argument('-w',
                        '--work_dir',
                        type=str,
                        default='./work_dir/train/bp4d/HACK',
                        help='work dir name')
    parser.add_argument('-d',
                        '--data_dir',
                        type=str,
                        default='./data/25bp4d-stride1-T1-ac512r4',
                        help='data dir name')
    parser.add_argument('-k', '--kfold', type=int, default=3, help='kfold')
    parser.add_argument('--num_class',
                        type=int,
                        default=12,
                        help='num of class to detect')

    args = parser.parse_args()

    if not os.path.exists(args.config_dir):
        os.mkdir(args.config_dir)

    for k in range(args.kfold):

        label_freq = compute_label_frequency(
            os.path.join(args.data_dir, 'train' + str(k) + '_label.pkl'))

        class_freq = compute_class_frequency(
            os.path.join(args.data_dir, 'train' + str(k) + '_label.pkl'))

        lambda_representation = 4

        pcc_postfix = '_unique100'
        pcc_matrix_postfix = '_unique'

        desired_caps = {
            'work_dir': os.path.join(args.work_dir, str(k)),
            'feeder': 'feeder.feeder_HACK.Feeder',
            'train_feeder_args': {
                'label_path':
                os.path.join(args.data_dir, 'train' + str(k) + '_label.pkl'),
                'image_path':
                os.path.join(args.data_dir,
                             'train' + str(k) + '_imagepath.pkl'),
                'prototype_path':
                './misc/pcc_bp4d_train' + str(k) + pcc_postfix + '.npy',
                'sample_weights':
                np.loadtxt('./misc/pcc_bp4d_train' + str(k) + pcc_postfix +
                           '_weights.txt').tolist(),
                'image_size':
                256,
                'istrain':
                True,
                'num_frame':
                8,
            },
            'test_feeder_args': {
                'label_path':
                os.path.join(args.data_dir, 'val' + str(k) + '_label.pkl'),
                'image_path':
                os.path.join(args.data_dir, 'val' + str(k) + '_imagepath.pkl'),
                'prototype_path':
                './misc/pcc_bp4d_train' + str(k) + pcc_postfix + '.npy',
                'sample_weights':
                np.loadtxt('./misc/pcc_bp4d_train' + str(k) + pcc_postfix +
                           '_weights.txt').tolist(),
                'image_size':
                256,
                'istrain':
                False,
                'num_frame':
                1,
            },
            'batch_size': 1,
            'test_batch_size': 1,
            'num_worker': 1,
            'debug': False,
            'model': 'net.HACK.Model',
            'model_args': {
                'num_class': args.num_class,
                'backbone': 'resnet18',
                'pooling': True,
                'd_clf': 256,
                'd_contrast': 256,
            },
            'log_interval': 500,
            'save_interval': 5,
            'device': [0],
            'base_lr': 0.0001,
            'scheduler': 'cosine',
            'scheduler_args': {
                'lr_decay_rate': 0.3,
                'lr_decay_epochs': [],
            },
            'num_epoch': 15,
            'optimizer': 'SGD',
            'weight_decay': 0.0005,
            'loss': ['URC', 'BRC', 'MRC', 'bce', 'dice'],
            'pretrain': True,
            'prototype_enable_epoch': 1,
            'seed': 17,
            'loss_args': {
                'dice': {
                    'weights': label_freq.tolist(),
                    'lambda_clf': 1,
                },
                'bce': {
                    'weights': label_freq.tolist(),
                    'lambda_clf': 1,
                },
                'MRC': {
                    'lambda_prototype':
                    lambda_representation,
                    'prototype_weights':
                    np.loadtxt('./misc/pcc_bp4d_train' + str(k) + pcc_postfix +
                               '_weights.txt').tolist(),
                    'num_class':
                    args.num_class,
                    'temperature':
                    0.1,
                },
                'URC': {
                    'temperature': 0.1,
                    'feature_dim': 256,
                    'lambda_ctr': lambda_representation,
                    'class_weights': class_freq.tolist(),
                    'instance_weights': label_freq.tolist(),
                },
                'BRC': {
                    'temperature':
                    0.1,
                    'lambda_ctr':
                    lambda_representation,
                    'adjacent_matrix':
                    json.load(
                        open(
                            './misc/pcc_matrix/pcc_bp4d_train' + str(k) +
                            pcc_matrix_postfix + '.json',
                            'r'))['pcc_bp4d_train' + str(k) +
                                  pcc_matrix_postfix],
                    'pcc_threshold':
                    0.4,
                    'is_weights':
                    False,
                },
            },
        }

        yamlpath = os.path.join(args.config_dir, 'train' + str(k) + '.yaml')
        with open(yamlpath, "w", encoding="utf-8") as f:
            yaml.dump(desired_caps, f, Dumper=yaml.RoundTripDumper)

        cmdline = "python main.py " + args.processor_name + " -c " + yamlpath
        print(cmdline)
        os.system(cmdline)