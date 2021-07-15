from __future__ import division

import argparse
import csv
from collections import OrderedDict
import os
import sys
import torch.nn.functional as F

import scipy.misc
import torch
import torch.backends.cudnn as cudnn

sys.path.append(os.path.split(os.getcwd())[0])
import models
from models.readout_nets.center_bias_layer import CenterBiasLayer
from utils import np_transforms
from utils.dataset import get_dataset_config, TestDataset
from utils.loaders import TestLoader
from utils.loss import KLD

model_names = sorted(name for name in models.__dict__
    if not name.startswith("_") and callable(models.__dict__[name]))

def parse_args():
    parser = argparse.ArgumentParser(description='Predict saliency-map')
    parser.add_argument('-i', '--img_files', nargs='+', default=None,
                        help='path to image files')
    parser.add_argument('--training-dataset-name', default='osie',
                        help='name of dataset in training phase(default: osie)')
    parser.add_argument('--root-training-dataset-dir', default='/home/wuchenjunlin/new_program/datasets', type=str,
                        help='root training dataset directory')
    parser.add_argument('--target-dataset-name', default='osie',
                        help='name of dataset for prediction(default: pascals)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='dpnsal131_dilation_multipath',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: dpnsal131_dilation_multipath)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--use-center-bias-layer', dest='use_center_bias_layer', action='store_true',
                        help='using center bias layer')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                        help='distributed backend')
    parser.add_argument('--root-out-dir', default='/home/wuchenjunlin/new_program/outputs/output', type=str,
                        help='output directory')
    parser.add_argument('--sub-out-dir', default='output', type=str,
                        help='output directory under root-out-dir')
    parser.add_argument('--filename2number', dest='filename2number', action='store_true',
                        help='convert file name into number')
    parser.add_argument('--enlarge-img', dest='enlarge_img', action='store_true',
                        help='enlarge output image')
    parser.add_argument('--enlarge_input', dest='enlarge_input', action='store_true',
                        help='enlarge input image')
    parser.add_argument('--cpu', dest='cpu', action='store_true',
                        help='use cpu')

    return parser.parse_args()


def main(args):
    args.out_dir = os.path.join(args.root_out_dir, args.sub_out_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=False)
    if args.cpu:
        if not args.distributed:
            model = torch.nn.DataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if not args.distributed:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        if 'center_bias_state_dict' not in checkpoint:
            args.use_center_bias_layer = False
        if args.use_center_bias_layer:
            center_bias_layer = CenterBiasLayer().cuda()
            center_bias_layer.load_state_dict(checkpoint['center_bias_state_dict'])
            center_bias_layer.eval()
            weight = center_bias_layer.weight.data[0,0,:,:].cpu().numpy()
            out_weight_file = os.path.join(args.out_dir, 'center_bias.png')
            scipy.misc.imsave(out_weight_file, weight)
    else:
        raise OSError("No checkpoint found at '{}'".format(args.resume))

    with open(os.path.join(args.out_dir, 'config.csv'), 'w') as f:
        columns = ['Model', 'Resume', 'Center bias', 'Training dataset', 'Target dataset']
        values = [args.arch, args.resume, args.use_center_bias_layer, args.training_dataset_name, args.target_dataset_name]
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerow(values)

    mean = get_dataset_config(args.training_dataset_name, args.root_training_dataset_dir).mean_rgb
    if args.enlarge_input:
        resize_function = np_transforms.ResizeToInnerRectangle(rec_long_side=640, rec_short_side=480)
    else:
        resize_function = np_transforms.ResizeToInnerRectangle(rec_long_side=640, rec_short_side=480)
    test_dataset = TestDataset(
        args.img_files,
        mean,
        transform=np_transforms.Compose([
            resize_function,
            #np_transforms.Resize(scale_factor=1.0),
            np_transforms.ToTensor(),
        ]),
    )
    test_loader = TestLoader(test_dataset)

    for i, (file_name, inputs) in enumerate(test_loader, start=1):
        if args.cpu:
            input_var = torch.autograd.Variable(inputs, volatile=True)
            print("input1:",input_var.size())
        else:
            inputs = F.upsample(inputs, size = (480,640), mode='bilinear') 
            input_var = torch.autograd.Variable(inputs.cuda(), volatile=True)
            #print("input2:",input_var.size())
            
        
        if args.use_center_bias_layer:
            saliency_map = center_bias_layer(model(input_var))
        else:
            saliency_map = model(input_var) # return tensor whose size is (1, 1, H, W).
        saliency_map = saliency_map[0,0,:,:].data.cpu().numpy()

        if args.filename2number:
            out_file = os.path.join(args.out_dir, str(i)+'.png')
        else:
            out_file = os.path.join(args.out_dir, file_name+'.png')
        if args.enlarge_img:
            saliency_map = scipy.misc.imresize(saliency_map, (input_var.size(2), input_var.size(3)))
        scipy.misc.imsave(out_file, saliency_map)
        print('- out_file: {0}'.format(out_file))


if __name__ == '__main__':
    args = parse_args()
    main(args)
