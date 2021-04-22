# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/20 13:48
# @Author : liumin
# @File : export_torchscript.py

import argparse
import torch


# ONNX export
def export_torchscript(opt):
    device = torch.device('cuda:' + str(opt.device) if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(opt.weights, map_location=device)  # load
    model = ckpt['model'].float().fuse().eval()  # FP32 model

    # Input
    img = torch.zeros(1, 3, *opt.img_size).to(device)

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    export_torchscript(opt)