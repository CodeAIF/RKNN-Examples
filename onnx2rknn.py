#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ONN2RKNN 
@File    ：onnx2rknn.py
@Author  ：zz
@Date    ：2022/3/8 下午4:56 
'''
import argparse
import os
from rknn.api import RKNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--onnx', type=str, default='./2.7_80x80_MiniFASNetV2.onnx', help='weights path')  # from yolov5/models/
    parser.add_argument('--rknn', type=str, default='', help='save path')
    parser.add_argument("-p", '--precompile', action="store_true", help='pre')
    parser.add_argument("-o", '--original', action="store_true", help='source model')
    parser.add_argument("-bs", '--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    ONNX_MODEL = opt.onnx
    if opt.rknn:
        RKNN_MODEL = opt.rknn
    else:
        RKNN_MODEL = "%s.rknn" % os.path.splitext(ONNX_MODEL)[0]
    rknn = RKNN()
    print('--> config model')
    if opt.original:
        print("---------------")
        rknn.config(channel_mean_value='0 0 0 1',reorder_channel='0 1 2',
                    batch_size=opt.batch_size, target_platform="rk3399pro")  # reorder_channel='0 1 2',
    else:
        print("-----------2-----------")
        rknn.config(batch_size=8, reorder_channel='2 1 0', mean_values=[[0, 0, 0]], std_values=[[1.0, 1.0, 1.0]])
    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    assert ret == 0, "Load onnx failed!"

    # Build model
    print('--> Building model')
    if opt.precompile:
        ret = rknn.build(do_quantization=False, dataset='./dataset.txt', pre_compile=True)  # pre_compile=True
    else:
        ret = rknn.build(do_quantization=False, dataset='./dataset.txt')
    assert ret == 0, "Build onnx failed!"
    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    assert ret == 0, "Export %s.rknn failed!" % opt.rknn
    print('done')
