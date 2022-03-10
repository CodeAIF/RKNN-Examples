#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ONN2RKNN 
@File    ：Rknninfer.py
@Author  ：zz-ff
@Date    ：2022/3/9 下午4:30 
'''
# -*- coding: UTF-8 -*-
import os
import cv2
import time
import sys
import onnx
import numpy as np
import onnxruntime as ort
from rknn.api import RKNN



def load_model(model_name):
    print('-->loading model')
    rknn.load_rknn(model_name)
    print('loading model done')
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn


def onnx_inference(onnx_path, imgpath):

    predictor = onnx.load(onnx_path)
    onnx.checker.check_model(predictor)
    onnx.helper.printable_graph(predictor.graph)
    # predictor = backend.prepare(predictor, device="CPU")  # default CPU

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    label_name = ort_session.get_outputs()[0].name

    cvimg = cv2.imread(imgpath)

    # resize image to [128, 128]
    resized = cv2.resize(cvimg, (IMG_SIZE, IMG_SIZE))


    image = np.transpose(resized, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    feature = ort_session.run(None, {input_name: image})

    return feature


if __name__ == '__main__':
    RKNN_MODEL = './live/2.7_80x80_MiniFASNetV2.rknn'
    onnx_model = "./live/2.7_80x80_MiniFASNetV2.onnx"
    imgp = "test.jpg"
    IMG_SIZE = 80
    # Create RKNN object
    rknn = RKNN()
    # Export RKNN model
    rknn = load_model(RKNN_MODEL)

    # Set inputs
    img = cv2.imread(imgp)
    # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print("live--output:",outputs)

    onnxout=onnx_inference(onnx_model,imgp)
    print("onnx output:",onnxout)

    rknn.release()
