"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from collections import namedtuple

import cupy
import numpy as np
import torch
from PIL import Image

LOCAL_AFFINE_KERNEL_PATH = "best_local_affine_kernel.cu"


def smooth_local_affine(output_cpu, input_cpu, epsilon, patch, h, w, f_r, f_e):
    with open(LOCAL_AFFINE_KERNEL_PATH) as f:
        m = cupy.RawModule(code=f.read())

    _reconstruction_best_kernel = m.get_function("reconstruction_best_kernel")
    _bilateral_smooth_kernel = m.get_function("bilateral_smooth_kernel")
    _best_local_affine_kernel = m.get_function("best_local_affine_kernel")
    Stream = namedtuple("Stream", ["ptr"])
    s = Stream(ptr=torch.cuda.current_stream().cuda_stream)

    filter_radius = f_r
    sigma1 = filter_radius / 3
    sigma2 = f_e
    radius = (patch - 1) / 2

    filtered_best_output = torch.zeros(np.shape(input_cpu)).cuda()
    affine_model = torch.zeros((h * w, 12)).cuda()
    filtered_affine_model = torch.zeros((h * w, 12)).cuda()

    input_ = torch.from_numpy(input_cpu).cuda()
    output_ = torch.from_numpy(output_cpu).cuda()
    _best_local_affine_kernel(
        grid=(int((h * w) / 256 + 1), 1),
        block=(256, 1, 1),
        args=[
            output_.data_ptr(),
            input_.data_ptr(),
            affine_model.data_ptr(),
            np.int32(h),
            np.int32(w),
            np.float32(epsilon),
            np.int32(radius),
        ],
        stream=s,
    )

    _bilateral_smooth_kernel(
        grid=(int((h * w) / 256 + 1), 1),
        block=(256, 1, 1),
        args=[
            affine_model.data_ptr(),
            filtered_affine_model.data_ptr(),
            input_.data_ptr(),
            np.int32(h),
            np.int32(w),
            np.int32(f_r),
            np.float32(sigma1),
            np.float32(sigma2),
        ],
        stream=s,
    )

    _reconstruction_best_kernel(
        grid=(int((h * w) / 256 + 1), 1),
        block=(256, 1, 1),
        args=[
            input_.data_ptr(),
            filtered_affine_model.data_ptr(),
            filtered_best_output.data_ptr(),
            np.int32(h),
            np.int32(w),
        ],
        stream=s,
    )
    numpy_filtered_best_output = filtered_best_output.cpu().numpy()
    return numpy_filtered_best_output


def smooth_filter(initImg, contentImg, f_radius=15, f_edge=1e-1):
    """
    :param initImg: intermediate output. Either image path or PIL Image
    :param contentImg: content image output. Either path or PIL Image
    :return: stylized output image. PIL Image
    """
    if type(initImg) == str:
        initImg = Image.open(initImg).convert("RGB")
    best_image_bgr = np.array(initImg, dtype=np.float32)
    bW, bH, bC = best_image_bgr.shape
    best_image_bgr = best_image_bgr[:, :, ::-1]
    best_image_bgr = best_image_bgr.transpose((2, 0, 1))

    if type(contentImg) == str:
        contentImg = Image.open(contentImg).convert("RGB")
    content_input = contentImg.resize((bH, bW))
    content_input = np.array(content_input, dtype=np.float32)
    content_input = content_input[:, :, ::-1]
    content_input = content_input.transpose((2, 0, 1))
    input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.0
    _, H, W = np.shape(input_)
    output_ = np.ascontiguousarray(best_image_bgr, dtype=np.float32) / 255.0
    best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, f_radius, f_edge)
    best_ = best_.transpose(1, 2, 0)
    result = Image.fromarray(np.uint8(np.clip(best_ * 255.0, 0, 255.0)))
    return result
