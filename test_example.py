import argparse
from datetime import datetime
from numpy import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.measure import label
from model_code import EGFPNet
import nibabel as nib
import SimpleITK as sitk
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='EGFPNet',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=10, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    args = parser.parse_args()

    return args


def Normalize(data):
    data_normalize = data.copy()
    max = data_normalize.max()
    min = data_normalize.min()
    data_normalize = (data_normalize - min) / (max - min)
    return data_normalize

def largestConnectComponent(bw_img):
    labeled_img, num = label(bw_img, connectivity=2, background=0, return_num=True)
    # plt.figure(), plt.imshow(labeled_img, 'gray')

    max_label = 1
    max_num = 0
    if num > 0:
        for i in range(1, num + 1):  # 这里从1开始，防止将背景设置为最大连通域
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
    lcc = (labeled_img == max_label)

    return lcc + 0

def test(args, model, test_image):
    # switch to evaluate mode
    model.eval()
    results = np.zeros(shape=(test_image.shape[0], test_image.shape[1], test_image.shape[2]))
    with torch.no_grad():
        image = np.zeros(shape=(args.batch_size, 1, test_image.shape[1], test_image.shape[2]))

        for j in range(int(test_image.shape[0] // args.batch_size)):
            for k in range(args.batch_size):
                if j * args.batch_size + k <= test_image.shape[0] - 1:
                    image[k, :, :, :] = test_image[j * args.batch_size + k, :, :]
                else:
                    break
            input = torch.from_numpy(image).cuda().float()

            out_final, out_edge, e5_and_edge_to_out, d5_and_edge_to_out, d4_and_edge_to_out, d3_and_edge_to_out = model(input)
            out_final_nd = torch.sigmoid(out_final).data.cpu().numpy()
            for k1 in range(args.batch_size):
                if j * args.batch_size + k1 <= test_image.shape[0] - 1:
                    results[j * args.batch_size + k1, :, :] = out_final_nd[k1, 0, :, :]
                else:
                    break
    return results

def read_niigz(file_path):
    nimg = nib.load(file_path)
    image_arr = nimg.get_data()
    image_arr = image_arr.transpose(2, 0, 1).astype('float32')
    image_arr_all = np.zeros([image_arr.shape[0], 512, 512])
    min_ax_1 = (512 - image_arr.shape[1]) // 2
    min_ax_2 = (512 - image_arr.shape[2]) // 2
    image_arr_all[:, min_ax_1: (image_arr.shape[1] + min_ax_1), min_ax_2: (image_arr.shape[2] + min_ax_2)] = image_arr
    return image_arr_all[:, 64:448, 64:448]

def diejia(image, zu):
    for i in range(len(zu)):
        image.append(zu[i])

def jietu(jixian_people, test_image_people):
    jixian_people_zuobiao = [0, 0, 0, 0]
    size = 224
    jixian_people_zuobiao[0] = max(0, jixian_people[0] - (size - jixian_people[1] + jixian_people[0]) // 2)
    jixian_people_zuobiao[1] = jixian_people_zuobiao[0] + size
    if jixian_people_zuobiao[1] >= 384:
        jixian_people_zuobiao[1] = 384
        jixian_people_zuobiao[0] = jixian_people_zuobiao[1] - size

    jixian_people_zuobiao[2] = max(0, jixian_people[2] - (size - jixian_people[3] + jixian_people[2]) // 2)
    jixian_people_zuobiao[3] = jixian_people_zuobiao[2] + size
    if jixian_people_zuobiao[3] >= 384:
        jixian_people_zuobiao[3] = 384
        jixian_people_zuobiao[2] = jixian_people_zuobiao[3] - size
    validate_image_people = np.zeros([test_image_people.shape[0], size, size])
    validate_image_people[:, :, :] = test_image_people[:, jixian_people_zuobiao[0]:jixian_people_zuobiao[1], jixian_people_zuobiao[2]:jixian_people_zuobiao[3]]
    return validate_image_people, jixian_people_zuobiao

def sxzy(out_slice):
    ########处理每一个切片的结果
    jixian_people = [0, 0, 0, 0]
    test_label_people_1 = out_slice.copy()
    for j_1 in range(out_slice.shape[0]):
        if test_label_people_1[j_1, :].sum() > 0:
            jixian_people[0] = j_1  # 最高点
            break

    for j_2 in range(out_slice.shape[0]):
        if test_label_people_1[out_slice.shape[0] - j_2 - 1, :].sum() > 0:
            jixian_people[1] = out_slice.shape[0] - j_2 - 1
            break  # 最低点

    test_label_people_2 = out_slice.copy()
    for j_3 in range(out_slice.shape[1]):
        if test_label_people_2[:, j_3].sum() > 0:
            jixian_people[2] = j_3
            break  # 最左点

    for j_4 in range(out_slice.shape[1]):
        if test_label_people_2[:, out_slice.shape[1] - j_4 - 1].sum() > 0:
            jixian_people[3] = out_slice.shape[1] - j_4 - 1
            break  # 最右点
    return jixian_people

def main(args):
    test_image_people = read_niigz(os.path.join('/media/csl/5b58b9b5-0162-4611-b882-6351023584d1/csl/medical_dataset/ACDC/training/', 'patient001/',  'patient001_frame01.nii.gz'))    #MRI image

    #first stage
    start = datetime.now()
    first_image_people = Normalize(test_image_people)
    model = EGFPNet.EGFPNet()     #coarse model
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('/media/csl/5b58b9b5-0162-4611-b882-6351023584d1/csl/methods_data/edge_pyramid_net/models/acdc/label1/' + 'EGFPNet_coarse' + str(0) + '.pth'))
    first_stage_results = test(args, model, first_image_people)

    first_stage_results[first_stage_results > 0.5] = 1
    first_stage_results[first_stage_results < 0.5] = 0
    first_stage_results = largestConnectComponent(first_stage_results)

    out_slice = np.zeros([first_stage_results.shape[1], first_stage_results.shape[2]])
    for i in range(first_stage_results.shape[0]):
        out_slice[first_stage_results[i, :, :] == 1] = 1
    jixian = sxzy(out_slice)

    crop_image_people, jixian_people_location = jietu(jixian, test_image_people)
    second_image_people = Normalize(crop_image_people)

    torch.cuda.empty_cache()
    # second stage
    model = EGFPNet.EGFPNet()     #fine model
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('/media/csl/5b58b9b5-0162-4611-b882-6351023584d1/csl/methods_data/edge_pyramid_net/models/acdc/label1/' + 'EGFPNet_old' + str(0) + '.pth'))
    second_stage_results = test(args, model, second_image_people)

    #mapping
    second_stage_results[second_stage_results >= 0.5] = 1
    second_stage_results[second_stage_results < 0.5] = 0
    results = np.zeros(shape=(test_image_people.shape[0], test_image_people.shape[1], test_image_people.shape[2]))
    results[:, jixian_people_location[0]:jixian_people_location[1], jixian_people_location[2]:jixian_people_location[3]] = second_stage_results
    end = datetime.now()
    print('times', (end - start) / second_stage_results.shape[0])
    # save
    results = results.astype(np.uint8)
    out_label = sitk.GetImageFromArray(results)
    out_image = sitk.GetImageFromArray(test_image_people)
    sitk.WriteImage(out_label, os.path.join('/home/csl/pycharm_project/medical_sementation/edge_pyramid_net/result/acdc/label1/EGFPNet/', 'label', 'patient001' + '.nii.gz'))
    sitk.WriteImage(out_image, os.path.join('/home/csl/pycharm_project/medical_sementation/edge_pyramid_net/result/acdc/label1/EGFPNet/', 'image', 'patient001' + '.nii.gz'))

    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parse_args()
    main(args)
