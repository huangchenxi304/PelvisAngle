import os
from collections.abc import Iterable
import argparse
from functools import partial
from PIL import Image
from PIL import ImageDraw, ImageFont
from math import pi as PI
import math
from tqdm import tqdm
import scipy.io as sio
from scipy.optimize import linear_sum_assignment as assign
import numpy as np

from model.utils import mkdir, toYaml, dis2, colorRGB, getPointsFromHeatmap, get_config

PATH_DIC = {
    'cephalometric': '../data/ISBI2015_ceph/raw',
    'hand': '../data/hand/jpg',
    'chest': '../data/08_11/pngs',
}

FONT_PATH = './times.ttf'
THRESHOLD = [2, 2.5, 3, 4, 6, 9, 10]
CEPH_PHYSICAL_FACTOR = 0.46875
WRIST_WIDTH = 50  # mm
DRAW_TEXT_SIZE_FACTOR = {'cephalometric': 1.13, 'hand': 1, 'chest': 1.39}
CLASSES = ["LFH1", "LFH2", "LFH3", "LFHCE", "LIPSS", "LIPTE", "LOPAC", "RFH1", "RFH2", "RFH3", "RFHCE", "RIPSS",
           "RIPTE", "ROPAC"]


def np2py(obj):
    if isinstance(obj, Iterable):
        return [np2py(i) for i in obj]
    elif isinstance(obj, np.generic):
        return np.asscalar(obj)
    else:
        return obj


def radial(pt1, pt2, factor=1):
    if not isinstance(factor, Iterable):
        factor = [factor] * len(pt1)
    return sum(((i - j) * s) ** 2 for i, j, s in zip(pt1, pt2, factor)) ** 0.5


def draw_text(image, text, factor, point):
    width = round(7 * factor)
    # padding = round(10*factor)
    # margin = round(5*factor)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('simhei.ttf', width)
    text_size = draw.textsize(text, font)
    text_w = point[0] + 5
    text_h = point[1] + 5
    # text_w = text_h = padding
    pos = [text_w, text_h, text_w + text_size[0], text_h + text_size[1]]
    # draw.rectangle(pos, fill='#000000')  # 用于填充
    draw.text((text_w, text_h), text, fill='#00ffff', font=font)  # blue
    return image


def draw_text_index(image, text_dic, factor):
    width = round(8 * factor)
    padding = round(10 * factor)
    margin = round(5 * factor)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('simhei.ttf', width)

    # text_size = draw.textsize(text, font)

    text_w = text_h = padding
    # pos = [text_w, text_h, text_w + text_size[0], text_h+text_size[1]]
    # draw.rectangle(pos, fill='#000000')  # 用于填充
    for key, value in text_dic.items():
        text = key + ": " + str(value)
        draw.text((text_w, text_h), text, fill='#00ffff', font=font)  # blue
        text_h = text_h + 12
    return image


def cal_all_distance(points, gt_points, factor=1):
    '''
    points: [(x,y,z...)]
    gt_points: [(x,y,z...)]
    return : [d1,d2, ...]
    '''
    n1 = len(points)
    n2 = len(gt_points)
    if n1 == 0:
        print("[Warning]: Empty input for calculating mean and std")
        return 0, 0
    if n1 != n2:
        raise Exception("Error: lengthes dismatch, {}<>{}".format(n1, n2))
    return [radial(p, q, factor) for p, q in zip(points, gt_points)]


def assigned_distance(points, gt_points, factor=1):
    n = len(points)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = radial(points[i], gt_points[j]) * factor
    return [mat[i, j] for i, j in zip(*assign(mat))]


def get_sdr(distance_list, threshold=THRESHOLD):
    ''' successfully detection rate (pixel)
    '''
    ret = {}
    n = len(distance_list)
    for th in threshold:
        ret[th] = sum(d <= th for d in distance_list) / n
    return ret


def saveLabels(path, points, size):
    with open(path, 'w') as f:
        f.write('{}\n'.format(len(points)))
        for pt in points:
            ratios = ['{:.4f}'.format(x / X) for x, X in zip(pt, size)]
            f.write(' '.join(ratios) + '\n')


def rotate_angle(ripte, lipte):
    angle = np.arctan2(lipte[1] - ripte[1], lipte[0] - ripte[0]) * 180

    return angle


def matrix_totate(original_point, angle):
    old_p = np.array([original_point[0], original_point[1]])

    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    new_p = np.dot(rotation_matrix, old_p)

    return tuple([round(new_p[0]), round(new_p[1])])


def angle_between_non_intersecting_lines(start_point_1, end_point_1, start_point_2, end_point_2):
    m1 = (end_point_1[1] - start_point_1[1]) / (end_point_1[0] - start_point_1[0])
    m2 = (end_point_2[1] - start_point_2[1]) / (end_point_2[0] - start_point_2[0])

    angle_rad = abs(math.atan(m1) - math.atan(m2))
    angle_deg = angle_rad * 180 / PI

    return angle_deg


def AHI(FHCE, FH3, OPAC):
    distance_H = (math.sqrt(math.pow(FH3[1] - FHCE[1], 2) + math.pow(FH3[0] - FHCE[0], 2))) * 2

    distance_A = distance_H / 2 + (abs(OPAC[0] - FHCE[0]))

    ahi = distance_A / distance_H *100

    return ahi


def evaluate(input_path, output_path, phase, save_img=True, assigned=False, IS_DRAW_TEXT=False):
    mkdir(output_path)
    dataset = os.path.basename(input_path).lower()
    image_path_pre = PATH_DIC[dataset]
    print('\n' + '-' * 20 + dataset + '-' * 20)
    print('input : ', input_path)
    print('output: ', output_path)
    print('image : ', image_path_pre)
    gen = [gt_p for gt_p in os.listdir(input_path) if gt_p.endswith('_gt.npy')]
    pbar = tqdm(gen, ncols=80)
    data_num = len(gen)
    out_label_path = os.path.join(output_path, 'labels')
    mkdir(out_label_path)
    out_gt_path = os.path.join(output_path, 'gt_laels')
    mkdir(out_gt_path)

    out_img_path = os.path.join(output_path, 'images')
    mkdir(out_img_path)

    physical_factor = 1
    if dataset == 'cephalometric':
        physical_factor = CEPH_PHYSICAL_FACTOR
    distance_list = []
    pixel_dis_list = []
    assigned_list = []
    for i, gt_p in enumerate(pbar):
        pbar.set_description('{:03d}/{:03d}: {}'.format(i + 1, data_num, gt_p))
        name = gt_p[:-7]
        heatmaps = np.load(os.path.join(input_path, name + '_output.npy'))
        img_size = heatmaps.shape[1:]
        cur_points = getPointsFromHeatmap(heatmaps)
        gt_map = np.load(os.path.join(input_path, gt_p))
        cur_gt = getPointsFromHeatmap(gt_map)

        if dataset == 'hand':
            physical_factor = WRIST_WIDTH / radial(cur_gt[0], cur_gt[4])
        cur_distance_list = cal_all_distance(cur_points, cur_gt, physical_factor)
        cur_pixel_dis = cal_all_distance(cur_points, cur_gt, 1)
        distance_list += cur_distance_list
        pixel_dis_list += cur_pixel_dis
        if assigned:
            assigned_list += assigned_distance(cur_points,
                                               cur_gt, physical_factor)
        saveLabels(out_label_path + '/' + name + '.txt', cur_points, img_size)
        saveLabels(out_gt_path + '/' + name + '.txt', cur_gt, img_size)

        if dataset == 'cephalometric':
            img_path = image_path_pre + '/' + name + '.bmp'
        elif dataset == 'hand':
            img_path = image_path_pre + '/' + name + '.jpg'
        else:
            img_path = image_path_pre + '/' + name
        img = Image.open(img_path)
        img = img.resize(img_size)
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        img = np.transpose(img, (1, 0, 2))
        number = 0
        predict_points = []

        for p, q in zip(cur_points, cur_gt):
            predict_points.append(p)
            if phase != 'single':
                colorRGB(img, [q], partial(dis2, q), 20, [0, 255, 0])
            colorRGB(img, [p], partial(dis2, p), 20, [255, 0, 0])

        img = np.transpose(img, (1, 0, 2))
        img = Image.fromarray(img)
        mre = np.mean(cur_distance_list)
        mre_str = '{:.3f}'.format(mre)
        # if IS_DRAW_TEXT:
        #     img = draw_text(img, mre_str, DRAW_TEXT_SIZE_FACTOR[dataset])

        for point in predict_points:
            img = draw_text(img, CLASSES[number], DRAW_TEXT_SIZE_FACTOR[dataset], point)
            number = number + 1

        LCE_angle = angle_between_non_intersecting_lines(predict_points[5], predict_points[12], predict_points[3],
                                                         predict_points[6])

        Ljdqxj = angle_between_non_intersecting_lines(predict_points[5], predict_points[12], predict_points[4],
                                                      predict_points[6])

        Lsharp_angle = angle_between_non_intersecting_lines(predict_points[5], predict_points[12], predict_points[5],
                                                            predict_points[6])

        LAHI = AHI(predict_points[3], predict_points[2], predict_points[6])

        RCE_angle = angle_between_non_intersecting_lines(predict_points[5], predict_points[12], predict_points[10],
                                                         predict_points[13])

        Rjdqxj = angle_between_non_intersecting_lines(predict_points[5], predict_points[12], predict_points[11],
                                                      predict_points[13])

        Rsharp_angle = angle_between_non_intersecting_lines(predict_points[5], predict_points[12],
                                                            predict_points[12], predict_points[13])

        RAHI = AHI(predict_points[10], predict_points[9], predict_points[13])

        text_dic = {'LCE_angle': 90 - LCE_angle, 'RCE_angle': 90 - RCE_angle, 'Ltonnis': Ljdqxj, 'Rronnis': Rjdqxj,
                    'Lsharp_angle': Lsharp_angle, 'Rsharp_angle': Rsharp_angle, 'LAHI': LAHI, 'RAHI': RAHI}
        # draw_text_index(img, text_dic, DRAW_TEXT_SIZE_FACTOR[dataset])

        before_json = name.rfind('.')
        file_name = name[:before_json]
        img.save(out_img_path + '/' + file_name + '.png')

        text_list = [90 - LCE_angle, Ljdqxj,Lsharp_angle,LAHI,90 - RCE_angle,Rjdqxj,Rsharp_angle,RAHI]

        # text_new_dict = {'CE角': str(round(90 - LCE_angle)) + ',' + str(round(90 - RCE_angle)),
        #                  '臼顶倾斜角': str(round(Ljdqxj)) + ',' + str(round(Rjdqxj)),
        #                  'Sharp角': str(round(Lsharp_angle)) + ',' + str(round(Rsharp_angle)),
        #                  '头臼指数': str(round(LAHI)) + ',' + str(round(RAHI))}

        txt_path = out_img_path + '/' + file_name + '.txt'

        with open(txt_path, 'w') as f:
            for t in text_list:

                f.write(str(round(t,1)) + '\n')


    if assigned:
        print('assigned...')
    return assigned_list if assigned else distance_list, pixel_dis_list


def analysis(li1, dataset):
    print('\n' + '-' * 20 + dataset + '-' * 20)
    summary = {}
    mean1, std1, = np.mean(li1), np.std(li1)
    sdr1 = get_sdr(li1)
    n = len(li1)
    summary['LANDMARK_NUM'] = n
    summary['MRE'] = np2py(mean1)
    summary['STD'] = np2py(std1)
    summary['SDR'] = {k: np2py(v) for k, v in sdr1.items()}
    print('MRE:', mean1)
    print('STD:', std1)
    print('SDR:')
    for k in sorted(sdr1.keys()):
        print('     {}: {}'.format(k, sdr1[k]))
    return summary


def get_args():
    parser = argparse.ArgumentParser()
    # optinal
    parser.add_argument("-s", "--save_img", action='store_true')
    parser.add_argument("-d", "--draw_text", action='store_true')
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-C", "--config", type=str)
    parser.add_argument("-a", "--assigned", action='store_true')
    parser.add_argument("-p", "--phase", choices=['test', 'single'], default='single')
    # required
    parser.add_argument("-i", "--input", type=str, default='../runs/GU2Net_runs/results/single_epoch000')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    dic = {}
    pixel_dic = {}
    if not args.output:
        output = os.path.join('.eval', args.input.replace('/', '_'))
    for d in os.listdir(args.input):
        inp = os.path.join(args.input, d)
        if os.path.isdir(inp):
            phy_dis, pixel_dis = evaluate(inp, os.path.join(output, d), args.phase, args.save_img,
                                          args.assigned, args.draw_text)
            phy_dis = np2py(phy_dis)
            pixel_dis = np2py(pixel_dis)
            dic[d] = phy_dis
            pixel_dic[d + '_pixel'] = pixel_dis
    toYaml(output + '/distance.yaml', dic)
    summary = {}
    li_total = []
    for d, phy_dis in dic.items():
        pixel_dis = pixel_dic[d + '_pixel']
        summary[d] = analysis(phy_dis, d)
        li_total += pixel_dis
    summary['total'] = analysis(li_total, 'total')
    toYaml(output + '/summary.yaml', summary)
