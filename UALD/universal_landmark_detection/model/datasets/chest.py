import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

# from data_pre_loc import json_to_numpy
from .data_pre_loc import json_to_numpy
from ..utils import gaussianHeatmap, transformer


class Chest(data.Dataset):

    def __init__(self, prefix, phase, file_name, transform_params=dict(), sigma=5, num_landmark=14, size=[512, 512],
                 use_abnormal=True, chest_set=None, exclude_list=None, use_background_channel=False):

        self.transform = transformer(transform_params)
        self.phase = phase
        self.size = tuple(size)
        self.num_landmark = num_landmark
        self.use_background_channel = use_background_channel

        self.pth_Image = os.path.join(prefix, 'pngs')
        self.pth_Label = os.path.join(prefix, 'labels')
        self.SINGLE = [[9000, 9000] for i in range(14)]
        # self.img_name_list = os.listdir(self.pth_Image)

        # file index
        # files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]
        files = os.listdir(self.pth_Image)
        # if chest_set is not None:
        #     files = [f for f in files if any(
        #         f.startswith(st) for st in chest_set)]
        # # if exclude_list is not None:
        # #     st = set(exclude_list)
        # #     files = [f for f in files if f not in st]
        # if not use_abnormal:
        #     files = [f for f in files if f[-1] == '0']

        n = len(files)
        train_num = 140  # round(n*0.7) # 180
        val_num = 17  # round(n*0.1)  # 24
        test_num = n - train_num - val_num
        if phase == 'train':
            self.indexes = files[:train_num]
        elif phase == 'validate':
            self.indexes = files[train_num:-test_num]
        elif phase == 'test':
            self.indexes = files[-test_num:]

        elif phase == 'single':
            self.indexes = [file_name]
        else:
            raise Exception("Unknown phase: {phase}".fomrat(phase=phase))
        self.genHeatmap = gaussianHeatmap(sigma, dim=len(size))

    def __getitem__(self, index):
        name = self.indexes[index]
        # print(name)
        ret = {'name': name}

        img, origin_size = self.readImage(
            os.path.join(self.pth_Image, name))

        points = []
        if self.phase == 'single':
            points = self.SINGLE
        else:
            points = self.readLandmark(index, origin_size)
        li = [self.genHeatmap(point, self.size) for point in points]
        if self.use_background_channel:
            sm = sum(li)
            sm[sm > 1] = 1
            li.append(1 - sm)
        gt = np.array(li)
        img, gt = self.transform(img, gt)
        ret['input'] = torch.FloatTensor(img)
        ret['gt'] = torch.FloatTensor(gt)
        return ret

    def __len__(self):
        return len(self.indexes)

    def readLandmark(self, index, origin_size):
        # path = os.path.join(self.pth_Label, name+'.txt')
        points = []
        img_name = self.indexes[index]
        before_json = img_name.rfind('.')
        file_name = img_name[:before_json]
        mask_name = file_name + '.json'
        points = json_to_numpy(os.path.join(self.pth_Label, mask_name))
        # with open(path, 'r') as f:
        #     n = int(f.readline())
        #     for i in range(n):
        #         ratios = [float(i) for i in f.readline().split()]
        #         pt = tuple([round(r*sz) for r, sz in zip(ratios, self.size)])
        #         points.append(pt)
        reg_points = []
        for p in points:
            reg_x = round(p[0] / origin_size[0] * self.size[1])
            reg_y = round(p[1] / origin_size[1] * self.size[0])
            pt = tuple([reg_x, reg_y])
            reg_points.append(pt)
        return reg_points

    def readImage(self, path):
        '''Read image from path and return a numpy.ndarray in shape of cxwxh
        '''
        img = Image.open(path)
        origin_size = img.size

        # resize, width x height,  channel=1
        img = img.resize(self.size)
        arr = np.array(img)
        # channel x width x height: 1 x width x height
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = np.expand_dims(np.transpose(arr, (1, 0)), 0).astype(np.float)
        # conveting to float is important, otherwise big bug occurs
        for i in range(arr.shape[0]):
            arr[i] = (arr[i] - arr[i].mean()) / (arr[i].std() + 1e-20)
        return arr, origin_size
