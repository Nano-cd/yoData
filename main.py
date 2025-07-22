# -*- coding=utf-8 -*-

# 1. 裁剪(需改变bbox)
# 2. 平移(需改变bbox)
# 3. 改变亮度
# 4. 加噪声
# 5. 旋转角度(需要改变bbox)
# 6. 镜像(需要改变bbox)
# 7. cutout
# 注意:
# random.seed(),相同的seed,产生的随机数是一样
import time
import random
import copy
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
import argparse


# 显示图片
def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200, 800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=5,
                 crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                 add_noise_rate=0.5, flip_rate=0.5,
                 cutout_rate=0.5, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5,
                 is_addNoise=True, is_changeLight=True, is_cutout=False, is_rotate_img_bbox=True,
                 is_crop_img_bboxes=True, is_shift_pic_bboxes=True, is_filp_pic_bboxes=True):

        # 配置各个操作的属性
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

        # 是否使用某种增强方式
        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_cutout = is_cutout
        self.is_rotate_img_bbox = is_rotate_img_bbox
        self.is_crop_img_bboxes = is_crop_img_bboxes
        self.is_shift_pic_bboxes = is_shift_pic_bboxes
        self.is_filp_pic_bboxes = is_filp_pic_bboxes

    # 加噪声
    def _addNoise(self, img):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        # return cv2.GaussianBlur(img, (11, 11), 0)
        return random_noise(img, mode='gaussian',  clip=True) * 255

    # 调整亮度
    def _changeLight(self, img):
        alpha = random.uniform(0.35, 1)
        blank = np.zeros(img.shape, img.dtype)
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, 0)

    # cutout
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):
        '''
        原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : 框的坐标
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''

        def cal_iou(boxA, boxB):
            '''
            boxA, boxB为两个框，返回iou
            boxB为bouding box
            '''
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            iou = interArea / float(boxBArea)
            return iou

        # 得到h和w
        if img.ndim == 3:
            h, w, c = img.shape
        else:
            _, h, w, c = img.shape
        mask = np.ones((h, w, c), np.float32)
        for n in range(n_holes):
            chongdie = True  # 看切割的区域是否与box重叠太多
            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0,
                             h)  # numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box) > threshold:
                        chongdie = True
                        break
            mask[y1: y2, x1: x2, :] = 0.
        img = img * mask
        return img

    # 旋转
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        # ---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        return rot_img, rot_bboxes

    # 裁剪
    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        # ---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = y_min  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # ---------------------- 裁剪boundingbox ----------------------
        # 裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        return crop_img, crop_bboxes

    # 平移
    def _shift_pic_bboxes(self, img, bboxes):
        '''
        参考:https://blog.csdn.net/sty945/article/details/79387054
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 平移图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- 平移boundingbox ----------------------
        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])

        return shift_img, shift_bboxes

    # 镜像
    def _filp_pic_bboxes(self, img, bboxes):
        '''
            参考:https://blog.csdn.net/jningwei/article/details/78753607
            平移后的图片要包含所有的框
            输入:
                img:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 翻转图像 ----------------------

        flip_img = copy.deepcopy(img)
        h, w, _ = img.shape
        sed = random.random()

        if 0 < sed < 0.33:  # 0.33的概率水平翻转，0.33的概率垂直翻转,0.33是对角反转
            flip_img = cv2.flip(flip_img, 0)  # _flip_x
            inver = 0
        elif 0.33 < sed < 0.66:
            flip_img = cv2.flip(flip_img, 1)  # _flip_y
            inver = 1
        else:
            flip_img = cv2.flip(flip_img, -1)  # flip_x_y
            inver = -1

        # ---------------------- 调整boundingbox ----------------------
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]

            if inver == 0:
                # 0：垂直翻转
                flip_bboxes.append([x_min, h - y_max, x_max, h - y_min])
            elif inver == 1:
                # 1：水平翻转
                flip_bboxes.append([w - x_max, y_min, w - x_min, y_max])
            elif inver == -1:
                # -1：水平垂直翻转
                flip_bboxes.append([w - x_max, h - y_max, w - x_min, h - y_min])
        return flip_img, flip_bboxes

    # 图像增强方法
    def dataAugment(self, img, bboxes):
        '''
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        '''
        change_num = 0  # 改变的次数
        while change_num < 1:  # 默认至少有一种数据增强生效
            if self.is_rotate_img_bbox and random.random() < self.rotation_rate:
                change_num += 1
                angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                scale = random.uniform(0.7, 0.8)
                img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)

            if self.is_shift_pic_bboxes and random.random() < self.shift_rate:
                change_num += 1
                img, bboxes = self._shift_pic_bboxes(img, bboxes)

            if self.is_changeLight and random.random() < self.change_light_rate:
                change_num += 1
                img = self._changeLight(img)

            if self.is_addNoise and random.random() < self.add_noise_rate:
                change_num += 1
                img = self._addNoise(img)

            if self.is_cutout and random.random() < self.cutout_rate:
                change_num += 1
                img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes,
                                   threshold=self.cut_out_threshold)

            if self.is_filp_pic_bboxes and random.random() < self.flip_rate:
                change_num += 1
                img, bboxes = self._filp_pic_bboxes(img, bboxes)
        return img, bboxes


# TXT标签(YOLO格式)解析和保存工具
class ToolHelper():
    # 从yolo格式的txt文件中提取bounding box信息
    def parse_yolo_txt(self, path, img_width, img_height):
        '''
        输入：
            path: txt标签文件路径
            img_width: 图像原始宽度
            img_height: 图像原始高度
        输出：
            从txt文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, class_id]]
        '''
        coords = list()
        if not os.path.exists(path):
            return coords

        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])

                    # 将归一化坐标转换为绝对像素坐标 (x_min, y_min, x_max, y_max)
                    abs_w = width_norm * img_width
                    abs_h = height_norm * img_height
                    x_min = (x_center_norm * img_width) - (abs_w / 2)
                    y_min = (y_center_norm * img_height) - (abs_h / 2)
                    x_max = x_min + abs_w
                    y_max = y_min + abs_h
                    coords.append([x_min, y_min, x_max, y_max, class_id])
        return coords

    # 保存图片结果
    def save_img(self, file_name, save_folder, img):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        cv2.imwrite(os.path.join(save_folder, file_name), img)

    # 将增强后的bboxs保存为yolo格式的txt文件
    def save_yolo_txt(self, file_name, save_folder, class_ids, bboxs, img_width, img_height):
        '''
        :param file_name: 要保存的txt文件名 (e.g., "image_1.txt")
        :param save_folder: 保存txt文件的文件夹
        :param class_ids: 对象的类别ID列表
        :param bboxs: 对象的边界框列表 [[x_min, y_min, x_max, y_max], ...]
        :param img_width: 增强后图像的宽度
        :param img_height: 增强后图像的高度
        '''
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        content = []
        for class_id, box in zip(class_ids, bboxs):
            x_min, y_min, x_max, y_max = box

            # 将绝对像素坐标转换为YOLO格式的归一化坐标
            abs_w = x_max - x_min
            abs_h = y_max - y_min
            x_center = x_min + abs_w / 2
            y_center = y_min + abs_h / 2

            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = abs_w / img_width
            height_norm = abs_h / img_height

            # 确保坐标在[0, 1]范围内
            x_center_norm = np.clip(x_center_norm, 0, 1)
            y_center_norm = np.clip(y_center_norm, 0, 1)
            width_norm = np.clip(width_norm, 0, 1)
            height_norm = np.clip(height_norm, 0, 1)

            content.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

        with open(os.path.join(save_folder, file_name), 'w') as f:
            f.write("\n".join(content))


if __name__ == '__main__':

    need_aug_num = 20  # 每张图片需要增强的次数

    dataAug = DataAugmentForObjectDetection()  # 数据增强工具类
    toolhelper = ToolHelper()  # 工具类

    # 获取相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_path', type=str, default='E:/project_pycharm/MYMLOPs/comsuption_mid/dataset/demo/g/areaA',
                        help='原图片文件夹路径')
    parser.add_argument('--source_txt_path', type=str, default='E:/project_pycharm/MYMLOPs/comsuption_mid/dataset/demo/g/labels',
                        help='原YOLO格式标签文件夹路径')
    parser.add_argument('--save_img_path', type=str, default='E:/project_pycharm/MYMLOPs/comsuption_mid/dataset/demo/g/datasets/Images', help='增强后图片的保存文件夹')
    parser.add_argument('--save_txt_path', type=str, default='E:/project_pycharm/MYMLOPs/comsuption_mid/dataset/demo/g/datasets/labels',
                        help='增强后YOLO格式标签的保存文件夹')
    args = parser.parse_args()

    source_img_path = args.source_img_path  # 图片原始位置
    source_txt_path = args.source_txt_path  # txt的原始位置

    save_img_path = args.save_img_path  # 图片增强结果保存文件夹
    save_txt_path = args.save_txt_path  # txt增强结果保存文件夹

    # 如果保存文件夹不存在就创建
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    if not os.path.exists(save_txt_path):
        os.makedirs(save_txt_path)

    for parent, _, files in os.walk(source_img_path):
        files.sort()
        for file in files:
            cnt = 0

            # 分离文件名和后缀
            dot_index = file.rfind('.')
            if dot_index == -1: continue  # 跳过没有后缀的文件
            _file_prefix = file[:dot_index]
            _file_suffix = file[dot_index:]

            pic_path = os.path.join(parent, file)
            txt_path = os.path.join(source_txt_path, _file_prefix + '.txt')

            img = cv2.imread(pic_path)
            if img is None:
                print(f"警告：无法读取图片 {pic_path}")
                continue

            img_height, img_width, _ = img.shape

            # 解析得到box信息, 格式为[[x_min, y_min, x_max, y_max, class_id]]
            values = toolhelper.parse_yolo_txt(txt_path, img_width, img_height)
            if not values:
                print(f"信息：标签文件 {txt_path} 为空或不存在，跳过增强。")
                continue  # 如果没有标签，则不进行增强

            coords = [v[:4] for v in values]  # 得到框 [[x_min,y_min,x_max,y_max]]
            labels = [v[-1] for v in values]  # 对象的类别ID

            # show_pic(img, coords)  # 显示原图

            while cnt < need_aug_num:  # 继续增强
                auged_img, auged_bboxes = dataAug.dataAugment(copy.deepcopy(img), copy.deepcopy(coords))

                # 将浮点数bbox转为整数
                auged_bboxes_int = np.array(auged_bboxes).astype(np.int32)

                # 获取增强后图像的属性
                aug_height, aug_width, _ = auged_img.shape

                # 准备保存的文件名
                img_name = f'{_file_prefix}_{cnt + 1}{_file_suffix}'
                txt_name = f'{_file_prefix}_{cnt + 1}.txt'

                # 保存增强图片
                toolhelper.save_img(img_name, save_img_path, auged_img)

                # 保存增强后的YOLO格式标签文件
                toolhelper.save_yolo_txt(txt_name, save_txt_path, labels, auged_bboxes_int, aug_width, aug_height)

                # show_pic(auged_img, auged_bboxes)  # 显示强化后的图
                print(f"已生成增强文件: {img_name} 和 {txt_name}")
                cnt += 1
