import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
import sys
import random
import math


def transform_image(image_path, mask_path, resize_shape):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        mask = np.zeros((image.shape[0], image.shape[1]))
    if resize_shape != None:
        image = cv2.resize(image, dsize=(resize_shape[1], resize_shape[0]))
        mask = cv2.resize(mask, dsize=(resize_shape[1], resize_shape[0]))

    image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
    mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

    return image, mask


def get_gt(dataset, img_path):
    dir_path, file_name = os.path.split(img_path)
    dir2_path, base_dir = os.path.split(dir_path)
    base_dir2 = os.path.basename(dir2_path)

    if dataset == 'mvtec':
        if base_dir == 'good':
            mask_path = None
            has_anomaly = np.array([0], dtype=np.float32)
            return mask_path, has_anomaly
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0] + "_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            has_anomaly = np.array([1], dtype=np.float32)
            return mask_path, has_anomaly

    elif dataset == 'mt':
        if base_dir == 'MT_Free2':
            mask_path = None
            has_anomaly = np.array([0], dtype=np.float32)
            return mask_path, has_anomaly
        else:
            mask_file_name = file_name.split(".")[0] + ".png"
            mask_path = os.path.join(dir_path, mask_file_name)
            has_anomaly = np.array([1], dtype=np.float32)
            return mask_path, has_anomaly

    elif dataset == 'aitex':
        if base_dir == 'Defect_images':
            mask_path = os.path.join(dir_path, '../Mask_images/')
            mask_file_name = file_name.split(".")[0] + "_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            has_anomaly = np.array([1], dtype=np.float32)
            return mask_path, has_anomaly
        elif base_dir2 == 'NODefect_images':
            mask_path = None
            has_anomaly = np.array([0], dtype=np.float32)
            return mask_path, has_anomaly



class PathList():
    def __init__(self, args, root_dir):
        self.test_neg = []
        self.train_neg = []
        if args.dataset == 'mvtec':
            self.train_pos = sorted(glob.glob(root_dir+"/train/good/*.png"))
            test_dir = root_dir + "/test/"
            for ad in os.listdir(test_dir):
                if ad != "good":
                    self.test_neg += sorted(glob.glob(test_dir+ad+"/*.png"))
            self.test_pos = sorted(glob.glob(test_dir+"/good/*.png"))
        elif args.dataset == 'mt':
            self.train_pos = sorted(glob.glob(root_dir + "/train/MT_Free/*.jpg"))
            self.test_neg = sorted(glob.glob(root_dir + "/test/*/Imgs/*.jpg"))
            self.test_pos = sorted(glob.glob(root_dir + "/train/MT_Free2/*.jpg")) # 将mt_free中exp6划分为测试正常样本
        elif args.dataset == 'aitex':
            self.train_pos = sorted(glob.glob(root_dir + "/NODefect_images/23*/*.png"))
            self.test_neg = sorted(glob.glob(root_dir + "/Defect_images/*.png"))
            self.test_pos = sorted(glob.glob(root_dir + "/NODefect_images/26*/*.png"))  # 将nodefect中26*文件夹划分为测试正常样本
        self.test_paths = self.test_pos + self.test_neg

    def get_path(self, ad):
        if ad > 0:
            ad_num = math.ceil(ad * len(self.test_neg))
            self.test_neg = np.random.permutation(self.test_neg)
            self.train_neg = self.test_neg[:ad_num]
            self.test_neg = self.test_neg[ad_num:]
            self.test_paths = self.test_pos + self.test_neg.tolist()
        return self.train_pos, self.train_neg, self.test_paths


class TestDataset(Dataset):
    def __init__(self, args, test_paths, resize_shape=None):
        self.dataset = args.dataset
        self.test_paths = test_paths
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.test_paths)

    def __getitem__(self, item):
        image_path = self.test_paths[item]
        mask_path, has_anomaly = get_gt(self.dataset, image_path)
        test_img, test_gt = transform_image(image_path, mask_path, self.resize_shape)
        test_img = test_img / 255.0
        test_gt = test_gt / 255.0
        test_img = np.transpose(test_img, (2, 0, 1))
        test_gt = np.transpose(test_gt, (2, 0, 1))
        sample = {'image': test_img, 'mask': test_gt, 'has_anomaly': has_anomaly}
        return sample


class TrainDataset(Dataset):
    def __init__(self, args, train_pos, trains_neg, resize_shape=None):
        self.resize_shape = resize_shape
        self.dataset = args.dataset
        self.train_pos = train_pos
        self.train_neg = trains_neg
        self.trans = transforms.Compose([transforms.ToTensor()])
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.augmenters = iaa.SomeOf(3, [
                            sometimes(iaa.GammaContrast((0.5, 2.0))),   # per_channel=True
                            sometimes(iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30))),
                            sometimes(iaa.pillike.EnhanceSharpness()),
                            sometimes(iaa.AddToHueAndSaturation((-50, 50), per_channel=True)),
                            sometimes(iaa.Solarize(0.5, threshold=(32, 128))),
                            sometimes(iaa.Posterize()),
                            sometimes(iaa.Invert()),
                            sometimes(iaa.pillike.Autocontrast()),
                            sometimes(iaa.pillike.Equalize()),
                            # iaa.imgcorruptlike.GaussianNoise(severity=1),
                            # iaa.GaussianBlur(sigma=(0, 0.5)),
                            ])
        self.geometric = iaa.Sequential(
            [
                sometimes(iaa.Affine(scale=(0.9, 1.1))),
                sometimes(iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})),
                iaa.Sometimes(0.5, iaa.Affine(rotate=(-45, 45))),
                iaa.Sometimes(0.1, iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ])
        self.rot = iaa.Affine(rotate=(-90, 90))
        self.rot90 = iaa.Rot90((1, 3))

    def __len__(self):
        return len(self.train_pos)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 5, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]],
                              self.augmenters[aug_ind[3]],
                              self.augmenters[aug_ind[4]],
                              ]
                             )
        return aug

    def augment_image(self, train_pos, train_neg, gt):
        # aug = self.randAugmenter()
        train_neg = np.array(train_neg)
        gt = np.array(gt)
        # target_neg1 = train_neg
        # target_mask1 = gt
        # print(train_neg.shape, gt.shape)    #(256,256,3),(256,256,1)
        cat = np.append(train_neg, gt, axis=2)
        translate_neg = self.geometric(image=cat.astype('uint8'))
        train_neg = translate_neg[:, :, 0:-1]
        mask = translate_neg[:, :, -1]
        # target_mask2 = np.array(mask)
        # target_neg2 = np.array(train_neg)

        train_neg_aug = self.augmenters(image=train_neg.astype('uint8'))
        # target_neg2 = train_neg_aug

        train_neg_aug = train_neg_aug.astype(np.float32) / 255.0
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32) / 255.0


        beta = torch.rand(1).numpy()[0] * 0.5
        augmented_image = (1 - beta) * train_neg_aug * mask + beta * train_pos * mask

        no_anomaly = torch.rand(1).numpy()[0]
        # no_anomaly = 0
        if no_anomaly > 0.5:
            train_pos = train_pos.astype(np.float32)
            return train_pos, np.zeros_like(mask, dtype=np.float32), np.array([0.0], dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            mask = (mask).astype(np.float32)
            augmented_image = mask * augmented_image + train_pos * (1 - mask)
            has_anomaly = 1.0
            if np.sum(mask) == 0:
                has_anomaly = 0.0
            return augmented_image, mask, np.array([has_anomaly], dtype=np.float32) #, target_neg1, target_neg2, target_mask1, target_mask2



    def get_train_image(self, pos_path, neg_path):
        train_pos = cv2.imread(pos_path)
        train_pos = cv2.cvtColor(train_pos, cv2.COLOR_BGR2RGB)
        train_pos = cv2.resize(train_pos, dsize=(self.resize_shape[1], self.resize_shape[0]))
        mask_path, has_nomaly = get_gt(self.dataset, neg_path)
        train_neg, gt = transform_image(neg_path, mask_path, self.resize_shape)

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            if self.dataset == 'mvtec':
                train_pos = self.rot(image=train_pos)
            else:
                train_pos = self.rot90(image=train_pos)

        train_pos = np.array(train_pos).reshape((train_pos.shape[0], train_pos.shape[1], train_pos.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(train_pos, train_neg, gt)
        # augmented_image, anomaly_mask, has_anomaly, target_neg1, target_neg2, t_mask1, t_mask2 = self.augment_image(train_pos, train_neg, gt)

        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        train_pos = np.transpose(train_pos, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))

        # target_neg1 = np.array(target_neg1).astype(np.float32) / 255.0
        # target_neg2 = np.array(target_neg2).astype(np.float32) / 255.0
        # t_mask1 = np.array(t_mask1).reshape((t_mask1.shape[0], t_mask1.shape[1], 1)).astype(np.float32) / 255.0
        # t_mask2 = np.array(t_mask2).reshape((t_mask2.shape[0], t_mask2.shape[1], 1)).astype(np.float32) / 255.0

        # target_neg1 = np.transpose(target_neg1, (2, 0, 1))
        # target_neg2 = np.transpose(target_neg2, (2, 0, 1))
        # t_mask1 = np.transpose(t_mask1, (2, 0, 1))
        # t_mask2 = np.transpose(t_mask2, (2, 0, 1))

        augmented_image = np.ascontiguousarray(augmented_image)
        train_pos = np.ascontiguousarray(train_pos)
        anomaly_mask = np.ascontiguousarray(anomaly_mask)

        # target_neg1 = np.ascontiguousarray(target_neg1)
        # target_neg2 = np.ascontiguousarray(target_neg2)
        # t_mask1 = np.ascontiguousarray(t_mask1)
        # t_mask2 = np.ascontiguousarray(t_mask2)

        return train_pos, augmented_image, anomaly_mask, has_anomaly    #, target_neg1, target_neg2, t_mask1, t_mask2

    def __getitem__(self, idx):
        anomaly_source_idx = torch.randint(0, len(self.train_neg), (1,)).item()

        image, augmented_image, anomaly_mask, has_anomaly = self.get_train_image(
            self.train_pos[idx],
            self.train_neg[anomaly_source_idx])

        # image, augmented_image, anomaly_mask, has_anomaly, target_neg1, target_neg2, t_mask1, t_mask2 = self.get_train_image(self.train_pos[idx],
        #                                                                          self.train_neg[anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly}
                  # 'target_neg1':target_neg1, 'target_neg2':target_neg2, 'target_mask1':t_mask1, 'target_mask2': t_mask2}
        return sample











"""original"""

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, dataset, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.dataset = dataset
        if dataset == 'mvtec':
            self.image_paths = sorted(glob.glob(root_dir+"/test/*/*.png"))
        elif dataset == 'mt':
            self.image_paths = sorted(glob.glob(root_dir + "/test/*/Imgs/*.jpg"))
            self.image_paths += sorted(glob.glob(root_dir+"/train/MT_Free2/*.jpg"))# 将mt_free中exp6划分为测试正常样本
        elif dataset == 'aitex':
            self.image_paths = sorted(glob.glob(root_dir + "Defect_images/*.png"))
            self.image_paths += sorted(glob.glob(root_dir + "NODefect_images/26*/*.png"))  # 将nodefect中26*文件夹划分为测试正常样本
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.image_paths)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        dir_path, file_name = os.path.split(img_path)
        dir2_path, base_dir = os.path.split(dir_path)
        base_dir2 = os.path.basename(dir2_path)

        if self.dataset == 'mvtec':
            if base_dir == 'good':
                image, mask = self.transform_image(img_path, None)
                has_anomaly = np.array([0], dtype=np.float32)
            else:
                mask_path = os.path.join(dir_path, '../../ground_truth/')
                mask_path = os.path.join(mask_path, base_dir)
                mask_file_name = file_name.split(".")[0]+"_mask.png"
                mask_path = os.path.join(mask_path, mask_file_name)
                image, mask = self.transform_image(img_path, mask_path)
                has_anomaly = np.array([1], dtype=np.float32)

        elif self.dataset == 'mt':
            if base_dir == 'MT_Free2':
                image, mask = self.transform_image(img_path, None)
                has_anomaly = np.array([0], dtype=np.float32)
            else:
                mask_file_name = file_name.split(".")[0] + ".png"
                mask_path = os.path.join(dir_path, mask_file_name)
                image, mask = self.transform_image(img_path, mask_path)
                has_anomaly = np.array([1], dtype=np.float32)

        elif self.dataset == 'aitex':
            if base_dir == 'Defect_images':
                mask_path = os.path.join(dir_path, '../Mask_images/')
                mask_file_name = file_name.split(".")[0] + "_mask.png"
                mask_path = os.path.join(mask_path, mask_file_name)
                image, mask = self.transform_image(img_path, mask_path)
                has_anomaly = np.array([1], dtype=np.float32)
            elif base_dir2 == 'NODefect_images':
                image, mask = self.transform_image(img_path, None)
                has_anomaly = np.array([0], dtype=np.float32)


        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample


class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, args, root_dir, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.expand = np.arange(0, epoch) * args.expand /epoch
        # self.expand = self.expand[::-1]
        # self.transparence = np.arange(0, epoch) / epoch
        # self.transparence = self.transparence [::-1]
        self.T = 0
        self.set_t = args.set_t
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.dataset = args.dataset
        if args.dataset == 'mvtec':
            self.image_paths = sorted(glob.glob(root_dir+"/train/good/*.png"))
        elif args.dataset == 'mt':
            self.image_paths = sorted(glob.glob(root_dir + "/train/*/*.jpg"))
        elif args.dataset == 'aitex':
            self.image_paths = sorted(glob.glob(root_dir + "/NODefect_images/23*/*.png"))

        # anomaly_source_paths = np.random.permutation(anomaly_source_list)
        self.anomaly_source_paths = sorted(glob.glob(args.anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.rot90 = iaa.Rot90((1, 3))

    def __len__(self):
        return len(self.image_paths)

    def set_T(self, current, total):
        self.T = current/total

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        if self.set_t is None:
            threshold = 0.5
            beta = torch.rand(1).numpy()[0] * 0.8
        elif "linear" in self.set_t:
            threshold = 0.2 + 0.6 * self.T  # 小->大,异常面积大->小
            beta = 0.1 + 0.8 * self.T  # 异常1-beta，正常beta，beta小->大，异常纹理变浅
            # print("linear threshold:", threshold, "beta:", beta)
        elif "random" in self.set_t:
            r = math.ceil(self.T*4)
            threshold = np.random.uniform(low=0.2+0.15*(r-1), high=0.2+0.15*r, size=1)[0]
            beta = np.random.uniform(low=0.1+0.2*(r-1), high=0.1+0.2*r, size=1)[0]


        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)


    # def agu_texture(self, image, anomaly_source_list, anomaly_mask_list):



    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            if self.dataset == 'mvtec':
                image = self.rot(image=image)
            else:
                image = self.rot90(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        # idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()

        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}
        return sample
