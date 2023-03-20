import numpy as np
import imgaug.augmenters as iaa
import cv2
import matplotlib.pyplot as plt

# images = np.random.randint(low=0,high=255,size=(10,10,4),dtype='uint8')
#
# print(images)
#
# aug = iaa.Affine(
#         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#         rotate=(-25, 25),
#         shear=(-8, 8)
#     )
#
# geometric = iaa.Sequential([iaa.Affine(
#                        scale=(1, 1.2),
#                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#                        rotate=(-45, 45)),
#     iaa.PiecewiseAffine(scale=(0.01, 0.05))])
#
# move = iaa.PiecewiseAffine(scale=(0.01, 0.05))
#
# images = geometric(image=images)
#
# print(images)

# has_anomaly = np.array([1], dtype=np.float32)
# no_anomaly = np.array([0], dtype=np.float32)
# a = []
#
# for i in range(4):
#     a.append(has_anomaly)
#     a.append(no_anomaly)
#
# a = np.array(a)
# is_normal = a[0]
# print(a.shape)
# a = a.reshape(a.shape[0])
# print(a)

# auc = [93.2, 96.8, 96.6, 100, 97.1, 94.4, 100, 96.9, 99.7, 100, 100, 99.8, 93.1, 100, 100]
# auc_pixel = [89.4, 98.4, 96.5, 98.9, 96.0, 88.5, 99.3, 94.8, 95.0, 97.8, 99.3, 96.0, 96.7, 99.6, 96.0]
# ap = [98.6, 98.5, 99.0, 100, 99.5, 94.9, 100, 98.2, 99.9, 100, 100, 99.9, 97.7, 100, 100]
# ap_pixel = [46.4, 86.4, 64.2, 67.4, 44.5, 46.5, 95.2, 69.2, 66.4, 58.8, 95.0, 70.9, 45.3, 70.5, 80.8]
#
# auc = [92.1,100,99.1,100,95.4,93.0,100,97.7,100,100,99.8,100,92.5,100,99.2]
# auc_pixel = [81.0,98.6,98.1,97.9,97.9,77.2,99.5,97.1,99.1,97.0,99.8,93.8,95.0,98.3,95.6]
# ap = [98.2,100,99.7,100,99.0,91.4,100,98.5,100,100,100,100,97.1,100,99.7]
# # ap_pixel = [18.9,90.1,84.8,75.1,86.3,30.1,96.8,80.7,89.8,51.4,99.8,75.5,55.4,67.6,71.5]
# print(sum(ap_pixel)/len(auc))


# img = cv2.imread("/Users/yanzhenyu/Downloads/mvtec_anomaly_detection/pill/train/good/000.png", cv2.IMREAD_COLOR)
img = cv2.imread("/Users/yanzhenyu/Downloads/Magnetic-Tile-Defect/train/MT_Free/exp1_num_114376.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.title('title')
plt.imshow(img)
plt.show()