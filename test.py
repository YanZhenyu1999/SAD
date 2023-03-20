import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from tensorboard_visualizer import TensorboardVisualizer
from tensorboard_visualizer import TensorboardVisualizer
import os


def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc"
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc"
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap"
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap"
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += run_name
    fin_str += "\n"
    fin_str += "--------------------------\n"


    with open("./outputs/results.txt",'a+') as file:
        file.write(fin_str)


def test(obj_names, args, epoch, test_dataset=None):
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    for obj_name in obj_names:
        img_dim = 256
        run_name = args.base_model_name + obj_name+'_'
        if args.visualize:
            visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name + "/"))

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        old_ckpt = torch.load(os.path.join(args.checkpoint_path, run_name + ".pckl"), 'cpu')
        new_ckpt = {}
        for k, v in old_ckpt.items():
            new_k = k.replace('module.', '')
            new_ckpt[new_k] = v
        model.load_state_dict(new_ckpt)
        model.cuda()
        model.eval()

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        old_ckpt = torch.load(os.path.join(args.checkpoint_path, run_name + "_seg.pckl"), 'cpu')
        new_ckpt = {}
        for k, v in old_ckpt.items():
            new_k = k.replace('module.', '')
            new_ckpt[new_k] = v
        model_seg.load_state_dict(new_ckpt)
        model_seg.cuda()
        model_seg.eval()


        root_dir = args.data_path
        if args.dataset == 'mvtec':
            root_dir += obj_name

        if test_dataset is None:
            test_dataset = MVTecDRAEMTestDataset(args.dataset, root_dir, resize_shape=[img_dim, img_dim])

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(test_dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(test_dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        cnt_display = 0
        display_indices = np.random.randint(len(test_loader), size=(16,))

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(test_loader):

                gray_batch = sample_batched["image"].cuda()

                is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]    #bs=1
                anomaly_score_gt.append(is_normal)
                true_mask = sample_batched["mask"]
                true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))  # （256，256，1）

                gray_rec = model(gray_batch)
                joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)    # (1,2,256,256)
                # 通道0(像素0)代表正常，通道1(像素1)代表异常，softmax 0 1分类后，和gt的one-hot标签对比

                out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()    # (256,256)
                # 对mask做了一次平滑
                out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                                   padding=21 // 2).cpu().detach().numpy()
                # print("test")
                # print("true_mask_cv:", true_mask_cv.shape)
                # print("out_mask:", out_mask.size())
                # print("out_mask_sm:", out_mask_sm.size())
                # print("out_mask_cv:", out_mask_cv.shape)

                if i_batch in display_indices:
                    t_mask = out_mask_sm[:, 1:, :, :]
                    display_images[cnt_display] = gray_rec[0]
                    display_gt_images[cnt_display] = gray_batch[0]
                    display_out_masks[cnt_display] = t_mask[0]
                    display_in_masks[cnt_display] = true_mask[0]
                    cnt_display += 1


                image_score = np.max(out_mask_averaged)

                anomaly_score_prediction.append(image_score)

                flat_true_mask = true_mask_cv.flatten()
                flat_out_mask = out_mask_cv.flatten()
                total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
                total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
                mask_cnt += 1

        if args.visualize:
            visualizer.visualize_image_batch(display_images, epoch, image_name='test_rec')
            visualizer.visualize_image_batch(display_gt_images, epoch, image_name='test_img')
            visualizer.visualize_image_batch(display_out_masks, epoch, image_name='test_out_mask')
            visualizer.visualize_image_batch(display_in_masks, epoch, image_name='test_gt_mask')

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        print(obj_name)
        print("AUC Image:  " +str(auroc))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("AP Image:  " +str(ap))
        print("AP Pixel:  " +str(ap_pixel))
        print("==============================")

        # if args.dataset == 'mt' or 'aitex':
        log_path = "./outputs/" + str(args.dataset) + "/" + str(obj_name) + "_result.log"
        log = "AUC Image:  " + str(auroc) + '\n' + "AUC Pixel:  " + str(auroc_pixel) + '\n' + \
              "AP Image:  " + str(ap) + '\n' + "AP Pixel:  " + str(ap_pixel) + '\n' + "==============================" + '\n' + run_name
        with open(log_path, 'a+') as file:
            file.write(log + '\n')
        return [auroc, auroc_pixel, ap, ap_pixel]


    # print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    # print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
    # print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    # print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))
    # write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)





if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--dataset', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true', required=False)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    args = parser.parse_args()


    args = parser.parse_args()
    if args.dataset == 'mvtec':
        obj_list = ['capsule',
                     'bottle',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood'
                     ]
    elif args.dataset == 'mt':
        obj_list = ['mt']
    elif args.dataset == 'aitex':
        obj_list = ['aitex']


    with torch.cuda.device(args.gpu_id):
        test(obj_list, args)
        # test(obj_list, args.dataset, args.data_path, args.checkpoint_path, args.base_model_name, args.visualize, args.log_path)
