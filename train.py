import torch
from data_loader import MVTecDRAEMTrainDataset,PathList,TrainDataset,TestDataset
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork, Discriminator
from loss import FocalLoss, SSIM
import os
import math
from test_DRAEM import test
from itertools import cycle
import torch.nn as nn

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(obj_names, args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    base_name = str(args.model) + '_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"
    if args.set_t is not None:
        base_name += "set_" + str(args.set_t)+'_'
    if args.anomaly_source_path is not None and "Magnetic_Tile" in args.anomaly_source_path:
        base_name += "adsource_self_"
    if args.coslr is True:
        base_name += "coslr_"
    if args.ad != 0:
        base_name += "ad" + str(args.ad) + "_"
    if args.adv is not None:
        base_name += "adv_" + str(args.adv) + "_"

    args.base_model_name = base_name

    for obj_name in obj_names:
        run_name = base_name + obj_name+'_'

        visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path , run_name+"/"))

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        discriminator = Discriminator(in_channels=3, base_width=64)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)  #, device_ids=range(torch.cuda.device_count() - 1)
            model_seg = nn.DataParallel(model_seg)
            discriminator = nn.DataParallel(discriminator)


        model.cuda()
        model.apply(weights_init)

        model_seg.cuda()
        model_seg.apply(weights_init)

        discriminator.cuda()
        # discriminator.apply(weights_init)

        # if args.adv is True:

        loss_bce = torch.nn.BCELoss()
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.9))
        scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.8)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr},
                                      {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs * 0.8, args.epochs * 0.9], gamma=0.2,
                                                   last_epoch=-1)






        root_dir = args.data_path
        if args.dataset == 'mvtec':
            root_dir += obj_name

        if args.model == "DRAEM":
            train_dataset = MVTecDRAEMTrainDataset(args, root_dir, resize_shape=[256, 256])
            train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=16)
            test_dataset = None
        elif args.model == "SAD":
            Path = PathList(args, root_dir)
            train_pos_path, train_neg_path, test_paths = Path.get_path(args.ad)
            train_dataset = TrainDataset(args, train_pos_path, train_neg_path, resize_shape=[256, 256])
            train_neg_dataset = TestDataset(args, train_neg_path, resize_shape=[256, 256])
            test_dataset = TestDataset(args, test_paths, resize_shape=[256, 256])
            train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=16)
            train_neg_loader = DataLoader(train_neg_dataset, batch_size=args.bs//2, shuffle=True, num_workers=16)

        # if args.coslr is True:
        #     n_iter = math.ceil(len(train_dataset) / args.bs)
        #     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr * 2, total_steps=args.epochs * n_iter)

        auroc_best = [0, 0, 0, 0, 0]
        all_best = [[0, 0], [0, 0], [0, 0], [0, 0]]
        iter = 0

        #
        # for epoch in range(args.epochs):
        #     l2 = ssim = segment = 0
        #     model.train()
        #     model_seg.train()
        #
        #     # if args.set_t == "epoch_linear" or "epoch_random":
        #     #     train_dataset.set_T(epoch + 1, args.epochs)
        #
        #     for i_batch, sample_batched in enumerate(train_loader):
        #         # if args.set_t == "batch_linear" or "batch_random":
        #         #     train_dataset.set_T(i_batch+1, n_iter+1)
        #
        #         pos_batch = sample_batched["image"].cuda()
        #         aug_batch = sample_batched["augmented_image"].cuda()
        #         aug_mask = sample_batched["anomaly_mask"].cuda()
        #         # target_neg1 = sample_batched["target_neg1"].cuda()
        #         # target_neg2 = sample_batched["target_neg2"].cuda()
        #         # target_mask1 = sample_batched["target_mask1"].cuda()
        #         # target_mask2 = sample_batched["target_mask2"].cuda()
        #
        #         aug_rec = model(aug_batch)
        #         joined_in = torch.cat((aug_rec.detach(), aug_batch), dim=1)
        #
        #         out_mask = model_seg(joined_in)
        #         out_mask_sm = torch.softmax(out_mask, dim=1)
        #
        #         l2_loss = loss_l2(aug_rec,pos_batch)
        #         ssim_loss = loss_ssim(aug_rec, pos_batch)
        #
        #         segment_loss = loss_focal(out_mask_sm, aug_mask)
        #         loss = l2_loss + ssim_loss + segment_loss
        #
        #         l2 += l2_loss
        #         ssim += ssim_loss
        #         segment += segment_loss
        #
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        torch.cuda.empty_cache()

        for epoch in range(args.epochs):
            l2 = ssim = segment = dis = gen = 0
            model.train()
            model_seg.train()
            discriminator.train()
            for (batch_pos, batch_neg) in zip(train_loader, cycle(train_neg_loader)):
                pos_batch = batch_pos["image"].cuda()
                aug_batch = batch_pos["augmented_image"].cuda()
                aug_mask = batch_pos["anomaly_mask"].cuda()
                aug_label = batch_pos["has_anomaly"].cuda()    # 0正常，1伪异常
                # aug_label = aug_label.reshape(aug_label.shape[0]).cuda()
                neg_batch = batch_neg["image"].cuda()
                neg_label = batch_neg["has_anomaly"].cuda()    # 全1异常
                # neg_label = neg_label.reshape(neg_label.shape[0]).cuda()

                optimizer_D.zero_grad()
                pos_loss = loss_l2(discriminator(pos_batch), torch.ones([pos_batch.size(0), 1]).cuda())
                aug_loss = loss_l2(discriminator(aug_batch), 1-aug_label)
                neg_loss = loss_l2(discriminator(neg_batch), 1-neg_label)

                aug_rec = model(aug_batch)
                rec = aug_rec.detach()

                fake_loss = loss_l2(discriminator(rec), torch.zeros([rec.size(0),1]).cuda())
                dis_loss = (pos_loss + aug_loss + neg_loss + fake_loss)/3
                dis_loss.backward()
                optimizer_D.step()

                gen_loss = loss_l2(discriminator(aug_rec), 0.5 * torch.ones([aug_rec.size(0), 1]).cuda())

                joined_in = torch.cat((rec, aug_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                l2_loss = loss_l2(aug_rec, pos_batch)
                ssim_loss = loss_ssim(aug_rec, pos_batch)

                segment_loss = loss_focal(out_mask_sm, aug_mask)
                loss = l2_loss + ssim_loss + segment_loss + gen_loss * 0.1

                l2 += l2_loss
                ssim += ssim_loss
                segment += segment_loss
                dis += dis_loss
                gen += gen_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.visualize and iter % 200 == 0:
                    # print("log_loss", iter)
                    visualizer.plot_loss(l2_loss.item(), iter, loss_name='l2_loss')
                    visualizer.plot_loss(ssim_loss.item(), iter, loss_name='ssim_loss')
                    visualizer.plot_loss(segment_loss.item(), iter, loss_name='segment_loss')
                    if args.adv is not None:
                        visualizer.plot_loss(dis_loss.item(), iter, loss_name='dis_loss')
                        visualizer.plot_loss(gen_loss.item(), iter, loss_name='gen_loss')
                if args.visualize and iter % 400 == 0:
                    # print("log_img", iter)
                    t_mask = out_mask_sm[:, 1:, :, :]   # (bs,2,256,256) 在2通道上做softmax，取1值的代表异常mask
                    visualizer.visualize_image_batch(aug_batch, iter, image_name='batch_augmented')
                    visualizer.visualize_image_batch(pos_batch, iter, image_name='batch_recon_target')
                    visualizer.visualize_image_batch(aug_rec, iter, image_name='batch_recon_out')
                    visualizer.visualize_image_batch(aug_mask, iter, image_name='mask_target')
                    visualizer.visualize_image_batch(t_mask, iter, image_name='mask_out')
                    # visualizer.visualize_image_batch(target_neg1, iter, image_name='target_neg1')
                    # visualizer.visualize_image_batch(target_neg2, iter, image_name='target_neg2')
                    # visualizer.visualize_image_batch(target_mask1, iter, image_name='target_mask1')
                    # visualizer.visualize_image_batch(target_mask2, iter, image_name='target_mask2')

                iter += 1

            scheduler.step()
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
            torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))

            logs = 'Epoch[{}/{}], '.format(epoch, args.epochs) + \
                   'l2_loss: {:.8f}, '.format(l2/len(train_dataset)) + \
                   'ssim_loss: {:.8f}, '.format(ssim/len(train_dataset)) + \
                   'segment_loss: {:.8f},'.format(segment/len(train_dataset))

            if args.adv is not None:
                # torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_path, run_name + "_dis.pckl"))
                logs += 'dis_loss: {:.8f},'.format(dis/(len(train_dataset)+len(train_neg_dataset))) + 'gen_loss: {:.8f}'.format(gen/len(train_dataset))

            print(logs)

            log_path = "./outputs/" + str(args.dataset) + "/" + str(obj_name) + "_result.log"
            if not os.path.exists(os.path.dirname(log_path)):
                os.makedirs(os.path.dirname(log_path))
            with open(log_path, 'a+') as file:
                    file.write(logs + '\n')

            with torch.cuda.device(args.gpu_id):
                res = test([obj_name], args, epoch, test_dataset)   # auroc, auroc_pixel, ap, ap_pixel
                if res[0] >= auroc_best[0]:
                    auroc_best = res + [epoch]
                for i in range(4):
                    if res[i] >= all_best[i][0]:
                        all_best[i] = [res[i], epoch]


        logs = "best_result: auc, auc_pixel, ap, ap_pixel \n" + \
               "best_auroc" + str(auroc_best) + '\n' + \
               "best_all:" + str(all_best) + '\n' + \
               "=============================="
        print(logs)
        with open(log_path, 'a+') as file:
            file.write(logs + '\n')


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    # parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--gpu_id', type=int, nargs='+', help='gpu list', required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=False)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--dataset', action='store', type=str, required=True)
    parser.add_argument('--obj_id', action='store', type=int, default=-1, required=False)
    parser.add_argument('--set_t', action='store', type=str, default=None, required=False)
    parser.add_argument('--base_model_name', action='store', type=str, required=False)
    parser.add_argument('--coslr', action='store_true', required=False)
    parser.add_argument('--ad', action='store', type=float, default=0, required=False)
    parser.add_argument('--model', action='store', type=str, default="SAD", required=True)
    parser.add_argument('--adv', action='store', type=str, default=None, required=False)

    args = parser.parse_args()

    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]
    if args.dataset == 'mvtec':
        if int(args.obj_id) == -1:
            obj_list = [
                         'capsule',
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
            picked_classes = obj_list
        else:
            picked_classes = obj_batch[int(args.obj_id)]
    elif args.dataset == 'mt':
        picked_classes = ['mt']
    elif args.dataset == 'aitex':
        picked_classes = ['aitex']

    gpu_list = list(args.gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(id) for id in gpu_list)

    with torch.cuda.device(gpu_list):
        train_on_device(picked_classes, args)



