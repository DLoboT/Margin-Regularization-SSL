# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:23:17 2021

@author: daliana91
"""

import argparse


def load_args():
    parser = argparse.ArgumentParser()

    # Pre training    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./experiments/v1/',
                        help="Experiment directory containing params.json")
    parser.add_argument('--restore_from', default=None,
                        help="Optional, file containing weights to reload before training")
    
    parser.add_argument('--gt_dir', default='./Campo_Verde/Labels_uint8/', help="Directory with the gt dataset")
    parser.add_argument('--mask_dir', default='./Campo_Verde/TrainTestMask_50_50.tif', help="Directory with the mask")
   
    # SAR data
    parser.add_argument('--img_sar', default='./Campo_Verde/Rasters/', help="Directory with theimages dataset")
    parser.add_argument('--img_seq1', default=[1,7], help="SAR images for sequence 1")
    parser.add_argument('--img_seq2', default=[8,14], help="Optical image for sequence 2")
    parser.add_argument('--date_label1', default=4, help="Label image for sequence 1")
    parser.add_argument('--date_label2', default=11, help="Label image for sequence 2")

    #Optical data
    # parser.add_argument('--img_opt', default='./Campo_Verde/Cuts_Landsat/', help="Directory with the images dataset")
    # parser.add_argument('--img_seq1', default=[1,0], help="SAR images for sequence 1")
    # parser.add_argument('--img_seq2', default=[3,0], help="Optical image for sequence 2")
    # parser.add_argument('--date_label1', default=3, help="Label image for sequence 1")
    # parser.add_argument('--date_label2', default=11, help="Label image for sequence 2")

    parser.add_argument('--samples_per_epoch',  type=bool, default=True)
    parser.add_argument('--save_dir', default='./experiments/v1')
    
    #Pretext Task
    parser.add_argument('--img_size', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_classes', type=int, default=11)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--checkpoints', type=str, default = 'checkpoints/checkpoint_pretrain_model_10cluster.pth')
    parser.add_argument('--pretrain', type=bool, default= False)
    parser.add_argument('--CustomSampler', type=bool, default= True)
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--print_intervals', type=int, default=2)
    parser.add_argument('--similarity', type=str, default='cosine')
    #'checkpoints/checkpoint_pretrain_model.pth'
    #'checkpoints/checkpoint_pretrain_model_4cluster_0.8margin.pth'
    #'checkpoints/checkpoint_pretrain_model_4cluster.pth'
    # Network
    parser.add_argument('--proj_hidden', type=int, default=2048)#2048
    parser.add_argument('--proj_out', type=int, default=2048)#2048
    parser.add_argument('--pred_hidden', type=int, default=512)#512
    parser.add_argument('--pred_out', type=int, default=2048)#2048

    # DownStream Task
    parser.add_argument('--train_downtask', type=bool, default=True)
    parser.add_argument('--down_lr', type=float, default=0.001)
    parser.add_argument('--down_epochs', type=int, default=10)
    parser.add_argument('--down_batch_size', type=int, default=512)
    parser.add_argument('--_eval_kmeans', type=bool, default= False)
    parser.add_argument('--supervised', type=bool, default= False)

    args = parser.parse_args()

    return args
