# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:39:49 2021

@author: daliana91
"""

import logging
import os
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cl_cv_config import load_args
from model import Model, DownStreamModel
from torch.utils.data import DataLoader
from preprocess import Load_data, CustomSampler, train_test, SimSiamDataset, train_subset
from cl_cv_utils import plotme, save_checkpoint, metrics_, tsne_visualization, pca_visualization, visualization, pca_tsne_visualization, cluster_acc
import matplotlib.pyplot as plt
import datetime
import sys

from icecream import ic

def pre_train(epoch, train_loader, model, optimizer, args):
    model.train()
    losses, step = 0., 0.
    for img, x1, x2, target in train_loader:
        if args.cuda:
            x1, x2 = x1.cuda(), x2.cuda()
        d1, d2, _ = model(x1, x2)
        """In PyTorch, we need to set the gradients to zero before starting to
        do backpropragation because PyTorch accumulates the gradients on
        subsequent backward passes."""
        optimizer.zero_grad()
        loss = d1 + d2
        loss.backward()
        optimizer.step()
        losses += loss.item()
        step += 1
        if step > args.samples_per_epoch/args.batch_size: break      

    print('[Epoch: {0:4d}, loss: {1:.3f}'.format(epoch, losses / step))
    
    return losses / step


def _train(epoch, train_loader, model, optimizer, criterion, args):
    model.train()

    losses, acc, step, total = 0., 0., 0., 0.
    for data, target in train_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        logits = model(data)
        optimizer.zero_grad()
        loss = criterion(logits, target)
        loss.backward()
        losses += loss.item()
        optimizer.step()

        pred = F.softmax(logits, dim=-1).max(-1)[1]
        acc += pred.eq(target).sum().item()

        step += 1
        total += target.size(0)
        samples_per_epoch = 30000
        if step > samples_per_epoch/args.batch_size: break      

    print('[Down Task Train Epoch: {0:4d}], loss: {1:.3f}, acc: {2:.3f}'.format(epoch, losses / step, acc / total * 100.))

def _eval(epoch, test_loader, model, criterion, args):
    
    model.eval()    
    losses, acc, step, total = 0., 0., 0., 0.
    # Disabling gradient calculation for inference, 
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
                
            logits = model(data)
            
            loss = criterion(logits, target)
            losses += loss.item()
            pred = F.softmax(logits, dim=-1).max(-1)[1]
            
            # lv = lv.cpu().numpy()
            acc += pred.eq(target).sum().item()

            step += 1
            total += target.size(0)

            try:
                logits_ = torch.cat((logits_, logits))
                pred_ = torch.cat((pred_, pred))
                target_ = torch.cat((target_, target))
         
            except NameError:
                logits_ = logits
                pred_ = pred
                target_= target

    
    
#        print(pred_.size())
        #Accuracy metrics
        print('[Down Task Test Epoch: {0:4d}], loss: {1:.3f}, acc: {2:.3f}'.format(epoch, losses / step, acc / total * 100.))

        return acc / total * 100, logits_, pred_, target_
def _eval_kmeans(test_loader, model, args):
    model.eval()
    c = 0
    acc = 0
    # Disabling gradient calculation for inference,
    with torch.no_grad():
        for data, target in test_loader:
#            print(np.unique(target))
            if args.cuda:
                data, target = data.cuda(), target.cuda()
          
            logits = model(data)

            try:
                logits_ = torch.cat((logits_, logits))
                target_ = torch.cat((target_, target))
            except NameError:
                logits_ = logits
                target_ = target
            
            c += 1
            if not c % 10000: print(c)   
    
    print(logits_.size())

    from sklearn.cluster import KMeans, MiniBatchKMeans
    kmeans = KMeans(n_clusters=args.n_classes,init='k-means++', n_init=10, n_jobs=1)
    pred_ = kmeans.fit_predict(logits_.cpu().numpy())

    # from sklearn.neighbors import KNeighborsClassifier
    # neigh = KNeighborsClassifier()
    # neigh.fit(logits_.cpu().numpy(), target_.cpu().numpy())
    # pred_=neigh.predict(logits_.cpu().numpy())
    
    acc = cluster_acc(target_.cpu().numpy(), pred_)
    print('Accuracy: ', acc * 100)
    
    return acc, logits_, pred_, target_


def train_eval_down_task(down_model, down_train_loader, down_test_loader, args):
    down_optimizer = optim.SGD(down_model.parameters(), lr=args.down_lr, weight_decay=args.weight_decay, momentum=args.momentum)
    down_criterion = nn.CrossEntropyLoss()
    down_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(down_optimizer, T_max=args.down_epochs)
    if not args._eval_kmeans:
        if args.train_downtask:
            # down_train_loader has img1, target    
            for epoch in range(1, args.down_epochs + 1):
                """Here I train my train set with output(img1,target) follow the downstream 
                model using a crossentropy loss. Afterthat, I also evaluate the train set 
                with the softmax, argmax prediction"""
                _train(epoch, down_train_loader, down_model, down_optimizer, down_criterion, args)         
                down_lr_scheduler.step()               
                if epoch % args.print_intervals == 0:
                    acc, logits_, pred_, target_ = _eval(epoch, down_test_loader, down_model, down_criterion, args)
                    plotme(logits_, target_, True, epoch)
            """Here I evaluate the model trained using the test loader """        
        else:
                epoch=0
        acc, logits_, pred_, target_ = _eval(epoch, down_test_loader, down_model, down_criterion, args) 
    else:
        acc, logits_, pred_, target_ = _eval_kmeans(down_test_loader, down_model, args) 
     
    return acc, logits_, pred_, target_

def main(args):
    
    #################################### Load data as an iterable ###################################   
    data, labels, mask_img = Load_data(args)

    #Data for training
    labels_train, index_train, coords_train = train_test(labels, mask_img, args, idx=1 )
    
    if args.samples_per_epoch:
        percent =5
        print('Train with %(percent)d percent of samples in the Pretext task'% {"percent": percent})
        index_train_sub = train_subset(labels_train, index_train.copy(), coords_train, percent=percent)
    
    train_data = SimSiamDataset(args, data, labels_train, index_train_sub, coords_train)
    #Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    #Data for testing
    labels_test, index_test, coords_test  = train_test(labels, mask_img, args, idx=2)
    test_data = SimSiamDataset(args, data, labels_test, index_test, coords_test,mode='test')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    down_percent = 100
    print('Train with %(percent)d percent of samples in the Downstreamtask task'% {"percent": down_percent})
    index_train_sub_down = train_subset(labels_train, index_train.copy(), coords_train, percent=down_percent)
    down_train_data = SimSiamDataset(args, data, labels_train, index_train_sub_down, coords_train, downstream=True)
    down_train_loader = DataLoader(down_train_data, batch_size=args.down_batch_size, shuffle=True, num_workers=args.num_workers)

    # ic(len(index_train_sub))
    # ic(len(down_train_data))
    # ic(len(down_train_loader))

    down_test_data = SimSiamDataset(args, data, labels_test, index_test, coords_test, mode='test', downstream=True)
    down_test_loader = DataLoader(down_test_data, batch_size=args.down_batch_size, shuffle=False, num_workers=args.num_workers)
        
    # ic(len(down_test_data))
    # ic(len(down_test_loader))
    # ic(len(index_test))
    #################################### Load data as an iterable ################################### 
     
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    
    
    model = Model(args)
    down_model = DownStreamModel(args)

    if args.cuda:
        model = model.cuda()
        down_model = down_model.cuda()

    if args.pretrain:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        train_losses, epoch_list, acc_list, interval_list = [], [], [], []
        for epoch in range(1, args.epochs + 1):
            
            a = datetime.datetime.now().replace(microsecond=0)
            if args.CustomSampler:
                if epoch==1: print('Train with the debiasing mode')
                
                index_train_sub = CustomSampler(train_data, index_train_sub, model, args, labels_train, coords_train)
             
            if args.samples_per_epoch:   
                if epoch % 20 == 0:
                    index_train_sub = train_subset(labels_train, index_train.copy(), coords_train, percent=percent)
                    if args.CustomSampler:
                        index_train_sub = CustomSampler(train_data, index_train_sub, model, args, labels_train, coords_train)
                   
                train_data = SimSiamDataset(args, data, labels_train, index_train_sub, coords_train)
                #Data loader. Combines a dataset and the debiasing sampler, and provides an iterable over the given dataset.
                train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle = True, num_workers=args.num_workers)

                # ic(len(index_train_sub))
                # ic(len(train_data))   

            # Here I pre-trained the two branch of the Syamese Networks with their encoder
            # proyeccions and predictions using the cosine learning rate strategy.            
            train_loss = pre_train(epoch, train_loader, model, optimizer, args)
            # Here when I reached to 100 epochs I used the down_model to evaluate the down_train_loader       

            if epoch % args.print_intervals == 0:
                save_checkpoint(model, optimizer, args, epoch)
                args.down_epochs = 1
                acc, logits_, pred_, target_= train_eval_down_task(down_model, down_train_loader, down_test_loader, args)
                interval_list.append(epoch)
                acc_list.append(acc)
                
            lr_scheduler.step()
            train_losses.append(train_loss)
            epoch_list.append(epoch)
            
            print(' Cur lr: {0:.5f}'.format(lr_scheduler.get_last_lr()[0]))
            b = datetime.datetime.now()
            print("Time to train one epoch", b-a)

        plt.plot(epoch_list, train_losses)
        plt.xlabel('epochs') 
        plt.ylabel('training loss')
        plt.grid()
        plt.savefig('pretrain_curve.png', dpi=300)
        
        plt.plot(interval_list, acc_list)
        plt.xlabel('epochs') 
        plt.ylabel('pre_train test acc')
        plt.grid()
        plt.savefig('acc_pre_test_curve.png', dpi=300)
    
    else:
        
       _, logits_, pred_, target_ = train_eval_down_task(down_model, down_train_loader, down_test_loader, args)
    #    print(logits_.shape)
    #    print(softmax_.shape)
    #    print(softmax_.cpu().numpy().mean(0).shape)
       
    #    metrics_(args.save_dir, target_, pred_, args)
       plotme(logits_, target_, True, epoch)

    #    checkpoints = 'checkpoints/checkpoint_pretrain_model_4cluster_0.8margin.pth'
    #    model = Model(args,checkpoints)
    #    down_model_db = DownStreamModel(args, checkpoints)

    #    _, logits_, pred_, target_, softmax_db = train_eval_down_task(down_model, down_train_loader, down_test_loader, args)

    #    keys = ["Soybean", "Maize", "Cotton", "Others"]
    #    xx = np.arange(len(keys))
    #    plt.bar(xx, softmax_.cpu().numpy().mean(0), width=0.2, label="Standard CNN")
    #    plt.bar(xx+0.2,softmax_db.cpu().numpy().mean(0), width=0.2, label="DB-VAE")
    #    plt.xticks(xx, keys); 
    #    plt.title("Network predictions on test dataset")
    #    plt.ylabel("Probability"); plt.legend(loc="upper left")
    #    plt.savefig('class_prob.png', dpi=300)

if __name__ == '__main__':

    args = load_args()
    
    # Create the input data pipeline
    logging.info("Creating the datasets...")
    assert os.path.isdir(args.img_sar), "Couldn't find the dataset at {}".format(args.img_sar)
    assert os.path.isdir(args.gt_dir), "Couldn't find the dataset at {}".format(args.gt_dir)
    
    # get coordinates of training=1 or test=2 samples 
    
    main(args)
