# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:17:18 2022

@author: daliana91
"""

import os
import numpy as np
import sys
from osgeo import gdal
import glob
import multiprocessing
import subprocess, signal
from sklearn import preprocessing as pp
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2
import time
from cl_cv_config import load_args
import pandas as pd
from sklearn import metrics
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    
    assert y_pred.size == y_true.size
    
    D = max(y_pred.max(), y_true.max()) + 1
    
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    
    #ind = linear_assignment(w.max() - w)
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def load_image(patch):
    # Read Image
    print (patch)
    gdal_header = gdal.Open(patch)
    # get array
    img = gdal_header.ReadAsArray()
    
    return img

def kill_child_processes(signum, frame):
    parent_id = os.getpid()
    ps_command = subprocess.Popen("ps -o pid --ppid %d --noheaders" % parent_id, shell=True, stdout=subprocess.PIPE)
    ps_output = ps_command.stdout.read()
    retcode = ps_command.wait()
    for pid_str in ps_output.strip().split("\n")[:-1]:
        os.kill(int(pid_str), signal.SIGTERM)
    sys.exit()
    
def load_image_SAR(list_dir):
    # Read Image
    files_tif = glob.glob(os.path.join(list_dir) + '/*.tif')
    for path in files_tif:
        print (path)
        gdal_header = gdal.Open(path)
        # get array
        img = gdal_header.ReadAsArray()
        img[np.isnan(img)] = 0
        img = 10.0**(img/10.0)     # from db to intensity
        img[img>1] = 1   # for SAR 
        img = img.reshape((img.shape[0],img.shape[1],1))
        try:
            stack_img = np.concatenate((stack_img,img), axis=-1)
        except:
            stack_img = img.copy()
            
    return stack_img


def create_stack_SAR(raster_path,start,end):
    list_dir = os.listdir(raster_path)
    list_dir.sort(key=lambda f: int((f.split('_')[0])))
    list_dir = [os.path.join(raster_path,f) for f in list_dir[start:end]]
    num_cores = multiprocessing.cpu_count()    
    
    pool = multiprocessing.Pool(num_cores)
    img_stack = pool.map(load_image_SAR, list_dir)    
    signal.signal(signal.SIGTERM, kill_child_processes)
    pool.close()
    pool.join()
    
    img_stack = np.array(img_stack)
    img_stack = np.rollaxis(img_stack,0,3) 
    img_stack = img_stack.reshape((img_stack.shape[0],img_stack.shape[1],
                                   img_stack.shape[2]*img_stack.shape[3]))
    return img_stack

def load_norm(image, coords = None, scaler_filename = None):
    # load images, create stack and normalize
    row,col,depth = image.shape
    if not os.path.isfile(scaler_filename):
        img_tmp = image[coords]
        scaler = pp.StandardScaler().fit(img_tmp)
        img_tmp = []                
        joblib.dump(scaler, scaler_filename) 
    else:
        print('Loading scaler')
        scaler = joblib.load(scaler_filename) 

    image = image.reshape((row*col,depth))
    image = scaler.transform(image)
    image = image.reshape((row,col,depth))    
    return image

def mapping_labels(labels_all, classes):
    # Mapping labels   
    lbl_tmp = labels_all.copy()
    labels2new_labels = dict((c, i) for i, c in enumerate(classes))
    new_labels2labels = dict((i, c) for i, c in enumerate(classes))
    for j in range(len(classes)):
        labels_all[lbl_tmp == classes[j]] = labels2new_labels[classes[j]]
    
    return labels_all

#slightly modified
def get_IoU(outputs, labels):
    EPS = 1e-8
    outputs = outputs.int()
    labels = labels.int()
    # Taken from: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0

    iou = (intersection + EPS) / (union + EPS)  # We smooth our devision to avoid 0/0

    return iou.mean()

def metrics_(save_dir, y_true, y_pred, args):
    
    name ='SimSiam_metrics'     
       
    #Accuracy metrics  
    y_true = y_true.cpu().numpy()
    if not args._eval_kmeans:
        y_pred = y_pred.cpu().numpy()
    else:
        y_pred= assigment_cluster(y_true, y_pred)

    acc =  np.round(accuracy_score(y_true, y_pred),5)
    recall = 100*np.round(recall_score(y_true, y_pred, average = None), 5)
    precision = 100*np.round(precision_score(y_true, y_pred,average = None), 5)
    f1score = 100*f1_score(y_true, y_pred, average = None)
    
    recall_total = 100*np.round(recall_score(y_true, y_pred, average = 'macro'), 5)
    precision_total = 100*np.round(precision_score(y_true, y_pred,average = 'macro'), 5)
    f1score_total = 100*np.round(f1_score(y_true, y_pred, average = 'macro'),5)
    
    print(f1score_total, recall_total, precision_total)
    
#    iou = 100*get_IoU(y_pred, y_true)
    
    print('acc sklearn:', acc)    
    f1 = np.asarray(f1score)
    print('f1: ',f1)
    recall = np.asarray(recall)
    print('Recall: ',recall)
    precision = np.asarray(precision)
    print('Precision:', precision)
    
    np.savetxt(save_dir + '/f1score.csv', f1, fmt="%d", delimiter=",")
    np.savetxt(save_dir + '/recall.csv', recall, fmt="%d", delimiter=",")
    np.savetxt(save_dir + '/precision.csv', precision, fmt="%d", delimiter=",")
     
    f = open('metrics.txt','w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
    f.write('{}\n{}\n{}\n'.format(f1score_total, recall_total, precision_total))
    f.close()

    sns.set(font_scale=3)
    classes = args.n_classes
    cm = confusion_matrix(y_true, y_pred, labels =classes)
    # Normalise
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    avr_acc = round(np.sum(cm.diagonal())/len(classes)*100,2)
    print('Accuracy from Confusion Matrix', avr_acc)
    
    ax= plt.subplot()
    sns.set(font_scale=2) # Adjust to fit
    sns.heatmap(cm, annot=True, ax = ax, fmt='.1%', cmap='Blues',cbar = False); #annot=True to annotate cells        
    # labels, title and ticks
    ax.set_xlabel('Predicted labels',fontsize=20)
    ax.set_ylabel('True labels',fontsize=20)
#    ax.set_title('Average class accuracy: ' + str(avr_acc))
    ax.xaxis.set_tick_params(labelsize=20); ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_ticklabels([str(i) for i in classes]); ax.yaxis.set_ticklabels([str(i) for i in classes])
    plt.savefig(os.path.join(save_dir,'conf_mat_{}.png'.format(str(name))), dpi = 300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    
    
def save_checkpoint(model, optimizer, args, epoch):
    print('\nModel Saving...')
    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join('checkpoints', 'checkpoint_pretrain_model.pth'))
    

def assigment_cluster(y_true, y_pred):
    
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1 
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    
    aux = np.zeros_like(y_pred)
    for i in range(len(ind)):        
        aux[y_pred==ind[i,0]] = ind[i,1]
    y_pred = aux
    
    return y_pred

def pca_visualization(features, pred, labels, epoch):
    
    time_start = time.time()
    
    features = features.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    features = np.array(features)
    labels = np.array(labels)
    
    pred = assigment_cluster(labels, pred)
    pca = PCA()
    pca_result = pca.fit_transform(features)
    
    # Building the color mapping       
    colours = {}
    colours[0] = 'Soybean'
    colours[1] = 'Cotton'
    colours[2] = 'Maize'
    colours[3] = 'Others'

    fig = plt.figure(figsize=(16,10))
    cvec = [colours[label] for label in pred] 
#    # set up the axes for the first plot
    ax = fig.add_subplot()
    df = pd.DataFrame({"x":pca_result[:, 0], "y":pca_result[:, 1], "Classes":cvec})
    sns.scatterplot(x="x", y="y", data=df, hue="Classes", legend="auto")
    ax.set_xlabel('pca_one')
    ax.set_ylabel('pca_two')    
    ax.grid(True)
#    ax.axis([-5, 20, -10, 10])   
    plt.savefig('pca_visualization_pred_{}.png'.format(str(epoch)), dpi=300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=(16,10))
    cvec = [colours[label] for label in labels] 
    # set up the axes for the first plot
    ax = fig.add_subplot()
    df = pd.DataFrame({"x":pca_result[:, 0], "y":pca_result[:, 1], "Classes":cvec})
    sns.scatterplot(x="x", y="y", data=df, hue="Classes", legend="full")
    ax.set_xlabel('pca_one')
    ax.set_ylabel('pca_two')    
    ax.grid(True)
#    ax.axis([-5, 20, -10, 10])   
    plt.savefig('pca_visualization_target_{}.png'.format(str(epoch)), dpi=300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
def tsne_visualization(features, pred, labels, epoch):
    
    '''Since t-SNE scales quadratically in the number 
    of objects N, its applicability is limited to data
    sets with only a few thousand input objects; beyond 
    that, learning becomes too slow to be practical
    (and the memory requirements become too large). It is 
    highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high.'''
    
    features = features.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    features = np.array(features)
    labels = np.array(labels)
    pred = np.array(pred)

    features = features[:int(len(features)*0.25)]
    pred = pred[:int(len(features)*0.25)]
    labels = labels[:int(len(labels)*0.25)]
    
    time_start = time.time()
#    feed the data into the t-SNE algorithm
    tsne_result = TSNE(n_components=2, perplexity=15, learning_rate=10, verbose=2).fit_transform(features)
    
    # Building the color mapping    
    colours = {}
    colours[0] = 'Soybean'
    colours[1] = 'Cotton'
    colours[2] = 'Maize'
    colours[3] = 'Others'
    
    fig = plt.figure(figsize=(16,10))
    cvec = [colours[label] for label in pred] 
#    # set up the axes for the first plot
    ax = fig.add_subplot()
    df = pd.DataFrame({"x":tsne_result[:, 0], "y":tsne_result[:, 1], "Classes":cvec})
    sns.scatterplot(x="x", y="y", data=df, hue="Classes", legend="auto")
    ax.set_xlabel('tsne_one')
    ax.set_ylabel('tsne_two')    
    ax.grid(True)
#    ax.axis([-5, 20, -10, 10])   
    plt.savefig('tsne_visualization_pred_{}.png'.format(str(epoch)), dpi=300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=(16,10))
    cvec = [colours[label] for label in labels] 
    # set up the axes for the first plot
    ax = fig.add_subplot()
    df = pd.DataFrame({"x":tsne_result[:, 0], "y":tsne_result[:, 1], "Classes":cvec})
    sns.scatterplot(x="x", y="y", data=df, hue="Classes", legend="full")
    ax.set_xlabel('tsne_one')
    ax.set_ylabel('tsne_two')    
    ax.grid(True)
#    ax.axis([-5, 20, -10, 10])   
    plt.savefig('tsne_visualization_target_{}.png'.format(str(epoch)), dpi=300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
def pca_tsne_visualization(features, pred, labels, epoch):
    
    '''Since t-SNE scales quadratically in the number 
    of objects N, its applicability is limited to data
    sets with only a few thousand input objects; beyond 
    that, learning becomes too slow to be practical
    (and the memory requirements become too large). It is 
    highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high.'''
    
    features = features.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    features = np.array(features)
    labels = np.array(labels)
    pred = np.array(pred)

    features = features[:int(len(features)*0.25)]
    pred = pred[:int(len(features)*0.25)]
    labels = labels[:int(len(labels)*0.25)]
    
    time_start = time.time()
    #    New dataset containing the fifty dimensions generated by the PCA reduction algorithm
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(features)
    
    #    feed the data into the t-SNE algorithm
    tsne_result = TSNE(n_components=2, perplexity=15, learning_rate=10, verbose=2).fit_transform(pca_result_50)
    
    # Building the color mapping    
    colours = {}
    colours[0] = 'Soybean'
    colours[1] = 'Cotton'
    colours[2] = 'Maize'
    colours[3] = 'Others'
    
    fig = plt.figure(figsize=(16,10))
    cvec = [colours[label] for label in pred] 
#    # set up the axes for the first plot
    ax = fig.add_subplot()
    df = pd.DataFrame({"x":tsne_result[:, 0], "y":tsne_result[:, 1], "Classes":cvec})
    sns.scatterplot(x="x", y="y", data=df, hue="Classes", legend="auto")
    ax.set_xlabel('pca_tsne_one')
    ax.set_ylabel('pca_tsne_two')    
    ax.grid(True)
#    ax.axis([-5, 20, -10, 10])   
    plt.savefig('pca_tsne_visualization_pred_{}.png'.format(str(epoch)), dpi=300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=(16,10))
    cvec = [colours[label] for label in labels] 
    # set up the axes for the first plot
    ax = fig.add_subplot()
    df = pd.DataFrame({"x":tsne_result[:, 0], "y":tsne_result[:, 1], "Classes":cvec})
    sns.scatterplot(x="x", y="y", data=df, hue="Classes", legend="full")
    ax.set_xlabel('pca_tsne_one')
    ax.set_ylabel('pca_tsne_two')    
    ax.grid(True)
#    ax.axis([-5, 20, -10, 10])   
    plt.savefig('pca_tsne_visualization_target_{}.png'.format(str(epoch)), dpi=300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    
def Load_data(args, idx = None):
    # List of the folders
    labels_list = glob.glob(args.gt_dir + '/*.tif')
    labels_list.sort()    
    #    Loading the labels
    label_seq1 = load_image(labels_list[args.date_label1-1])
    label_seq2 = load_image(labels_list[args.date_label2-1])
    
    # Loading the mask of train-test
    mask_img = load_image(args.mask_dir)
    if idx is None:
        coords = np.where(mask_img!=0)
    else:
        coords = np.where(mask_img==idx)
                
    # Load data and normalize
    scaler_seq1 = os.path.join(args.model_dir,"scaler_seq1.save")
    data_seq1 = create_stack_SAR(args.img_sar, args.img_seq1[0]-1, args.img_seq1[1])
    data_seq1 = load_norm(data_seq1, coords, scaler_seq1)
#    channels = data_seq1.shape[-1]
#    print(channels)
    scaler_seq2 = os.path.join(args.model_dir,"scaler_seq2.save")
    data_seq2 = create_stack_SAR(args.img_sar,args.img_seq2[0]-1, args.img_seq2[1])
    data_seq2 = load_norm(data_seq2, coords, scaler_seq2)
    
    data_all = np.concatenate((data_seq1, data_seq2), axis=-1)
    labels_all = np.concatenate((label_seq1[:, :, np.newaxis], label_seq2[:, :, np.newaxis]), axis=-1)
    
    return data_all, labels_all, mask_img


def gray2rgb(image):
    """
    Funtion to convert classes values from 0,1,3,4 to rgb values
    """
    row,col = image.shape
    image = image.reshape((row*col))
    rgb_output = np.zeros((row*col, 3))
    rgb_map = [[0,0,255],[0,255,0],[0,255,255],[255,255,0],[255,255,255]]
    for j in np.unique(image):
        rgb_output[image==j] = np.array(rgb_map[j])
    
    rgb_output = rgb_output.reshape((row,col,3))  
    rgb_output = cv2.cvtColor(rgb_output.astype('uint8'),cv2.COLOR_BGR2RGB)
    return rgb_output 

def visualization(args, predict_labels):
    
    args = load_args()
    # Loading the mask of train-test
    data, label, mask_img = Load_data(args, idx=1)
    
    label[mask_img!=2] = 0       
    predict_labels[mask_img!=2] = 0

    predict_see = gray2rgb(np.uint(predict_labels))
    reference_see = gray2rgb(np.uint(label))
    cv2.imwrite('./predict_total_.tiff', predict_see)
    cv2.imwrite('./test_reference.tiff', reference_see)

# Function to return the latent variables for an input image batch
def get_latent_var(train_data, db_model, args, batch_size=1024):
 
  train_loader = DataLoader(train_data, batch_size= batch_size, shuffle=False, num_workers=args.num_workers)
#   device = torch.device("cpu")
#   db_model.to(device)

  db_model.eval()
  with torch.no_grad():
    for img, x1, x2, target in train_loader:
        if args.cuda:
            img, x1 = img.cuda(), x2.cuda()
        _, _ ,latent_var = db_model(img,x1)
        
        try:
            latent_var_ = torch.cat((latent_var_, latent_var))
        except NameError:
            latent_var_ = latent_var

    print('Latent_var tensor:', latent_var_.shape)
     
#   for start_ind in range(0, N, batch_size):
#     end_ind = min(start_ind+batch_size, N+1)
#     batch = (images[start_ind:end_ind])
#     _, batch_mu, _ = db_model.encode(batch)
#     mu[start_ind:end_ind] = batch_mu
 
  return latent_var_

      
def pixels_class(label, data, flag=True):
    l, c = np.unique(label, return_counts=True)
    
    # Choosing the same proportion of pixels for each class  
    # 5000 for 4 classes
    for i in range(len(l)):
        index = (label==l[i])
        
        q = label[index][0:1000]
        
        if flag:
            p = data[index][0:1000, :] 
        
        if i==0:
            y_true_mat = q
            if flag:
                data_mat = p
        else:
            y_true_mat = np.concatenate((y_true_mat, q), axis= 0 )
            if flag:
                data_mat = np.concatenate((data_mat, p), axis = 0)
    if flag:
        return data_mat, y_true_mat
   
    return y_true_mat  
def plotme(data, target, clear, epoch):
    if clear:
        plt.clf() # Draw new figure each time

    data = data.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    data, target = pixels_class(target, data)
    print(data.shape)
    print(target.shape)

    # pca = PCA(n_components=2).fit(data)
    # data = pca.transform(data)
    data = TSNE(n_components=2, perplexity=50, early_exaggeration =12, learning_rate=10, verbose=2).fit_transform(data)
       
    class_1 = data[target==0]
    class_2 = data[target==1]
    class_3 = data[target==2]
    class_4 = data[target==3]

    class_5 = data[target==4]
    class_6 = data[target==5]
    class_7 = data[target==6]
    class_8 = data[target==7]
    class_9 = data[target==8]
    class_10 = data[target==9]
    class_11 = data[target==10]

    
    plt.figure(figsize=(16,10),facecolor='white')

    c1 = plt.scatter(class_1[:,0], class_1[:,1], s=50, c='r',alpha=0.5, marker='o')	
    c2 = plt.scatter(class_2[:,0], class_2[:,1], s=50, c='g', alpha=0.5, marker='o')
    c3 = plt.scatter(class_3[:,0], class_3[:,1], s=50, c='b', alpha=0.5, marker='o')
    c4 = plt.scatter(class_4[:,0], class_4[:,1], s=50, c='y', alpha=0.5, marker='o')

    c5 = plt.scatter(class_5[:,0], class_5[:,1], s=50, c='c', alpha=0.5, marker='o')
    c6 = plt.scatter(class_6[:,0], class_6[:,1], s=50, c='m', alpha=0.5, marker='o')
    c7 = plt.scatter(class_7[:,0], class_7[:,1], s=50, c='pink', alpha=0.5, marker='o')
    c8 = plt.scatter(class_8[:,0], class_8[:,1], s=50, c='gray', alpha=0.5, marker='o')
    c9 = plt.scatter(class_9[:,0], class_9[:,1], s=50, c='olive', alpha=0.5, marker='o')
    c10 = plt.scatter(class_10[:,0], class_10[:,1], s=50, c='peru', alpha=0.5, marker='o')
    c11 = plt.scatter(class_11[:,0], class_11[:,1], s=50, c='indigo', alpha=0.5, marker='o')

    plt.axis('off')
#    plt.title('Dataset with 4 clusters')
    
    plt.savefig('SSL_visualization{}.png'.format(str(epoch)), dpi=300)
    plt.savefig('SSL_visualization{}.pdf'.format(str(epoch)), dpi=300)
