from graphcnn.helper import *
import scipy.io
import numpy as np
import datetime
import graphcnn.setup.helper
import graphcnn.setup as setup
#import h5py
from skmultilearn.problem_transform import LabelPowerset # to help determine class_weights


def load_ucmerced_dataset():

    dataset = scipy.io.loadmat('dataset/newtotpatt.mat')#dataset1.mat, totgra.mat
    dataset = dataset['newtotpatt'] #dataset1, totgra
    #dataset = np.transpose(dataset);
    edges = np.squeeze(dataset['edges']) #adjacecny matrix
    index = np.squeeze(dataset['index'])  # image index to keep track
    classes = np.squeeze(dataset['class'])  #image class number to keep track
    #'''
    #loading features in which NaN values have been replaced
    features1 = scipy.io.loadmat('dataset/features1n.mat') #normfeatures1.mat
    features1 = features1['features1n']
    features1 = np.squeeze(features1['val'])
    features2 = scipy.io.loadmat('dataset/features2n.mat') #normfeatures1.mat
    features2 = features2['features2n']
    features2 = np.squeeze(features2['val'])
    features3 = scipy.io.loadmat('dataset/features3n.mat') #normfeatures1.mat
    features3 = features3['features3n']
    features3 = np.squeeze(features3['val'])
    features = np.concatenate((features1,features2))
    features = np.concatenate((features,features3))
    #features = np.squeeze(dataset(features))
    #features = features['val']
    #features = features[0]
    print type(features[0])
    print features.shape
    for i in range(0,len(features)):
        if np.isnan(features[i]).any() == True:
            print('features %d have NaN:'% i,np.isnan(features[i]).any())

    '''
    f = h5py.File('dataset/pattfeatures.mat','r')#dataset1.mat
    test = f['features'] #dataset1
    #dataset = np.transpose(dataset);
    test = np.squeeze(test['val']) #adjacecny matrix

    st=test
    for i in range(30400):
       st[i] = test[i]
    features = f[st]
    #index = np.squeeze(dataset['index'][:])  # image index to keep track
    #val = np.array(val)
    #index = np.array(index)
    #classes = np.squeeze(dataset['class'])  #image class number to keep track
    


    #features = features['val']
    #features = test
    #features = f[features]
    #str1 = ''.join(features(i) for i in features[:])
    #features = np.array([features])
    #features[0] = np.array(features[0])
    print type(features[0])
    print features[0].shape
    #for i in range(0,len(features)):
    #    if np.isnan(features[i]).any() == True:
    #        print('features %d have NaN:'% i,np.isnan(features[i]).any())
    '''      
    #loading multi-labels 
    labels = scipy.io.loadmat('dataset/pattlabels.mat') #LandUse_multilabels
    labels = labels['labels']
    #labels = np.transpose(labels)

    #loading positive-labels 
    #pos = scipy.io.loadmat('dataset/Indexes.mat') ##labels
    #pos = pos['ind']
    #pos = np.transpose(pos)

    #neg = scipy.io.loadmat('dataset/Index-.mat') ##labels
    #neg = labels['neg_ind']
    #neg = np.transpose(neg)
    #load pairs
    #pairs = scipy.io.loadmat('dataset/pair.mat') 
    
    # Calculating class weights

    lp = LabelPowerset()
    trans_labels = lp.transform(labels)
    unique, counts = np.unique(trans_labels, return_counts=True)
    class_freq = 1.0 / counts
    weight_mat = np.zeros((np.shape(trans_labels)))
    for i in range(len(weight_mat)):
        weight_mat[i] = class_freq[np.where(trans_labels[i]==unique)]
        
    # Calculating label weights
    sum_labels = np.sum(labels, axis=0, dtype=np.float32)
    sum_tot = np.sum(sum_labels, dtype=np.float32)
    label_freq = np.true_divide(sum_labels, sum_tot)
    
    return features, edges, labels, weight_mat, label_freq, index, classes
