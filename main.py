import random
import numpy as np
import time
import tensorflow as tf 
import input_data
import math
import scipy.io
from sklearn.model_selection import train_test_split
from numpy import array
from numpy.linalg import norm



import glob
import os
import itertools
import csv
import scipy.io as sio
from numpy import array



#mnist = input_data.read_data_sets("/tmp/data",one_hot=False)
alpha = 0.0001
margin=1


import pdb
def create_pairs1(train,n):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''

    pair = []
    labels = []
    print ('HI BAEEEEBY')
 
   
    for d in range(n):
        
        for i in range(38):
	     arr = np.arange(38)
             np.random.shuffle(arr)
             arr = list(arr)

             z1, z2 = np.random.choice(train[i]), np.random.choice(train[i])
             pair += [[z1, z2]]
	     arr.remove(i)

             z1, z2 = np.random.choice(train[i]), np.random.choice(train[np.random.choice(arr)])
             pair += [[z1, z2]]
             labels += [0, 1]

 

    print len(np.array(pair))
    return np.array(pair), np.array(labels)



'''

def create_pairs(pairs):
    
    pair = []
    labels = []
    print ('HI BAEEEEBY')
    #pair = array(pairs)
    #print len(pairs)
    for d in range(len(pairs)):
        #for i in range(n):
            z1, z2 = pairs[d][0], pairs[d][1]
            pair += [[z1, z2]]
            z1, z2 = pairs[d][0], pairs[d][3]
            pair += [[z1, z2]]
	    labels += [0, 1]
	    z1, z2 = pairs[d][0], pairs[d][2]
            pair += [[z1, z2]]
            z1, z2 = pairs[d][0], pairs[d][4]
            pair += [[z1, z2]]
            labels += [0, 1]
    #print len(np.array(pair))
 
    return np.array(pair), np.array(labels)
'''

def mlp(input_,input_dim,output_dim,name="mlp"):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[input_dim,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.001,stddev=0.02))
	#print('oooooo la laaaa...')
        #print w.shape
        #print type(w)
        w = tf.nn.l2_normalize(w,dim=None)
        #print w
        #print type(w)
        return tf.nn.relu(tf.matmul(input_,w))
       
def build_model_mlp(X_,_dropout):

    model= mlpnet(X_,_dropout)
    return model

def load_model( sess, saver):
        latest = tf.train.latest_checkpoint(snapshot_path)
        print(latest)
        if latest == None:
            return 0
        saver.restore(sess, latest)
        i = int(latest[len(snapshot_path + 'model-'):])
        print("Model restored at %d." % i)
        return i
        
def save_model( sess, saver, i):
        if not os.path.exists(snapshot_path):
                os.makedirs(snapshot_path)
        latest = tf.train.latest_checkpoint(snapshot_path)
        #if latest == None or i != int(latest[len(self.snapshot_path + 'model-'):]):
        if 1:
            print('Saving model at %d' % i)
            #verify_dir_exists(self.snapshot_path)
            result = saver.save(sess, snapshot_path + 'model', global_step=i)
            print('Model saved to %s' % result)

def dense_to_one_hot(labels_dense, num_classes=38):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  #print labels_dense.shape
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels+1, num_classes))
  #print index_offset + labels_dense.ravel()     #    1 OR 0        CHECK
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot



def mlpnet(image,_dropout):
    #l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005)
    
    #image = tf.convert_to_tensor (image)
    l1 = mlp(image,256,256,name='l1') #256,128
    l1 = tf.nn.dropout(l1,_dropout)
    l2 = mlp(l1,256,256,name='l2') #128,128
    #l2 = tf.nn.dropout(l2,_dropout)
    #l3 = mlp(l2,256,256,name='l3') #same
    #l3 = tf.nn.dropout(l3,_dropout)
    #l4 = mlp(l3,256,256,name='l4') #same
    #l4 = tf.nn.dropout(l4,_dropout)
    #l5 = mlp(l4,256,256,name='l5') #same

    
    return l2
def contrastive_loss( y,d,b1,b2,alpha):


    tmp= (1-y) *tf.square(d)
    #tmp= tf.mul(y,tf.square(d))
    tmp2 = (y) *tf.square(tf.maximum((margin - d),0))
    #b1 = abs(abs(b1) - onevec)
    #b2 = abs(abs(b2) - onevec)
	
    #tmp3 = tf.reduce_max(tf.reduce_sum(b1,1,keep_dims=True))
    #tmp4 = tf.reduce_max(tf.reduce_sum(b1,1,keep_dims=True))

    
    #b11 = tf.sign(b1)
    #b22 = tf.sign(b2)              
    #tmp5 =  tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(b1,b11),2),1,keep_dims=True))
    #tmp6 =  tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(b2,b22),2),1,keep_dims=True))
    tmp7 = tf.norm(b1)
    tmp8 = tf.norm(b2)
    #tmp5 = tf.norm(b1,ord=1)
    #tmp6 = tf.norm(b2,ord=1)
    return tf.reduce_sum(tmp +tmp2 + alpha *(tmp7 +tmp8))/batch_size/2 # 

def compute_accuracy(prediction,labels):
    return labels[prediction.ravel() > 0.5].mean()
    #return tf.reduce_mean(labels[prediction.ravel() < 0.5])
def next_batch(s,e,inputs,labels):
    #print ('Here you see me')
    #print len(inputs)
    input1 = inputs[s:e,0]
    input2 = inputs[s:e,1]
    #inputa = input1
    inputa = np.zeros((128,256))
    inputb = np.zeros((128,256))
    #print input1.shape[0]
    for i in range(input1.shape[0]):
  	inputa[i] = fcfeat[input1[i]]
    for i in range(input2.shape[0]):
  	inputb[i] = fcfeat[input2[i]]
    #print inputs.shape
    #print inputa.shape
 
    y= np.reshape(labels[s:e],(len(range(s,e)),1))
    return inputa,inputb,y


    
# Initializing the variables
#init = tf.initialize_all_variables()
# the data, shuffled and split between train and test sets
#X_train = mnist.train._images
#y_train = mnist.train._labels
#X_test = mnist.test._images
#y_test = mnist.test._labels
path = 'src'
snapshot_path = path+'/snapshots/'

fcfeat = scipy.io.loadmat('pattGCN.mat')
fcfeat = fcfeat['fc_features']


X_train = scipy.io.loadmat('X_train.mat')
X_train = X_train['train_ind']
X_train = np.transpose(X_train)
train = X_train

X_test = scipy.io.loadmat('X_test.mat')
X_test = X_test['test_ind']
X_test = np.transpose(X_test)

y_train = scipy.io.loadmat('y_train.mat')
y_train = y_train['y_train']
y_train = np.transpose(y_train)

y_test = scipy.io.loadmat('y_test.mat')
y_test = y_test['y_test']
y_test = np.transpose(y_test)

yeast = scipy.io.loadmat('labels.mat')
test = yeast['test']
test = np.transpose(test)
train = yeast['train']
train = np.transpose(train)


#pairs1 = scipy.io.loadmat('new_pairs.mat')
#pos = pairs1['pos']
#pos = np.transpose(pos)
#neg = pairs1['neg']
#neg = np.transpose(neg)

#tpairs = scipy.io.loadmat('test_pairs.mat')
#tpairs = tpairs['test_indb']
#tpairs = np.transpose(tpairs)

true_labels = scipy.io.loadmat('pattlabels.mat') #LandUse_multilabels
true_labels = true_labels['labels']
#true_labels = np.transpose(true_labels)

train_images = np.zeros((23833,256))
test_images = np.zeros((6567,256))
#train_images = extract_images(local_file)



for i in range(23826):
  	train_images[i] = fcfeat[X_train[i]]
print('##')
X_train = train_images


train_labels = dense_to_one_hot(y_train, num_classes=38)
print train_labels.shape

for i in range(6566):
        test_images[i] = fcfeat[X_test[i]]
X_test = test_images


test_labels = dense_to_one_hot(y_test, num_classes=38)

batch_size =128
global_step = tf.Variable(0,trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10,0.1,staircase=True)
# create training+test positive and negative pairs
print('####################')
print X_train.shape
#digit_indices = [np.where(y_train == i)[0] for i in range(38)]
tr_pairs, tr_y= create_pairs1(train,5000)
digit_indices = [np.where(y_test == i)[0] for i in range(38)]
te_pairs, te_y = create_pairs1(test,5)

images_L = tf.placeholder(tf.float32,shape=([None,256]),name='L')
images_R = tf.placeholder(tf.float32,shape=([None,256]),name='R')
labels = tf.placeholder(tf.float32,shape=([None,1]),name='gt')
dropout_f = tf.placeholder("float")


with tf.variable_scope("siamese") as scope:
    model1 = build_model_mlp(images_L,dropout_f)
    scope.reuse_variables()
    model2 = build_model_mlp(images_R,dropout_f)

distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(model1,model2),2),1,keep_dims=True))
loss = contrastive_loss(labels,distance,model1,model2,alpha)
#contrastice loss
t_vars = tf.trainable_variables()
d_vars  = [var for var in t_vars if 'l' in var.name]
batch = tf.Variable(0)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)
#optimizer = tf.train.RMSPropOptimizer(0.0001,momentum=0.9,epsilon=1e-6).minimize(loss)
# Launch the graph
with tf.Session() as sess:
    #sess.run(init)
    tf.initialize_all_variables().run()
    # Training cycle
    for epoch in range(40):
        avg_loss = 0.
        avg_acc = 0.
        total_batch = int(len(np.array(tr_pairs))/batch_size)#int(X_train.shape[0]/batch_size)
        start_time = time.time()
        # Loop over all batches
        for i in range(total_batch):
            s  = i * batch_size
            e = (i+1) *batch_size
            # Fit training using batch data
            input1,input2,y=next_batch(s,e,tr_pairs,tr_y)
            #print('######')
            _,loss_value,predict=sess.run([optimizer,loss,distance], feed_dict={images_L:input1,images_R:input2 ,labels:y,dropout_f:0.9})
            feature1=model1.eval(feed_dict={images_L:input1,dropout_f:0.9})
            feature2=model2.eval(feed_dict={images_R:input2,dropout_f:0.9})
            tr_acc = compute_accuracy(predict,y) #hide
            if math.isnan(tr_acc) and epoch != 0:
                print('tr_acc %0.2f' % tr_acc)
                pdb.set_trace()
            avg_loss += loss_value
            avg_acc +=tr_acc*100
            
        #print('epoch %d loss %0.2f' %(epoch,avg_loss/total_batch))
        duration = time.time() - start_time
        print('epoch %d  time: %f loss %0.5f acc %0.2f' %(epoch,duration,avg_loss/(total_batch),avg_acc/total_batch))
    y = np.reshape(tr_y,(tr_y.shape[0],1))

    '''
    input1 = tr_pairs[:,0]
    input2 = tr_pairs[:,1]
    inputa = np.zeros((input1.shape[0],256))
    inputb = np.zeros((input1.shape[0],256))
    for i in range(input1.shape[0]):
  	inputa[i] = fcfeat[input1[i]]
    for i in range(input2.shape[0]):
  	inputb[i] = fcfeat[input2[i]]

    print inputa.shape
    print inputb.shape
    predict=distance.eval(feed_dict={images_L:inputa,images_R:inputb,labels:y,dropout_f:1.0})
    tr_acc = compute_accuracy(predict,y)
    print('Accuracy training set %0.2f' % (100 * tr_acc))
    '''
    # Test model
    input1 = te_pairs[:,0]
    input2 = te_pairs[:,1]
    #print te_pairs
    #print input1.shape
    inputa = np.zeros((input1.shape[0],256))
    inputb = np.zeros((input1.shape[0],256))
    
    for i in range(input1.shape[0]-4):
        #print input1[i]
  	inputa[i] = fcfeat[input1[i]]
    for i in range(input2.shape[0]):
  	inputb[i] = fcfeat[input2[i]]
    print inputa.shape
    print inputb.shape
    predict=distance.eval(feed_dict={images_L:inputa,images_R:inputb,labels:y,dropout_f:1.0})
    y = np.reshape(te_y,(te_y.shape[0],1))
    te_acc = compute_accuracy(predict,y)
    print('Accuracy test set %0.2f' % (100 * te_acc))

    saver = tf.train.Saver()
    save_model(sess, saver, epoch)

with tf.Session() as sess:
 sess.run(tf.global_variables_initializer())
 svr = load_model(sess, saver)
 #sess.run(tf.local_variables_initializer(), tf.variable_initialization)
 num = 30400
 batch_size = 30400
 inputa = np.zeros((30400,256))
 for i in range(inputa.shape[0]):
  	inputa[i] = fcfeat[i]
 fc_features = np.zeros((num,256)) # to store fully-connected layer's features

 for ind in range(0,num/batch_size): #processing the datapoints batchwise
                    ind_s = ind*batch_size
                    ind_e = (ind+1)*batch_size
		    
 		    feature1=model1.eval(feed_dict={images_L:inputa,dropout_f:1.0})
                    #reports = sess.run(self.reports, feed_dict={self.net.is_training:1})
		    #feature1 =  tf.contrib.layers.l1_regularizer(feature1)
                    fc_features[ind_s:ind_e,:] = feature1
                    
 sio.savemat('siam_features_patt.mat', {'fc_features':fc_features}) #saving
                
print('done')

image_R =  tf.contrib.layers.l1_regularizer(images_R)



#Accuracy test set 96.67





