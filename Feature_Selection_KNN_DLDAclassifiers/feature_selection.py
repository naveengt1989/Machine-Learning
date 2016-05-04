import numpy as np
import mlpy
import itertools
from collections import defaultdict

def input(train_data,test_data):
    genes=[i for i in raw_input().split('\t')]
    print genes
    for i in xrange(60): 
        inp = raw_input().split('\t')
        train_data.append([float(j) for j in inp])
    for i in xrange(235): 
        inp = raw_input().split('\t')
        test_data.append([float(j) for j in inp])
    return genes    

def select(data,subset):
    sel = []
    sel = data[:,subset]
    return sel 

def dlda_exh(train_data,test_data,Y_train,Y_test,d,genes,indices):
    ################# DLDA-Error Estimation(exhaustive search) #############################
    err_set =defaultdict(list)
    dlda = mlpy.DLDA(delta=0.5)
    for subset in itertools.combinations(indices,d):
        mismatch=0
        selX =[]
        selX = select(train_data,list(subset))
        dlda.learn(selX, Y_train)
        Y_pred = dlda.pred(selX) 
        for i in range(len(Y_pred)):
            if int(Y_pred[i])!=int(Y_train[i]): 
                mismatch += 1
        err_set[float(mismatch)/float(len(Y_pred))].append(subset)             
    print min(err_set.keys()), err_set[min(err_set.keys())]  # Error estimate
    print [genes[x] for x in err_set[min(err_set.keys())][0]]
    selX = select(test_data,list(err_set[min(err_set.keys())][0]))
    mismatch=0
    Y_pred = dlda.pred(selX) 
    for i in range(len(Y_pred)):
        if int(Y_pred[i])!=int(Y_test[i]): 
            mismatch += 1
    print float(mismatch)/float(len(Y_test))           # test set error estimate 
    ######################################################################

def dlda_fss(train_data,test_data,Y_train,Y_test,d,genes):
    ################# DLDA-Error Estimation(forward sequential search) ######################
    selX = [] 
    taken = defaultdict(int)
    final_error=0.0
    final_subset=[]
    dlda = mlpy.DLDA(delta=0.5)
    for i in xrange(d):
        err_set =defaultdict(list)
        for j in xrange(train_data.shape[1]):
            if taken[j]!=0:
                continue
            mismatch=0
            if(np.array(selX).shape[0]==0):
                selX.append(np.ravel(train_data[:,j]))
                selX = np.array(selX).transpose()
            else:
                selX = np.append(selX,train_data[:,j],axis=1)
            dlda.learn(selX, Y_train)
            Y_pred = dlda.pred(selX) 
            for i in range(len(Y_pred)):
                if int(Y_pred[i])!=int(Y_train[i]): 
                    mismatch += 1
            err_set[float(mismatch)/float(len(Y_train))].append(j)    
            selX= np.delete(selX,-1,1)
        selX = np.append(selX,train_data[:,err_set[min(err_set.keys())][0]],axis=1)
        final_subset.append(err_set[min(err_set.keys())][0])
        final_error = min(err_set.keys())
        taken[err_set[min(err_set.keys())][0]]=1 
    print final_error, [genes[x] for x in final_subset] # Error estimate 
    mismatch=0
    selX = select(test_data,final_subset)
    Y_pred = dlda.pred(selX) 
    for i in range(len(Y_pred)):
        if int(Y_pred[i])!=int(Y_test[i]): 
            mismatch += 1
    print float(mismatch)/float(len(Y_test))    # test set error estimate 
    ######################################################################

def knn_fss(train_data,test_data,Y_train,Y_test,d,genes,K):
    ################# KNN-Error Estimation(forward sequential search) #############################
    selX = [] 
    taken = defaultdict(int)
    final_error=0.0
    final_subset=[]
    dlda = mlpy.KNN(k=K)
    for i in xrange(d):
        err_set =defaultdict(list)
        for j in xrange(train_data.shape[1]):
            if taken[j]!=0:
                continue
            mismatch=0
            if(np.array(selX).shape[0]==0):
                selX.append(np.ravel(train_data[:,j]))
                selX = np.array(selX).transpose()
            else:
                selX = np.append(selX,train_data[:,j],axis=1)
            dlda.learn(selX, Y_train)
            Y_pred = dlda.pred(selX) 
            for i in range(len(Y_pred)):
                if int(Y_pred[i])!=int(Y_train[i]): 
                    mismatch += 1
            err_set[float(mismatch)/float(len(Y_train))].append(j)    
            selX= np.delete(selX,-1,1)
        selX = np.append(selX,train_data[:,err_set[min(err_set.keys())][0]],axis=1)
        final_subset.append(err_set[min(err_set.keys())][0])
        final_error = min(err_set.keys())
        taken[err_set[min(err_set.keys())][0]]=1 
    print final_error, [genes[x] for x in final_subset] # Error estimate 
    mismatch=0
    selX = select(test_data,final_subset)
    Y_pred = dlda.pred(selX) 
    for i in range(len(Y_pred)):
        if int(Y_pred[i])!=int(Y_test[i]): 
            mismatch += 1
    print float(mismatch)/float(len(Y_test))      # test set error estimate 
    ######################################################################

def knn_exh(train_data,test_data,Y_train,Y_test,d,genes,indices,K):   
    ################# 3NN-Error Estimation(exhaustive search)  #############################
    err_set =defaultdict(list)
    dlda = mlpy.KNN(k=K)
    for subset in itertools.combinations(indices,d):
        mismatch=0
        selX =[]
        selX = select(train_data,list(subset))
        dlda.learn(selX, Y_train)
        Y_pred = dlda.pred(selX) 
        for i in range(len(Y_pred)):
            if int(Y_pred[i])!=int(Y_train[i]): 
                mismatch += 1
        err_set[float(mismatch)/float(len(Y_pred))].append(subset)             
    print min(err_set.keys()), err_set[min(err_set.keys())]  # Error estimate 
    print [genes[x] for x in err_set[min(err_set.keys())][0]]
    selX = select(test_data,list(err_set[min(err_set.keys())][0]))
    mismatch=0
    Y_pred = dlda.pred(selX) 
    for i in range(len(Y_pred)):
        if int(Y_pred[i])!=int(Y_test[i]): 
            mismatch += 1
    print float(mismatch)/float(len(Y_test))# test set error estimate 
    ######################################################################

train_data= []
test_data= []
genes = input(train_data,test_data)
train_data = np.matrix(train_data)   
test_data = np.matrix(test_data)   
NUM_FEATURES=train_data.shape[1]-1
indices = np.arange(0,NUM_FEATURES,1)
Y_train = np.array(train_data[:,NUM_FEATURES])
Y_test = np.array(test_data[:,NUM_FEATURES])
train_data = train_data[:,0:NUM_FEATURES]
test_data = test_data[:,0:NUM_FEATURES]
Y_train = Y_train.ravel()
Y_test = Y_test.ravel()
#dlda_fss(train_data,test_data,Y_train,Y_test,8,genes)
knn_fss(train_data,test_data,Y_train,Y_test,8,genes,3)
#dlda_exh(train_data,test_data,Y_train,Y_test,3,genes,indices)
#knn_exh(train_data,test_data,Y_train,Y_test,3,genes,indices,3)



