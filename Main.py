import numpy as np
import os
import pandas as pd
from numpy import matlib
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from BlockChain import Blockchain
from FA import FA
from FFO import FFO
from Global_Vars import Global_Vars
from Model_ADASYN_CNN import Model_ADASYN_CNN
from Model_ADeepCRF import Model_ADeepCRF
from Model_ANN import Model_ANN
from Model_ANFIS import Model_ANFIS
from Model_SVM import Model_SVM_Feat
from NGO import NGO
from PROPOSED import PROPOSED
from SOA import SOA
from objective_function import Objfun
import random as rn
from Plot_Results import *

No_of_Dataset = 2

# Read Dataset 1
an = 0
if an == 1:
    dir = './Dataset/Dataset1/'
    list_dir = os.listdir(dir)
    file = dir + list_dir[0]
    read = pd.read_csv(file)
    read = read.values
    read = np.delete(read, 3, 1)  # Not a propoer data
    str_type = [1, 2, 3, 5, 9, 13]
    New = np.zeros(read.shape).astype('int')
    for j in range(read.shape[1]):
        if j in str_type:
            d = np.zeros((read.shape[0])).astype('int')
            uni = np.unique(str(read[:, j]))
            for k in range(len(uni)):
                ind = np.where(read[:, j] == uni[k])
                d[ind[0]] = k + 1
            New[:, j] = d.astype('int')
        else:
            New[:, j] = read[:, j].astype('float')
    Targ = read[:, 5]
    Necessary_data = New
    Necessary_data = np.delete(Necessary_data, 6, 1)
    uni = np.unique(Targ)
    tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
    for i in range(len(uni)):
        ind3 = np.where((Targ == uni[i]))
        tar[ind3[0], i] = 1
    Necessary_data, tar = shuffle(Necessary_data, tar)
    np.save('Data_1.npy', Necessary_data)
    np.save('Target_1.npy', tar)

# Read dataset 2
an = 0
if an == 1:
    dir = './Dataset/Dataset2/'
    list_dir = os.listdir(dir)
    file = dir + list_dir[0]
    read = pd.read_csv(file)
    read = read.values
    Targ = read[:, 2]
    Data = np.delete(read, 2, 1)
    str_type = [0]
    New = np.zeros(Data.shape).astype('int64')
    for j in range(Data.shape[1]):
        if j in str_type:
            d = np.zeros((Data.shape[0])).astype('int')
            uni = np.unique(Data[:, j])
            for k in range(len(uni)):
                ind = np.where(Data[:, j] == uni[k])
                d[ind[0]] = k + 1
            New[:, j] = d.astype('int')
        else:
            New[:, j] = Data[:, j].astype('float')
    uni = np.unique(Targ)
    tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
    for i in range(len(uni)):
        ind3 = np.where((Targ == uni[i]))
        tar[ind3[0], i] = 1
    New, tar = shuffle(New, tar)
    np.save('Data_2.npy', New)
    np.save('Target_2.npy', tar)

# Stored in a Blockchain
an = 0
if an == 1:
    for a in range(No_of_Dataset):  # For Dataset
        Secured_Data = np.load('Data_' + str(a + 1) + '.npy', allow_pickle=True)
        private_key_authority = b'\x01' * 32
        blockchain = Blockchain(private_key_authority)
        Data = []
        for i in range(len(Secured_Data)):
            print(a, i)
            raw_data = Secured_Data[i]
            certificate_data = raw_data.tolist()
            certificate = blockchain.add_certificate(certificate_data)
            Data.append(certificate)
        np.save('Digital_Certificate_Data_' + str(a + 1) + '.npy', Data)

# Feature Extraction
an = 0
if an == 1:
    for a in range(No_of_Dataset):
        Data = np.load('Data_' + str(a + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        Feat_SVM = Model_SVM_Feat(Data, Target)
        model = TSNE(n_components=2, random_state=0)
        Feat_TSNE = model.fit_transform(Data)
        Feat = np.concatenate((Feat_SVM, Feat_TSNE), axis=1)
        np.save('Feature_' + str(a + 1) + '.npy', Feat)

# Optimization for Classification
an = 0
if an == 1:
    Best = []
    Fit = []
    for a in range(No_of_Dataset):
        Data = np.load('Feature_' + str(a + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        Global_Vars.Feat = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3
        xmin = matlib.repmat([5, 0.01, 100], Npop, 1)
        xmax = matlib.repmat([255, 0.99, 500], Npop, 1)
        fname = Objfun
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.asarray(np.random.uniform(xmin[p1, p2], xmax[p1, p2]))
        Max_iter = 50

        print("FFO...")
        [bestfit1, fitness1, bestsol1, time1] = FFO(initsol, fname, xmin, xmax, Max_iter)

        print("FA...")
        [bestfit2, fitness2, bestsol2, time2] = FA(initsol, fname, xmin, xmax, Max_iter)

        print("NGO...")
        [bestfit4, fitness4, bestsol4, time3] = NGO(initsol, fname, xmin, xmax, Max_iter)

        print("SOA...")
        [bestfit3, fitness3, bestsol3, time4] = SOA(initsol, fname, xmin, xmax, Max_iter)

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

        Bestsol = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
        Fitness = ([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])
        Best.append(Bestsol)
        Fit.append(Fitness)
    np.save('Bestsol.npy', np.asarray(Best))
    np.save('Fitness.npy', np.asarray(Fit))

# Classification (Varying Batch Size)
an = 0
if an == 1:
    Eval_all = []
    for a in range(No_of_Dataset):
        Feat = np.load('Feature_' + str(a + 1) + '.npy', allow_pickle=True)
        Bstsol = np.load('Bestsol.npy', allow_pickle=True)[a]
        Target = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        EVAL = []
        Batch_Size = [4, 8, 16, 32, 48]
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        for batch in range(len(Batch_Size)):
            Eval = np.zeros((10, 23))
            for i in range(len(Bstsol)):
                sol = Bstsol[i, :]
                Eval[i, :] = Model_ADeepCRF(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval[5:] = Model_ANN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :] = Model_ADASYN_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :] = Model_ANFIS(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :] = Model_ADeepCRF(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Eval_Batch.npy', np.asarray(Eval_all))

# Classification (Varying Hidden Neuron count)
an = 0
if an == 1:
    Eval_all = []
    for a in range(No_of_Dataset):
        Feat = np.load('Feature_' + str(a + 1) + '.npy', allow_pickle=True)
        Bstsol = np.load('Bestsol.npy', allow_pickle=True)[a]
        Target = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        EVAL = []
        Batch_size = [4, 8, 16, 32, 48]
        for Batch in range(len(Batch_size)):
            Eval = np.zeros((10, 23))
            learnperc = round(Feat.shape[0] * 0.75)
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            for i in range(len(Bstsol)):
                sol = Bstsol[i, :]
                Eval[i, :] = Model_ADeepCRF(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval[5:] = Model_ANN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :] = Model_ADASYN_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :] = Model_ANFIS(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :] = Model_ADeepCRF(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Eval_Hidden.npy', np.asarray(Eval_all))

Plot_Results()
Plot_table()
Confusion_matrix()
Plot_ROC()
Plot_Fitness()
New_Plot()
