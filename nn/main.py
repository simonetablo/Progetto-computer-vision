import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import os

import my_models
import my_datasets
import train
import validation

#torch.set_default_tensor_type(torch.DoubleTensor)

if __name__=="__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    
    shuffle_datasets=False
    if(shuffle_datasets==True):
        my_datasets.shuffle_dataset()
        print('> datasets shuffled')

    
    train_batch_size=16
    val_batch_size=400
    test_batch_size=700
    num_threads=4
        
    features_dir=''
    query='random_query.npy'
    database='random_database.npy'
    snow='random_snow.npy'
    night='random_night.npy'
    query_test='globalfeats_q_test.npy'
    database_test='globalfeats_db_test.npy'
    query_test_2='globalfeats_q_test_2.npy'
    database_test_2='globalfeats_db_test_2.npy'
    #split_train=[0, 2000]
    #split_val=[2000, 2400]
    split_test=[0, 700]

    pos_per_query=2
    neg_per_query=10
    
    #creating datasets
    #train_dataset=my_datasets.getTrainDescriptors(features_dir, neg_per_query, split_train, query, database)
    #train_loader=DataLoader(dataset=train_dataset, num_workers=num_threads, batch_size=train_batch_size, collate_fn=my_datasets.collate_fn, shuffle=False)
    #
    #val_dataset=my_datasets.getTestDescriptors(features_dir, split_val, query, database)
    #val_loader=DataLoader(dataset=val_dataset, num_workers=num_threads, batch_size=val_batch_size, shuffle=False)

    test_dataset=my_datasets.getTestDescriptors(features_dir, split_test, query_test, database_test)
    test_loader=DataLoader(dataset=test_dataset, num_workers=num_threads, batch_size=test_batch_size, shuffle=False)

    margin=1
    learning_rate=1e-5
    
    k_fold=6
    fold_scores=pd.DataFrame(columns=['fold', 'n.epoch', 'train_loss', 'train_accuracy', 'val_accuracy', 'test_accuracy', 't.l.margin', 'learning_reate', 'NpQ'])

    total_size = 2400
    fraction = 1/k_fold
    segment = int(total_size * fraction)

    unique=str(datetime.now())
    
    dir='../results/'+unique
    os.mkdir(dir)
    
    for k in range(k_fold):

        print('> Fold n.'+str(k+1))
        
        model=nn.Module()
        seqmodel=my_models.seq_model(4096, 4096, 5)
        flatten=my_models.Flatten()
        L2norm=my_models.L2Norm()
        model.add_module('pool', nn.Sequential(*[seqmodel, flatten, L2norm]))
        #model=nn.DataParallel(model.pool)
        model.to(device)
        print(' model created')

        train_i00 = 0
        train_i01 = k * segment
        val_i0 = train_i01
        val_i1 = k * segment + segment
        train_i10 = val_i1
        train_i11 = total_size

        split_train=[train_i00, train_i01, train_i10, train_i11]
        split_val=[val_i0, val_i1]        

        train_dataset=my_datasets.getTrainDescriptors(features_dir, neg_per_query, split_train, query, database, snow)
        train_loader=DataLoader(dataset=train_dataset, num_workers=num_threads, batch_size=train_batch_size, collate_fn=my_datasets.collate_fn, shuffle=False)

        val_dataset=my_datasets.getTestDescriptors(features_dir, split_val, query, database)
        val_loader=DataLoader(dataset=val_dataset, num_workers=num_threads, batch_size=val_batch_size, shuffle=False)

        print(' datasets created')


        #defining loss and optimizer
        criterion=nn.TripletMarginLoss(margin=margin, p=2, reduction='sum').to(device)
        optimizer=optim.Adam(model.parameters(), lr=learning_rate)
    
        num_epoch=10

        print(" > Train started. %d epochs set" %num_epoch)
        train_accuracy_history=[]
        validation_accuracy_history=[]
        train_loss_history=[]
        for epoch in range(num_epoch):
            print('  > Epoch %d' %(epoch+1))
            train_loss, train_accuracy=train.train(model, train_batch_size, pos_per_query, neg_per_query, device, optimizer, criterion, train_loader)
            print('    TRAIN: loss: %.6f  ,  accuracy: %.6f' %(train_loss, train_accuracy))
            validation_accuracy, validation_results=validation.validate(model, val_batch_size, device, val_loader)
            print('    VALIDATION: accuracy: %.6f' %(validation_accuracy))
            validation_results.to_csv(dir+'/validation_results_F'+str(k)+'_E'+str(epoch)+'.csv')
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)
            validation_accuracy_history.append(validation_accuracy)

        print(' > Testing model')
        test_accuracy, test_results=validation.validate(model, test_batch_size, device, test_loader)
        print('    TEST: accuracy: %.6f' %(test_accuracy))
        test_results.to_csv(dir+'/test_results_F'+str(k)+'.csv')

        torch.save(model.state_dict(), dir+'/weights_Kf'+str(k)+'.pth')

        fold_scores.loc[k]=[k, num_epoch, train_loss, train_accuracy, validation_accuracy, test_accuracy, margin, learning_rate, neg_per_query]

        r=1
        c=3
        fig = plt.figure(figsize=(30, 10), tight_layout=True)
        fig.add_subplot(r,c,1)
        plt.plot(np.arange(1,num_epoch+1), train_loss_history)
        plt.xlabel("epoch")
        plt.ylabel("train loss")
        

        fig.add_subplot(r,c,2)
        plt.plot(np.arange(1,num_epoch+1), train_accuracy_history)
        plt.xlabel("epoch")
        plt.ylabel("train accuracy")

        fig.add_subplot(r,c,3)
        plt.plot(np.arange(1,num_epoch+1), validation_accuracy_history)
        plt.xlabel("epoch")
        plt.ylabel("validation accuracy")

        fig.savefig(dir+'/plots_Kf'+str(k)+'.png')
    fold_scores.to_csv(dir+'/scores.csv')
