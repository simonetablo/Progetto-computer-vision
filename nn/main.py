import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import random
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime


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
        print('> datasets shuffled.')
        
    #creating model
    model=nn.Module()
    seqmodel=my_models.seq_model(4096, 4096, 5)
    flatten=my_models.Flatten()
    L2norm=my_models.L2Norm()
    model.add_module('pool', nn.Sequential(*[seqmodel, flatten, L2norm]))
    #model=nn.DataParallel(model.pool)
    model.to(device)
    print('> model created.')
    
    train_batch_size=16
    val_batch_size=415
    test_batch_size=265
    num_threads=4
        
    features_dir=''
    query='random_query.npy'
    database='random_database.npy'
    night='random_night.npy'
    split_train=[0, 2000]
    split_test=[2000, 2400]
    split_night=[0, 265]
    neg_per_query=10
    
    #creating datasets
    train_dataset=my_datasets.getTrainDescriptors(features_dir, neg_per_query, split_train, query, database)
    train_loader=DataLoader(dataset=train_dataset, num_workers=num_threads, batch_size=train_batch_size, collate_fn=my_datasets.collate_fn, shuffle=False)
    
    val_dataset=my_datasets.getTestDescriptors(features_dir, split_test, query, database)
    val_loader=DataLoader(dataset=val_dataset, num_workers=num_threads, batch_size=val_batch_size, shuffle=False)

    test_dataset=my_datasets.getTestDescriptors(features_dir, split_night, night, database)
    test_loader=DataLoader(dataset=test_dataset, num_workers=num_threads, batch_size=test_batch_size, shuffle=False)
    print('> datasets created.')
    
    margin=0.01
    learning_rate=1e-3
    
    #defining loss and optimizer
    criterion=nn.TripletMarginLoss(margin=margin, p=2, reduction='sum').to(device)
    optimizer=optim.Adam(model.parameters(), lr=learning_rate)
    
    num_epoch=20
    
    print("> Train started. %d epochs set." %num_epoch)
    train_accuracy_history=[]
    validation_accuracy_history=[]
    train_loss_history=[]
    for epoch in range(num_epoch):
        print('> Epoch %d' %(epoch+1))
        train_loss, train_accuracy=train.train(model, train_batch_size, neg_per_query, device, optimizer, criterion, train_loader)
        print('TRAIN: loss: %.6f  ,  accuracy: %.6f' %(train_loss, train_accuracy))
        validation_accuracy=validation.validate(model, val_batch_size, device, val_loader)
        print('VALIDATION: accuracy: %.6f' %(validation_accuracy))
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        validation_accuracy_history.append(validation_accuracy)

    print('> Testing model.')
    test_accuracy=validation.validate(model, test_batch_size, device, test_loader)
    print('TEST: accuracy: %.6f' %(test_accuracy))
    unique=str(datetime.now())

    torch.save(model.state_dict(), 'weights_E'+str(num_epoch)+'_'+unique+'.pth')
        
    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,num_epoch+1), train_loss_history)
    plt.xlabel("epoch")
    plt.ylabel("train loss")

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,num_epoch+1), train_accuracy_history)
    plt.xlabel("epoch")
    plt.ylabel("train accuracy")

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,num_epoch+1), validation_accuracy_history)
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy")

    plt.show()
    