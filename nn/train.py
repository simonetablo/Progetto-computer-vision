import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


#torch.set_default_tensor_type(torch.DoubleTensor)

def train(model, batch_size, pos_per_query, neg_per_query, device, optimizer, criterion, train_loader):
    model.train()
    epoch_loss=0
    epoch_accuracy=0
    for batch_idx, data in enumerate(train_loader):
        accuracy=0
        loss=0
        qdesc,pdesc,ndesc,ind=data
        input=torch.cat([qdesc, pdesc, ndesc]).float()
        input=input.to(device)
        optimizer.zero_grad()
        results=model.pool(input)
        sQ, sP, sN=torch.split(results, [batch_size, pos_per_query*batch_size, neg_per_query*batch_size])
        values=[]
        for b in range(batch_size*pos_per_query):
            c=b
            if(b>=batch_size):
                c-=batch_size
            min_dist=sum(((sQ[c:c+1] - sP[b:b+1])**2).reshape(4096))
            value=1
            for n in range(neg_per_query):
                loss += criterion(sQ[c:c+1], sP[b:b+1], sN[10*c+n:10*c+n+1])
                dist=sum(((sQ[c:c+1] - sN[10*c+n:10*c+n+1])**2).reshape(4096))
                if dist<min_dist:
                    value=0
            values.append(value)
        loss /= torch.tensor(batch_size*pos_per_query*neg_per_query).float().to(device)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
        accuracy=sum(values)/(batch_size*pos_per_query)
        epoch_accuracy+=accuracy
    epoch_accuracy/=len(train_loader)
    return(epoch_loss, epoch_accuracy)