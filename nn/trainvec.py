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

    start_time = time.time()

    for batch_idx, data in enumerate(train_loader):
        accuracy=0
        loss=0
        qdesc,pdesc,ndesc,ind=data
        input=torch.cat([qdesc, pdesc, ndesc]).float()
        input=input.to(device)
        optimizer.zero_grad()
        results=model.pool(input)
        sQ, sP, sN=torch.split(results, [batch_size, batch_size, neg_per_query*batch_size])
        values=[]
        negidx=0

        sDBn=torch.neg(sN)
        for k in range(int(qidx)):
            value=1
            sQv=sQ[k:k+1].expand(qidx, -1)
            npd=torch.dstack((sQv,sDBn))
            npsum=torch.sum(npd, axis=2)
            nptmp=torch.dstack((npsum,npsum))
            nppow=torch.prod(nptmp, axis=2)
            npfin=torch.sum(nppow, axis=1)
            res=torch.count_nonzero(npfin<npfin[k])
            if(res>0):
                value=0
            values.append(value)


        for b in range(batch_size):
            min_dist=sum(((sQ[b:b+1] - sP[b:b+1])**2).reshape(4096))
            value=1
            for n in range(neg_per_query):
                loss += criterion(sQ[b:b+1], sP[b:b+1], sN[negidx:negidx+1])
                dist=sum(((sQ[b:b+1] - sN[negidx:negidx+1])**2).reshape(4096))
                if dist<min_dist:
                    value=0
                negidx+=1
            values.append(value)
        loss /= torch.tensor(batch_size*neg_per_query).float().to(device)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
        accuracy=sum(values)/batch_size
        epoch_accuracy+=accuracy

    print("--- %s seconds ---" % (time.time() - start_time))
    
    epoch_accuracy/=len(train_loader)
    return(epoch_loss, epoch_accuracy)