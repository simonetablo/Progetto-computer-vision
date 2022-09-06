import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def validate(model, batch_size, device, val_loader):
    model.eval()
    epoch_accuracy=0
    for batch_idx, data in enumerate(val_loader):
        accuracy=0
        q, db=data
        input=torch.cat([q, db]).float()
        input=input.to(device)
        results=model.pool(input)
        sQ, sDB=torch.split(results, [q.shape[0], db.shape[0]])
        values=[]
        for i in range(int(q.shape[0])):
            min_dist=sum(((sQ[i:i+1] - sDB[i:i+1])**2).reshape(4096))
            value=1
            for j in range(db.shape[0]):
                dist=sum(((sQ[i:i+1] - sDB[j:j+1])**2).reshape(4096))
                if dist<=min_dist and i!=j:
                    value=0
            values.append(value)
        accuracy=sum(values)/q.shape[0]
        epoch_accuracy+=accuracy
    return epoch_accuracy