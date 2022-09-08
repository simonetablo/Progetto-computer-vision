import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

#torch.set_default_tensor_type(torch.DoubleTensor)

def validate(model, batch_size, device, val_loader, epoch):
    model.eval()
    epoch_accuracy=0
    calc=[]
    output=pd.DataFrame(columns=['query', 'database'])
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
            qidx=i
            dbidx=i
            for j in range(db.shape[0]):
                dist=sum(((sQ[i:i+1] - sDB[j:j+1])**2).reshape(4096))
                calc.append(float(dist))
                if dist<min_dist:
                    value=0
                    dbidx=j
            values.append(value)
            output=output.append({'query': qidx,'database': dbidx}, ignore_index=True)
        accuracy=sum(values)/q.shape[0]
        epoch_accuracy+=accuracy
    outname='output_E'+str(epoch)+'.csv'
    output.to_csv(outname)
    np.save('provafor', np.array(calc))
    return epoch_accuracy