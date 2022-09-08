import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

#torch.set_default_tensor_type(torch.DoubleTensor)

def validate(model, batch_size, device, val_loader):
    model.eval()
    epoch_accuracy=0
    cpu=torch.device('cpu')
    for batch_idx, data in enumerate(val_loader):
        accuracy=0
        q, db=data
        qidx=q.shape[0]
        dbidx=db.shape[0]
        input=torch.cat([q, db]).float()
        input=input.to(device)
        results=model.pool(input)
        sQ, sDB=torch.split(results, [qidx, dbidx])
        sQ=sQ.to(cpu, dtype=torch.float64)
        sDB=sDB.to(cpu, dtype=torch.float64)
        values=[]
        sDBn=torch.neg(sDB)
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
        accuracy=sum(values)/qidx
        epoch_accuracy+=accuracy
    return epoch_accuracy