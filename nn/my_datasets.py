import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random

#torch.set_default_tensor_type(torch.DoubleTensor)

class getTrainDescriptors(Dataset):
    def __init__(self, desc_dir, npq, split, query_desc, database_desc, transform=True):
        self.desc_dir=desc_dir
        self.npq=npq
        self.split=split
        self.transform=transform
        self.query_desc=np.load('./'+desc_dir+query_desc)[split[0]:split[1],]
        self.database_desc=np.load('./'+desc_dir+database_desc)[split[0]:split[1],]
        
    def __len__(self):
        return int(len(self.database_desc)/5)
    
    def __getitem__(self, idx):
        mult=idx*5
        indices=[]
        q_desc=self.query_desc[mult:mult+5, :]
        p_desc=self.database_desc[mult:mult+5, :]
        indices.append(idx)
        indices.append(idx)
        all_n_desc=np.concatenate((self.database_desc[:idx],self.database_desc[idx+5:]))
        n_idx=np.full((self.npq),idx)
        while idx>-1:
            if idx in n_idx:
                n_idx=np.random.choice(range(0, int(all_n_desc.shape[0]/5)),self.npq)
                break
        n_desc=[]
        for i in range(self.npq):
            ni=n_idx[i]*5
            n_desc.append(torch.tensor(all_n_desc[ni:ni+5, :]))
            indices.append(ni)
        n_desc=torch.stack(n_desc, 0)
        indices=np.array(indices)
        if self.transform:
            q_desc=torch.from_numpy(q_desc)
            p_desc=torch.from_numpy(p_desc)
            indices=torch.from_numpy(indices)
        return q_desc, p_desc, n_desc, indices
    
class getTestDescriptors(Dataset):
    def __init__(self, desc_dir, split, query_desc, database_desc, transform=True):
        self.desc_dir=desc_dir
        self.split=split
        self.transform=transform
        self.query_desc=np.load('./'+desc_dir+query_desc)[split[0]:split[1],]
        self.database_desc=np.load('./'+desc_dir+database_desc)[split[0]:split[1],]
        
    def __len__(self):
        return int(len(self.database_desc)/5)
    
    def __getitem__(self, idx):
        mult=idx*5
        query=self.query_desc[mult:mult+5, :]
        database=self.database_desc[mult:mult+5, :]
        if self.transform:
            query=torch.from_numpy(query)
            database=torch.from_numpy(database)
        return query, database


@staticmethod
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None, None
    query, positive, negatives, indices = zip(*batch)
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negatives = torch.cat(negatives, 0)
    indices = data.dataloader.default_collate(indices)
    #indices = torch.cat(indices, 0)
    return query, positive, negatives, indices


def shuffle_dataset():
    query_desc_file='globalfeats_q.npy'
    database_desc_file='globalfeats_db.npy'
    #night_desc_file='globalfeats_night.npy'
    
    labels=pd.read_csv('../dataset_selection/labels_db-q.csv')
    
    query_desc=np.load('./'+query_desc_file)
    database_desc=np.load('./'+database_desc_file)
    #night_desc=np.load('./'+night_desc_file)
    
    if query_desc.shape[0]<database_desc.shape[0]:
        ridx=int(query_desc.shape[0]/5)
        database_desc=database_desc[:query_desc.shape[0]]
    else:
        ridx=int(database_desc.shape[0]/5)
        query_desc=query_desc[:database_desc.shape[0]]
        
    random_indices=random.sample(range(ridx), ridx)
    r_query=np.zeros((query_desc.shape[0], query_desc.shape[1]))
    r_database=np.zeros((database_desc.shape[0], database_desc.shape[1]))
    #r_night=np.zeros((night_desc.shape[0], night_desc.shape[1]))
    
    r_labels=pd.DataFrame()

    for i in range(ridx):
        mi=i*5
        ri=random_indices[i]*5
        r_query[mi:mi+5]=query_desc[ri:ri+5]
        r_database[mi:mi+5]=database_desc[ri:ri+5]
        #r_night[mi:mi+5]=night_desc[ri:ri+5]
        newrow=pd.concat([labels.iloc[ri:ri+5, 5],labels.iloc[ri:ri+5, 9],labels.iloc[ri:ri+5, 0],labels.iloc[ri:ri+5, 4]], axis=1)
        r_labels=pd.concat([r_labels,newrow])
        
    np.save('random_query', r_query)
    np.save('random_database', r_database)
    #np.save('random_night', r_night)
    
    r_labels.to_csv('r_lables.csv')

