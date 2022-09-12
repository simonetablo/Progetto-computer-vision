import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random

#torch.set_default_tensor_type(torch.DoubleTensor)

class getTrainDescriptors(Dataset):
    def __init__(self, desc_dir, npq, split, query_desc, database_desc, database2_desc, transform=True):
        self.desc_dir=desc_dir
        self.npq=npq
        self.split=split
        self.transform=transform
        query_desc1=np.load('./'+desc_dir+query_desc)[split[0]:split[1],]
        query_desc2=np.load('./'+desc_dir+query_desc)[split[2]:split[3],]
        database_desc1=np.load('./'+desc_dir+database_desc)[split[0]:split[1],]
        database_desc2=np.load('./'+desc_dir+database_desc)[split[2]:split[3],]
        database2_desc1=np.load('./'+desc_dir+database2_desc)[split[0]:split[1],]
        database2_desc2=np.load('./'+desc_dir+database2_desc)[split[2]:split[3],]

        self.query_desc=np.concatenate((query_desc1, query_desc2), axis=0)
        self.database_desc=np.concatenate((database_desc1, database_desc2), axis=0)
        self.database2_desc=np.concatenate((database2_desc1, database2_desc2), axis=0)
        
    def __len__(self):
        return int(len(self.database_desc)/5)
    
    def __getitem__(self, idx):
        mult=idx*5
        indices=[]
        q_desc=self.query_desc[mult:mult+5, :]
        p1_desc=self.database_desc[mult:mult+5, :]
        p2_desc=self.database2_desc[mult:mult+5, :]
        p_desc=np.stack((p1_desc, p2_desc), axis=0)
        indices.append(idx)
        indices.append(idx)
        indices.append(idx)
        all_n_desc=np.concatenate((self.database_desc[:idx],self.database_desc[idx+5:],self.database2_desc[:idx],self.database2_desc[idx+5:]))
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
    query, positives, negatives, indices = zip(*batch)
    query = data.dataloader.default_collate(query)
    positives = torch.cat(positives, 0)
    negatives = torch.cat(negatives, 0)
    indices = data.dataloader.default_collate(indices)
    #indices = torch.cat(indices, 0)
    return query, positives, negatives, indices


def shuffle_dataset():
    query_desc_file='globalfeats_q.npy'
    database_desc_file='globalfeats_db.npy'
    night_desc_file='globalfeats_night.npy'
    snow_desc_file='globalfeats_snow.npy'
    
    all_labels=pd.read_csv('../dataset_selection/all_labels.csv')

    query_desc=np.load('./'+query_desc_file)
    database_desc=np.load('./'+database_desc_file)
    night_desc=np.load('./'+night_desc_file)
    snow_desc=np.load('./'+snow_desc_file)
    
    if query_desc.shape[0]<database_desc.shape[0]:
        ridx=int(query_desc.shape[0]/5)
        database_desc=database_desc[:query_desc.shape[0]]
    else:
        ridx=int(database_desc.shape[0]/5)
        query_desc=query_desc[:database_desc.shape[0]]
        
    random_indices=random.sample(range(ridx), ridx)
    r_query=np.zeros((query_desc.shape[0], query_desc.shape[1]))
    r_database=np.zeros((database_desc.shape[0], database_desc.shape[1]))
    r_night=np.zeros((night_desc.shape[0], night_desc.shape[1]))
    r_snow=np.zeros((snow_desc.shape[0], snow_desc.shape[1]))
    
    r_labels=pd.DataFrame(columns=['q_timestamp','db_timestamp','snow_timestamp','night_timestamp','serie','new_serie'])

    for i in range(ridx):
        mi=i*5
        ri=random_indices[i]*5
        r_query[mi:mi+5]=query_desc[ri:ri+5]
        r_database[mi:mi+5]=database_desc[ri:ri+5]
        r_night[mi:mi+5]=night_desc[ri:ri+5]
        r_snow[mi:mi+5]=snow_desc[ri:ri+5]
        ns=[i,i,i,i,i]
        r_labels=pd.concat([r_labels,all_labels[ri:ri+5]], ignore_index=True)
        
    np.save('random_query', r_query)
    np.save('random_database', r_database)
    np.save('random_night', r_night)
    np.save('random_snow', r_snow)
    
    r_labels.to_csv('random_labels.csv')




if __name__=="__main__":
    
    query='globalfeats_q.npy'
    database='globalfeats_db.npy'
    split_train=[0, 2000]
    split_test=[2000, 2415]
    negatives_per_query=10


    database_desc=np.load('./'+query)
    print(database_desc.shape)

    datasettrain=getTrainDescriptors(desc_dir='', npq=negatives_per_query, split=split_train, query_desc=query, database_desc=database)
    dataset=getTestDescriptors(desc_dir='', split=split_test, query_desc=query, database_desc=database)
    for Data in dataset:
        qdesc,pdesc=Data
        print(qdesc.shape)
        print(pdesc.shape)
        break

    #
    #first_data=dataset[0]
    #q, p, n, nn=first_data
    #print('q: ',q.shape)
    #print('p: ',p.shape)
    #print('n: ',n.shape)
    #
    #print('q: ',type(q))
    #print('p: ',type(p))
    #print('n: ',type(n))
    #
    #    
    #
    #class getImages(Dataset):
    #    def __init__(self, csv_file, img_dir, transform=None):
    #        self.labels=pd.read_csv(csv_file)
    #        self.img_dir=img_dir
    #        self.transfrom=transform
    #        
    #    def __len__(self):
    #        return len(self.lables*2)
    #    
    #    def __getitem__(self, idx):
    #        mult=idx*5
    #        query=os.path.join(self.img_dir, self.labels.iloc[mult:mult+5, 5])
    #        positive=os.path.join(self.img_dir, self.labels.iloc[mult:mult+5, 0])
    #
    #        negatives=[]
    #        for n in labels:
    #        
    #        return q_imgs, p_imgs, n_imgs