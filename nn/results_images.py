import re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

DIR='2022-09-12 01:33:20.075657'
TYPE='val'
FOLD='0'
EPOCH='0'
SERIE='5'


def plot(dir, type, fold, epoch, serie):

    labels=pd.read_csv('./random_labels.csv')

    if type=='val':
        results=pd.read_csv('../results/'+dir+'/'+'validation_results_F'+fold+'_E'+epoch+'.csv')
        query_dir='../dataset/downsampled_query/'
        database_dir='../dataset/downsampled_database/'
    elif type=="test":
        results=pd.read_csv('./results/'+dir+'/'+'test_results_F'+fold+'.csv')
        query_dir='../dataset/downsampled_query_test_2/'
        database_dir='../dataset/downsampled_database_test_2/'

    res=results.iloc[int(serie)]
    print(res)
    q_subset=labels.iloc[int(res.query)*5:int(res.query)*5+5]
    db_subset=labels.iloc[int(res.positive)*5:int(res.positive)*5+5]
    
    timestampsq=q_subset.q_timestamp.reset_index(drop=True)
    serieq=q_subset.serie.reset_index(drop=True)
    timestampsdb=db_subset.db_timestamp.reset_index(drop=True)
    seriedb=db_subset.serie.reset_index(drop=True)
    
    string=database_dir+str(timestampsdb[0])+'.png'
    pic1=mpimg.imread(string)
    string=database_dir+str(timestampsdb[1])+'.png'
    pic2=mpimg.imread(string)
    string=database_dir+str(timestampsdb[2])+'.png'
    pic3=mpimg.imread(string)
    string=database_dir+str(timestampsdb[3])+'.png'
    pic4=mpimg.imread(string)
    string=database_dir+str(timestampsdb[4])+'.png'
    pic5=mpimg.imread(string)
    
    string=query_dir+str(timestampsq[0])+'.png'
    pic6=mpimg.imread(string)
    string=query_dir+str(timestampsq[1])+'.png'
    pic7=mpimg.imread(string)
    string=query_dir+str(timestampsq[2])+'.png'
    pic8=mpimg.imread(string)
    string=query_dir+str(timestampsq[3])+'.png'
    pic9=mpimg.imread(string)
    string=query_dir+str(timestampsq[4])+'.png'
    pic10=mpimg.imread(string)
    
    rows=2
    columns=5
    
    fig = plt.figure(figsize=(20, 15))
    
    fig.add_subplot(rows, columns, 1)
    plt.imshow(pic1)
    plt.axis('off')
    plt.title(str(timestampsdb[0])+"\n"+ str(seriedb[0]))
    
    fig.add_subplot(rows, columns, 2)
    plt.imshow(pic2)
    plt.axis('off')
    plt.title(str(timestampsdb[1])+"\n"+ str(seriedb[1]))
    
    fig.add_subplot(rows, columns, 3)
    plt.imshow(pic3)
    plt.axis('off')
    plt.title(str(timestampsdb[2])+"\n"+ str(seriedb[2]))
    
    fig.add_subplot(rows, columns, 4)
    plt.imshow(pic4)
    plt.axis('off')
    plt.title(str(timestampsdb[3])+"\n"+ str(seriedb[3]))
    
    fig.add_subplot(rows, columns, 5)
    plt.imshow(pic5)
    plt.axis('off')
    plt.title(str(timestampsdb[4])+"\n"+ str(seriedb[4]))
    
    fig.add_subplot(rows, columns, 6)
    plt.imshow(pic6)
    plt.axis('off')
    plt.title(str(timestampsq[0])+"\n"+ str(serieq[0]))
    fig.add_subplot(rows, columns, 7)
    plt.imshow(pic7)
    plt.axis('off')
    plt.title(str(timestampsq[1])+"\n"+ str(serieq[1]))
    
    fig.add_subplot(rows, columns, 8)
    plt.imshow(pic8)
    plt.axis('off')
    plt.title(str(timestampsq[2])+"\n"+ str(serieq[2]))
    
    fig.add_subplot(rows, columns, 9)
    plt.imshow(pic9)
    plt.axis('off')
    plt.title(str(timestampsq[3])+"\n"+ str(serieq[3]))
    
    fig.add_subplot(rows, columns, 10)
    plt.imshow(pic10)
    plt.axis('off')
    plt.title(str(timestampsq[4])+"\n"+ str(serieq[4]))
    
    plt.show()

if __name__ == "__main__":
    plot(DIR, TYPE, FOLD, EPOCH, SERIE)