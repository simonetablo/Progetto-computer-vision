import image
from PIL import Image
import pandas as pd
def main(dataset):
    labels=pd.read_csv('./dataset_selection/selected_'+dataset+'.csv')
    timestamps=labels.timestamp.reset_index(drop=True)
    burst=labels.burst.reset_index(drop=True)
    
    if dataset=='query':
        str1='/media/simone/Giochi/dataset/2015-08-13-16-02-58/2015-08-13-16-02-58_stereo_centre_0'
        str2='/2015-08-13-16-02-58/stereo/centre/'
    elif dataset=='database':
        str1='/media/simone/Giochi/dataset/2015-11-13-10-28-08/2015-11-13-10-28-08_stereo_centre_0'
        str2='/2015-11-13-10-28-08/stereo/centre/'
        
    for row in labels.index:
        string=str1+str(burst[row])+str2+str(timestamps[row])+'.png'
        img=image.load_image(string)
        immagine=Image.fromarray(img)
        immagine=immagine.resize((426,320), Image.ANTIALIAS)
        immagine.save('./dataset/downsampled_'+dataset+'/'+str(timestamps[row])+'.png')
        
if __name__ == "__main__":
    main(dataset='database')