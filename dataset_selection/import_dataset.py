import image
from PIL import Image
import pandas as pd
def main():
    labels=pd.read_csv('selected_query.csv')
    timestamps=labels.timestamp.reset_index(drop=True)
    burst=labels.burst.reset_index(drop=True)
    for row in labels.index:
        string='/media/simone/Giochi/dataset/2015-08-13-16-02-58/2015-08-13-16-02-58_stereo_centre_0'+str(burst[row])+'/2015-08-13-16-02-58/stereo/centre/'+str(timestamps[row])+'.png'
        img=image.load_image(string)
        immagine=Image.fromarray(img)
        immagine=immagine.resize((426,320), Image.ANTIALIAS)
        immagine.save('../dataset/downsampled_query/'+str(timestamps[row])+'.png')
        
if __name__ == "__main__":
    main()