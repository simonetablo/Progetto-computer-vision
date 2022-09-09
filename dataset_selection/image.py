
import re
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
import numpy as np

BAYER_STEREO = 'gbrg'
BAYER_MONO = 'rggb'


def load_image(image_path, model=None):
    """Loads and rectifies an image from file.

    Args:
        image_path (str): path to an image from the dataset.
        model (camera_model.CameraModel): if supplied, model will be used to undistort image.

    Returns:
        numpy.ndarray: demosaiced and optionally undistorted image

    """
    if model:
        camera = model.camera
    else:
        camera = re.search('(stereo|mono_(left|right|rear))', image_path).group(0)
    if camera == 'stereo':
        pattern = BAYER_STEREO
    else:
        pattern = BAYER_MONO

    img = Image.open(image_path)
    img = demosaic(img, pattern)
    if model:
        img = model.undistort(img)

    return np.array(img).astype(np.uint8)

#Prove burst immagini
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

def main():
    img_index=[0,0,0,0,0]
    plot(137, img_index)
    
def plot(num, img_index):    
    database=pd.read_csv('./dataset_selection/selected_database_test_2.csv')
    query=pd.read_csv('./dataset_selection/selected_query_test_2.csv')

    rowsdb=database[(database.serie==num)]
    rowsq=query[(query.serie==num)]
    
    timestampsdb=rowsdb.timestamp.reset_index(drop=True)
    burstdb=rowsdb.burst.reset_index(drop=True)
    
    timestampsq=rowsq.timestamp.reset_index(drop=True)
    burstq=rowsq.burst.reset_index(drop=True)
    
    
    
    string='/home/simone/Scaricati/2014-06-24-14-47-45/2014-06-24-14-47-45_stereo_centre_0'+str(burstdb[0])+'/2014-06-24-14-47-45/stereo/centre/'+str(timestampsdb[0])+'.png'
    pic1=load_image(string)
    string='/home/simone/Scaricati/2014-06-24-14-47-45/2014-06-24-14-47-45_stereo_centre_0'+str(burstdb[1])+'/2014-06-24-14-47-45/stereo/centre/'+str(timestampsdb[1])+'.png'
    pic2=load_image(string)
    string='/home/simone/Scaricati/2014-06-24-14-47-45/2014-06-24-14-47-45_stereo_centre_0'+str(burstdb[2])+'/2014-06-24-14-47-45/stereo/centre/'+str(timestampsdb[2])+'.png'
    pic3=load_image(string)
    string='/home/simone/Scaricati/2014-06-24-14-47-45/2014-06-24-14-47-45_stereo_centre_0'+str(burstdb[3])+'/2014-06-24-14-47-45/stereo/centre/'+str(timestampsdb[3])+'.png'
    pic4=load_image(string)
    string='/home/simone/Scaricati/2014-06-24-14-47-45/2014-06-24-14-47-45_stereo_centre_0'+str(burstdb[4])+'/2014-06-24-14-47-45/stereo/centre/'+str(timestampsdb[4])+'.png'
    pic5=load_image(string)
    
    string='/home/simone/Scaricati/2014-06-25-17-02-32/2014-06-25-17-02-32_stereo_centre_0'+str(burstq[0])+'/2014-06-25-17-02-32/stereo/centre/'+str(timestampsq[0])+'.png'
    pic6=load_image(string)
    string='/home/simone/Scaricati/2014-06-25-17-02-32/2014-06-25-17-02-32_stereo_centre_0'+str(burstq[1])+'/2014-06-25-17-02-32/stereo/centre/'+str(timestampsq[1])+'.png'
    pic7=load_image(string)
    string='/home/simone/Scaricati/2014-06-25-17-02-32/2014-06-25-17-02-32_stereo_centre_0'+str(burstq[2])+'/2014-06-25-17-02-32/stereo/centre/'+str(timestampsq[2])+'.png'
    pic8=load_image(string)
    string='/home/simone/Scaricati/2014-06-25-17-02-32/2014-06-25-17-02-32_stereo_centre_0'+str(burstq[3])+'/2014-06-25-17-02-32/stereo/centre/'+str(timestampsq[3])+'.png'
    pic9=load_image(string)
    string='/home/simone/Scaricati/2014-06-25-17-02-32/2014-06-25-17-02-32_stereo_centre_0'+str(burstq[4])+'/2014-06-25-17-02-32/stereo/centre/'+str(timestampsq[4])+'.png'
    pic10=load_image(string)
    
    rows=2
    columns=5
    
    fig = plt.figure(figsize=(20, 15))
    
    fig.add_subplot(rows, columns, 1)
    plt.imshow(pic1)
    plt.axis('off')
    plt.title(str(timestampsdb[0])+"\n 1.1")
    
    fig.add_subplot(rows, columns, 2)
    plt.imshow(pic2)
    plt.axis('off')
    plt.title(str(timestampsdb[1])+"\n 1.2")
    
    fig.add_subplot(rows, columns, 3)
    plt.imshow(pic3)
    plt.axis('off')
    plt.title(str(timestampsdb[2])+"\n 1.3")
    
    fig.add_subplot(rows, columns, 4)
    plt.imshow(pic4)
    plt.axis('off')
    plt.title(str(timestampsdb[3])+"\n 1.4")
    
    fig.add_subplot(rows, columns, 5)
    plt.imshow(pic5)
    plt.axis('off')
    plt.title(str(timestampsdb[4])+"\n 1.5")
    
    fig.add_subplot(rows, columns, 6)
    plt.imshow(pic6)
    plt.axis('off')
    plt.title(str(timestampsq[0])+"\n 2.1 , "+str(img_index[0]))
    
    fig.add_subplot(rows, columns, 7)
    plt.imshow(pic7)
    plt.axis('off')
    plt.title(str(timestampsq[1])+"\n 2.2 , "+str(img_index[1]))
    
    fig.add_subplot(rows, columns, 8)
    plt.imshow(pic8)
    plt.axis('off')
    plt.title(str(timestampsq[2])+"\n 2.3 , "+str(img_index[2]))
    
    fig.add_subplot(rows, columns, 9)
    plt.imshow(pic9)
    plt.axis('off')
    plt.title(str(timestampsq[3])+"\n 2.4 , "+str(img_index[3]))
    
    fig.add_subplot(rows, columns, 10)
    plt.imshow(pic10)
    plt.axis('off')
    plt.title(str(timestampsq[4])+"\n 2.5 , "+str(img_index[4]))
    
    plt.show()
    
if __name__ == "__main__":
    main()