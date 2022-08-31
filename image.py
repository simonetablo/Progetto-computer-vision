
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
import csv

labels=open('labels.csv')
reader=csv.reader(labels)

num=300

interestingrows=[row[0] for idx, row in enumerate(reader) if idx in range(num,num+6)]

str='/media/simone/Giochi/dataset/2015-11-13-10-28-08_stereo_centre_01/2015-11-13-10-28-08/stereo/centre/'+interestingrows[0]+'.png'
pic1=load_image(str)
str='/media/simone/Giochi/dataset/2015-11-13-10-28-08_stereo_centre_01/2015-11-13-10-28-08/stereo/centre/'+interestingrows[1]+'.png'
pic2=load_image(str)
str='/media/simone/Giochi/dataset/2015-11-13-10-28-08_stereo_centre_01/2015-11-13-10-28-08/stereo/centre/'+interestingrows[2]+'.png'
pic3=load_image(str)
str='/media/simone/Giochi/dataset/2015-11-13-10-28-08_stereo_centre_01/2015-11-13-10-28-08/stereo/centre/'+interestingrows[3]+'.png'
pic4=load_image(str)
str='/media/simone/Giochi/dataset/2015-11-13-10-28-08_stereo_centre_01/2015-11-13-10-28-08/stereo/centre/'+interestingrows[4]+'.png'
pic5=load_image(str)
str='/media/simone/Giochi/dataset/2015-11-13-10-28-08_stereo_centre_01/2015-11-13-10-28-08/stereo/centre/'+interestingrows[5]+'.png'
pic6=load_image(str)

rows=2
columns=3

fig = plt.figure(figsize=(10, 7))

fig.add_subplot(rows, columns, 1)
plt.imshow(pic1)
plt.axis('off')
plt.title("1")
  
fig.add_subplot(rows, columns, 2)
plt.imshow(pic2)
plt.axis('off')
plt.title("2")

fig.add_subplot(rows, columns, 3)
plt.imshow(pic3)
plt.axis('off')
plt.title("3")

fig.add_subplot(rows, columns, 4)
plt.imshow(pic4)
plt.axis('off')
plt.title("4")

fig.add_subplot(rows, columns, 5)
plt.imshow(pic5)
plt.axis('off')
plt.title("5")

fig.add_subplot(rows, columns, 6)
plt.imshow(pic6)
plt.axis('off')
plt.title("6")

plt.show()