import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from multiprocessing.pool import Pool


img_size=224
def Resize(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE) #for gray-scale images
    image = cv2.resize(image,(img_size, img_size))
    return image

def Resize_images(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        imagemat = Resize(filename)
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.jpg', imagemat) #create the edge image and store it to consecutive filenames
        filecnt+=1
    print("\n\n--saved in " + destdir + "--\n")

sourcedir = ('/home/linh/Downloads/CXR/data/bacterial')
destdir = ('/home/linh/Downloads/CXR/ori_resized/bacterial')

#sourcedir = ('/home/linh/Downloads/CXR/data/normal')
#destdir = ('/home/linh/Downloads/CXR/ori_resized/normal')

#sourcedir = ('/home/linh/Downloads/CXR/data/virus')
#destdir = ('/home/linh/Downloads/CXR/ori_resized/virus')
os.makedirs(destdir, exist_ok=True)
print("The new directory is created!")
#with Pool(28) as p:
#    p.map(Resize_images(sourcedir, destdir))
Resize_images(sourcedir, destdir)