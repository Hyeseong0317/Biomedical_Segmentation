# Image.fromarray is poorly defined with floating-point input; it's not well documented but the function assumes the input is laid-out as unsigned 8-bit integers.
# To produce the output you're trying to get, multiply by 255 and convert to uint8:
z = (z * 255).astype(np.uint8)

# numpy to jpg
from PIL import Image
im = Image.fromarray(A)
im.save("your_file.jpg")

# jpg to numpy
import numpy as np
from PIL import Image
img = Image.open("NASA.jpg")
imgArray = np.asarray(img)
print(imgArray.shape)


# numpy to jpg
from PIL import Image

from glob import glob
data_files = glob('/mnt/intern/3D_vessel/demo_code/data/test/original2D/*.npy')
examples = []
for fname in sorted(data_files):  
    examples += [fname]

for i in range(len((data_files))):
    img_path = examples[i] # img path
    A = np.load(img_path)
    im = Image.fromarray(A)
    
    if im.mode != 'RGB':    # 1 channel -> 3 channel
        im = im.convert('RGB')
        
    new_img_path = img_path.replace('original2D', 'original2DforDCGAN') # img path
    new_img_path = img_path.replace('npy', 'jpg')
    print(new_img_path)
    im.save(new_img_path)
