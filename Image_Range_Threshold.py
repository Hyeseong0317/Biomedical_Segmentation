import numpy as np
import pathlib
import os
import scipy.io
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as img

import scipy.ndimage   
 
def Rotation_3d(input, angle):
    """ Rotate an image by an angle 1.

    Returns:
    batch of rotated 3D images
    """
    
    angle = angle*10
 
   # X sum rotate
    rotated_Xsum = scipy.ndimage.interpolation.rotate(input, angle, mode='nearest', axes=(0, 1), reshape=False)
    # Y sum rotate
    rotated_Ysum = scipy.ndimage.interpolation.rotate(input, angle, mode='nearest', axes=(0, 1), reshape=False)
    # Z sum rotate
    rotated_Zsum = scipy.ndimage.interpolation.rotate(input, angle, mode='nearest', axes=(0, 1), reshape=False)

    return rotated_Xsum, rotated_Ysum, rotated_Zsum
    
    
rotated_Xsum, rotated_Ysum, rotated_Zsum = Rotation_3d(data_output, 10)

fig = plt.figure(figsize=(10,15))
rows = 1
cols = 3
FONTSIZE= 15

rotated_Xsum =  np.squeeze(rotated_Xsum.sum(axis=0))
ax7 = fig.add_subplot(rows, cols, 1)
ax7.imshow(rotated_Xsum, cmap='gray')
ax7.set_title('rotated_Xsum', fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
# ax3.axis("off")

rotated_Ysum =  np.squeeze(rotated_Ysum.sum(axis=1))
ax8 = fig.add_subplot(rows, cols, 2)
ax8.imshow(rotated_Ysum, cmap='gray')
ax8.set_title('rotated_Ysum', fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
# ax3.axis("off")

rotated_Zsum =  np.squeeze(rotated_Zsum.sum(axis=2))
ax9 = fig.add_subplot(rows, cols, 3)
ax9.imshow(rotated_Zsum, cmap='gray')
ax9.set_title('rotated_Zsum', fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
# ax3.axis("off")
plt.show()

#%%
# Get Frames x, y, z
x_frames = []
y_frames = []
z_frames = []
for i in range(60):
    
    x_frame, y_frame, z_frame = Rotation_3d(data_output, i)
    
    x_frames.append(x_frame)
    y_frames.append(y_frame)
    z_frames.append(z_frame)

print('x_frames : ', len(x_frames))
print('y_frames : ', len(y_frames))
print('z_frames : ', len(z_frames))

x_frames_test = []
y_frames_test = []
z_frames_test = []
#%%
# Get Summation of each axis
for i in range(len(x_frames)):
    x_frames_test.append(np.squeeze(x_frames[i].sum(axis=0)))
    y_frames_test.append(np.squeeze(y_frames[i].sum(axis=1)))
    z_frames_test.append(np.squeeze(z_frames[i].sum(axis=2)))
#%%
# Visualize Test
for i in range(len(x_frames)):
    fig = plt.figure(figsize=(10,15))
    rows = 3
    cols = 3
    FONTSIZE= 15

    data_output_3D_xaxis =  np.squeeze(x_frames[i].sum(axis=0))
    ax7 = fig.add_subplot(rows, cols, 7)
    ax7.imshow(data_output_3D_xaxis, cmap='gray')
    ax7.set_title('output', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    # ax3.axis("off")

    data_output_3D_yaxis =  np.squeeze(y_frames[i].sum(axis=1))
    ax8 = fig.add_subplot(rows, cols, 8)
    ax8.imshow(data_output_3D_yaxis, cmap='gray')
    ax8.set_title('output', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    # ax3.axis("off")

    data_output_3D_zaxis =  np.squeeze(z_frames[i].sum(axis=2))
    ax9 = fig.add_subplot(rows, cols, 9)
    ax9.imshow(data_output_3D_zaxis, cmap='gray')
    ax9.set_title('output', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    # ax3.axis("off")
    plt.show()

#%%
#  Get Frames y_frames 
img_path = pathlib.Path('/mnt/intern/yframes')
from PIL import Image
for i in range(len(y_frames_test)):

    img_array = y_frames_test[i]
    normalized_img = cv2.normalize(img_array,  None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_img = Image.fromarray(normalized_img.astype(np.uint8))
    img_path = pathlib.Path('/mnt/intern/yframes')
    number = pathlib.Path(str(i) + '.jpeg')
    img_path = img_path / number
    print(img_path)
    img_path = str(img_path)
    
    normalized_img.save(img_path)

#%% 
#  Get Frames x_frames     
img_path = pathlib.Path('/mnt/intern/xframes')
from PIL import Image
for i in range(len(x_frames_test)):

    img_array = x_frames_test[i]
    normalized_img = cv2.normalize(img_array,  None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_img = Image.fromarray(normalized_img.astype(np.uint8))
    img_path = pathlib.Path('/mnt/intern/xframes')
    number = pathlib.Path(str(i) + '.jpeg')
    img_path = img_path / number
    print(img_path)
    img_path = str(img_path)
    
    normalized_img.save(img_path)

#%%
#  Get Frames z_frames  
img_path = pathlib.Path('/mnt/intern/zframes')
from PIL import Image
for i in range(len(z_frames_test)):

    img_array = z_frames_test[i]
    normalized_img = cv2.normalize(img_array,  None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_img = Image.fromarray(normalized_img.astype(np.uint8))
    img_path = pathlib.Path('/mnt/intern/zframes')
    number = pathlib.Path(str(i) + '.jpeg')
    img_path = img_path / number
    print(img_path)
    img_path = str(img_path)
    
    normalized_img.save(img_path)

#%% 
#  Get Frames x_frames    최대값 5로 제한해서 미세혈관 시각화하기
img_path = pathlib.Path('/mnt/intern/xframes')
from PIL import Image
for i in range(len(x_frames_test)):

    img_array = x_frames_test[i]
    img_array = img_array.astype(np.uint8)
    for j in range(img_array.shape[0]):
        for k in range(img_array.shape[1]):
            if img_array[j][k] >= 5:
                img_array[j][k] = 5
            else:
                pass

    normalized_img = cv2.normalize(img_array,  None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    print(normalized_img) 
    out = Image.fromarray(normalized_img)
    
    img_path = pathlib.Path('/mnt/intern/xframes')
    number = pathlib.Path(str(i) + '.jpeg')
    img_path = img_path / number
    print(img_path)
    img_path = str(img_path)

    out.save(img_path)
    
    #%%
#  Get Frames y_frames    최대값 5로 제한해서 미세혈관 시각화하기
img_path = pathlib.Path('/mnt/intern/yframes')
from PIL import Image
for i in range(len(y_frames_test)):

    img_array = y_frames_test[i]
    img_array = img_array.astype(np.uint8)
    for j in range(img_array.shape[0]):
        for k in range(img_array.shape[1]):
            if img_array[j][k] >= 5:
                img_array[j][k] = 5
            else:
                pass

    normalized_img = cv2.normalize(img_array,  None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_img = Image.fromarray(normalized_img.astype(np.uint8))
    img_path = pathlib.Path('/mnt/intern/yframes')
    number = pathlib.Path(str(i) + '.jpeg')
    img_path = img_path / number
    print(img_path)
    img_path = str(img_path)
    
    normalized_img.save(img_path)

#%%
#  Get Frames z_frames     최대값 5로 제한해서 미세혈관 시각화하기
img_path = pathlib.Path('/mnt/intern/zframes')
from PIL import Image
for i in range(len(z_frames_test)):
    
    img_array = z_frames_test[i]
    img_array = img_array.astype(np.uint8)
    for j in range(img_array.shape[0]):
        for k in range(img_array.shape[1]):
            if img_array[j][k] >= 5:
                img_array[j][k] = 5
            else:
                pass

    normalized_img = cv2.normalize(img_array,  None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_img = Image.fromarray(normalized_img.astype(np.uint8))
    img_path = pathlib.Path('/mnt/intern/zframes')
    number = pathlib.Path(str(i) + '.jpeg')
    img_path = img_path / number
    print(img_path)
    img_path = str(img_path)
    
    normalized_img.save(img_path)

#%%
# Create X Rotation Video
# initialize video writer
import os
X_FRAMES = os.listdir('/mnt/intern/xframes/')
X_FRAMES_SORTED = []
print(X_FRAMES)

for i in sorted(X_FRAMES, key=lambda x: int(x.split(".")[0])):
    print(i)
    X_FRAMES_SORTED.append(i)

print(X_FRAMES_SORTED)

height = 256
width = 400
X_FRAMES_PATH = '/mnt/intern/xframes/'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = 15
pathOut = '/mnt/intern/xframes/output.avi'
out = cv2.VideoWriter(pathOut, fourcc, fps, (width, height))

frame_array = []
for i in range(len(X_FRAMES_SORTED)):
    filename= X_FRAMES_PATH + X_FRAMES_SORTED[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    print(filename)
    #inserting the frames into an image array
    frame_array.append(img)
# new frame after each addition of water
for i in range(len(frame_array)):
     #add this array to the video
    out.write(frame_array[i])

# close out the video writer
out.release()

#%%
# Create Y Rotation Video 
# initialize video writer
import os
Y_FRAMES = os.listdir('/mnt/intern/yframes/')
Y_FRAMES_SORTED = []
print(Y_FRAMES)

for i in sorted(Y_FRAMES, key=lambda x: int(x.split(".")[0])):
    print(i)
    Y_FRAMES_SORTED.append(i)

print(Y_FRAMES_SORTED)

height = 256
width = 400
Y_FRAMES_PATH = '/mnt/intern/yframes/'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = 15
pathOut = '/mnt/intern/yframes/output.avi'
out = cv2.VideoWriter(pathOut, fourcc, fps, (width, height))

frame_array = []
for i in range(len(Y_FRAMES_SORTED)):
    filename= Y_FRAMES_PATH + Y_FRAMES_SORTED[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    print(filename)
    #inserting the frames into an image array
    frame_array.append(img)
# new frame after each addition of water
for i in range(len(frame_array)):
     #add this array to the video
    out.write(frame_array[i])

# close out the video writer
out.release()

#%%
# Create Z Rotation Video
# initialize video writer
import os
Z_FRAMES = os.listdir('/mnt/intern/zframes/')
Z_FRAMES_SORTED = []
print(Z_FRAMES)

for i in sorted(Z_FRAMES, key=lambda x: int(x.split(".")[0])):
    print(i)
    Z_FRAMES_SORTED.append(i)

print(Z_FRAMES_SORTED)

height = 256
width = 256
Z_FRAMES_PATH = '/mnt/intern/zframes/'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = 15
pathOut = '/mnt/intern/zframes/output.avi'
out = cv2.VideoWriter(pathOut, fourcc, fps, (width, height))

frame_array = []
for i in range(len(Y_FRAMES_SORTED)):
    filename= Z_FRAMES_PATH + Z_FRAMES_SORTED[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    print(filename)
    #inserting the frames into an image array
    frame_array.append(img)
# new frame after each addition of water
for i in range(len(frame_array)):
     #add this array to the video
    out.write(frame_array[i])

# close out the video writer
out.release()
