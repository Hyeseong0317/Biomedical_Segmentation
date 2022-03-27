#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io

def Dice_coef(y_pred, y_true, smooth=1.):

    y_pred_0 = y_pred[0, :, :, :].contiguous().view(-1)
    y_true_0 = y_true[0, :, :, :].contiguous().view(-1)
    intersection_0 = (y_pred_0 * y_true_0).sum()

    Dice_AR = (2. * intersection_0 + smooth) / (y_pred_0.sum() + y_true_0.sum() + smooth)

    return Dice_AR

def DiceLoss(y_pred, y_true, smooth=1.):

    y_pred_0 = y_pred[0, :, :, :].contiguous().view(-1)
    y_true_0 = y_true[0, :, :, :].contiguous().view(-1)
    intersection_0 = (y_pred_0 * y_true_0).sum()

    Dice_AR = (2. * intersection_0 + smooth) / (y_pred_0.sum() + y_true_0.sum() + smooth)

    return 1 - Dice_AR

connectivity_4 = np.array([[0,1,0],
                           [1,0,0],
                           [0,0,0]])

connectivity_8 = np.array([[1,1,1],
                           [1,0,0],
                           [0,0,0]])

def find_labels(labels, r, c, neighbors):
    
    tmp_labels = labels[r-1:r+2, c-1:c+2]*neighbors
    
    return np.sort(tmp_labels[np.nonzero(tmp_labels)]) # Ascending Order

def connected_component_labeling(bin_img, connectivity=connectivity_8):
    equivalent = []
    bin_img = np.pad(bin_img, (1,1), mode='constant', constant_values=0)
    labels = np.zeros_like(bin_img, dtype='int64')
    next_label = 1
    
    # 1st pass
    for r, row in enumerate(bin_img):
        for c, pixel in enumerate(row):
            if pixel!=0:
                neighbors = bin_img[r-1:r+2, c-1:c+2]*connectivity
                num_neighbors = np.count_nonzero(neighbors)
                
                if num_neighbors == 0:
                    labels[r,c] = next_label
                    equivalent.append([next_label,next_label])
                    next_label += 1
                else:
                    L = find_labels(labels, r, c, neighbors)
                    labels[r,c] = np.min(L) # Label collision --> assign the lowest value
                
                    uni_L = np.unique(L) # unique value in a window
                    if len(uni_L)>1: 
                        for i, e in enumerate(equivalent):
                            if uni_L[0] in e: 
                                equivalent[i].extend(uni_L[1:]) # if) uni_L[0] exists in a equivalent table, extend new unique values to the equivalent table
                                equivalent[i] = list(sorted(set(equivalent[i]))) # 
                  
# 2nd pass
    for e in equivalent: # e = [1,2] / e[0]=1, e[1]=2
        for f in reversed(e): # f = 2, 1
            labels[labels==f] = e[0] # f=2- >e[0]=1, f=1-->e[0]=1 ---> Convert all values to the lowest value in the window
    return labels, equivalent

def largest_labels(labels):
      
    unique_elements, counts_elements = np.unique(labels, return_counts=True)
    # print(unique_elements, counts_elements)

    largest_component_counts = sorted(counts_elements, reverse=True)[1]
    idx = np.where(counts_elements == largest_component_counts) # 값이 가장 많은 label의 index반환
    largest_component = unique_elements[idx] # largest component와  index가 매치하는 value 선택
    labels = np.where(labels == largest_component, 0, labels) # largest component 제거
    return labels
  
def threshold(labels):
    labels[labels>0] = 1
    return labels
  
#%%
#3D detailed DSC
DSCs = []
losses = []
for i in [119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 130]:
    print('10000' + str(i) + '.mat')
    data_label = scipy.io.loadmat('/mnt/intern/code/dataset/test/label/10000' + str(i) + '.mat')
    data_label = data_label['label']
    data_label = data_label[128:384, 128:384, 0:250] # (256, 256, 250)
    data_output = np.squeeze(np.load('/mnt/intern/code/result/3DUnet vanilla result batchsize1 instancenorm 전체데이터/10000' + str(i) + '.npy')) # (256, 256, 250)
    data_output = data_output[:, :, 0:250]

    for i in range(data_label.shape[2]):
    
        output, _ = connected_component_labeling(data_output[:, :, i])
        label, _ = connected_component_labeling(data_label[:, :, i])
        
        Detailed_output = largest_labels(output) # (258, 258) zero padding
        Detailed_label = largest_labels(label)  # (258, 258) zero padding
        
        Detailed_output = Detailed_output[1:-1, 1:-1] # (256, 256)
        Detailed_label = Detailed_label[1:-1, 1:-1] # (256, 256)
        
        Detailed_output = threshold(Detailed_output)
        Detailed_label = threshold(Detailed_label)
        
        data_label[:, :, i] = Detailed_label
        data_output[:, :, i] = Detailed_output

    data_label = np.expand_dims(data_label, axis=0)
    data_output =  np.expand_dims(data_output, axis=0)
    data_label = torch.from_numpy(data_label)
    data_output = torch.from_numpy(data_output)
    DSC = Dice_coef(data_output, data_label)
    loss = DiceLoss(data_output, data_label)
    
    DSCs.append(DSC)
    losses.append(loss)
    print(DSC)
    print(loss)
    
print(np.mean(DSCs))
print(np.mean(losses))
#%%
#2D detailed DSC
DSCs = []
losses = []
for i in [119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 130]:
    print('10000' + str(i) + '.mat')
    data_label = scipy.io.loadmat('/mnt/intern/code/dataset/test/label/10000' + str(i) + '.mat')
    data_label = data_label['label']
    data_label = data_label[128:384, 128:384, 0:250] # (256, 256, 250)
    data_output = np.squeeze(np.load('/mnt/intern/code/result2Dto3D/2DUnet vanilla result batchsize256 instancenorm 전체데이터 잘나옴/10000' + str(i) + 'slice.npy')) # (256, 256, 250)
    data_output = data_output[:, :, 0:250]

    for i in range(data_label.shape[2]):
    
        output, _ = connected_component_labeling(data_output[:, :, i])
        label, _ = connected_component_labeling(data_label[:, :, i])
        
        Detailed_output = largest_labels(output) # (258, 258) zero padding
        Detailed_label = largest_labels(label)  # (258, 258) zero padding
        
        Detailed_output = Detailed_output[1:-1, 1:-1] # (256, 256)
        Detailed_label = Detailed_label[1:-1, 1:-1] # (256, 256)
        
        Detailed_output = threshold(Detailed_output)
        Detailed_label = threshold(Detailed_label)
        
        data_label[:, :, i] = Detailed_label
        data_output[:, :, i] = Detailed_output

    data_label = np.expand_dims(data_label, axis=0)
    data_output =  np.expand_dims(data_output, axis=0)
    data_label = torch.from_numpy(data_label)
    data_output = torch.from_numpy(data_output)
    DSC = Dice_coef(data_output, data_label)
    loss = DiceLoss(data_output, data_label)

    DSCs.append(DSC)
    losses.append(loss)
    print(DSC)
    print(loss)
    
print(np.mean(DSCs))
print(np.mean(losses))

#%%
#Ensemble detailed DSC
DSCs = []
losses = []
for i in [119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 130]:
    print('10000' + str(i) + '.mat')
    data_label = scipy.io.loadmat('/mnt/intern/code/dataset/test/label/10000' + str(i) + '.mat')
    data_label = data_label['label']
    data_label = data_label[128:384, 128:384, 0:250] # (256, 256, 250)
    data_output = np.squeeze(np.load('/mnt/intern/code/ensemble_result/10000' + str(i) + '.npy')) # (256, 256, 250)
    data_output = data_output[:, :, 0:250]

    for i in range(data_label.shape[2]):
    
        output, _ = connected_component_labeling(data_output[:, :, i])
        label, _ = connected_component_labeling(data_label[:, :, i])
        
        Detailed_output = largest_labels(output) # (258, 258) zero padding
        Detailed_label = largest_labels(label)  # (258, 258) zero padding
        
        Detailed_output = Detailed_output[1:-1, 1:-1] # (256, 256)
        Detailed_label = Detailed_label[1:-1, 1:-1] # (256, 256)
        
        Detailed_output = threshold(Detailed_output)
        Detailed_label = threshold(Detailed_label)
        
        data_label[:, :, i] = Detailed_label
        data_output[:, :, i] = Detailed_output

    data_label = np.expand_dims(data_label, axis=0)
    data_output =  np.expand_dims(data_output, axis=0)
    data_label = torch.from_numpy(data_label)
    data_output = torch.from_numpy(data_output)
    DSC = Dice_coef(data_output, data_label)
    loss = DiceLoss(data_output, data_label)

    DSCs.append(DSC)
    losses.append(loss)
    print(DSC)
    print(loss)
    
print(np.mean(DSCs))
print(np.mean(losses))

#%%
# label확인
data_label = scipy.io.loadmat('/mnt/intern/code/dataset/test/label/10000120.mat')
data_label = data_label['label']
data_label = data_label[128:384, 128:384, 0:250] # (256, 256, 250)
data_label = data_label[:, :, 100]

result, equivalent = connected_component_labeling(data_label)

fig = plt.figure(figsize=(15,9))
rows = 1
cols = 2
FONTSIZE= 25
cmaps = plt.colormaps()

ax = fig.add_subplot(rows, cols, 1)
img = ax.imshow(result, cmap='jet') 
fig.colorbar(img, ax=ax)
ax.set_title('Focused DSC', fontsize=FONTSIZE)

result_label = largest_labels(result)
ax1 = fig.add_subplot(rows, cols, 2)
img2 = ax1.imshow(result_label, cmap='jet') 
fig.colorbar(img2, ax=ax1)
ax1.set_title('Focused DSC', fontsize=FONTSIZE)

# %%
#3D 확인
data_output = np.squeeze(np.load('/mnt/intern/code/result/3DUnet vanilla result batchsize1 instancenorm 전체데이터/10000120.npy')) # (256, 256, 250)
data_output = data_output[:,:,100]
print(data_output.shape)
result, equivalent = connected_component_labeling(data_output)
fig = plt.figure(figsize=(15,9))
rows = 1
cols = 2
FONTSIZE= 25
cmaps = plt.colormaps()

ax = fig.add_subplot(rows, cols, 1)
img = ax.imshow(result, cmap='jet') 
fig.colorbar(img, ax=ax)
ax.set_title('Focused DSC', fontsize=FONTSIZE)

result_label = largest_labels(result)
ax1 = fig.add_subplot(rows, cols, 2)
img2 = ax1.imshow(result_label, cmap='jet') 
fig.colorbar(img2, ax=ax1)
ax1.set_title('Focused DSC', fontsize=FONTSIZE)

#%%
#2D 확인
data_output = np.load('/mnt/intern/code/result2Dto3D/2DUnet vanilla result batchsize256 instancenorm 전체데이터 잘나옴/10000120slice.npy')
data_output = data_output[:, :, 100] # (256, 256)

result, equivalent = connected_component_labeling(data_output)
fig = plt.figure(figsize=(15,9))
rows = 1
cols = 2
FONTSIZE= 25
cmaps = plt.colormaps()

ax = fig.add_subplot(rows, cols, 1)
img = ax.imshow(result, cmap='jet') 
fig.colorbar(img, ax=ax)
ax.set_title('Focused DSC', fontsize=FONTSIZE)

result_label = largest_labels(result)
ax1 = fig.add_subplot(rows, cols, 2)
img2 = ax1.imshow(result_label, cmap='jet') 
fig.colorbar(img2, ax=ax1)
ax1.set_title('Focused DSC', fontsize=FONTSIZE)
# %%
#Ensemble 확인
data_output = np.squeeze(np.load('/mnt/intern/code/ensemble_result/10000120.npy'))
data_output = data_output[:, :, 100] # (256, 256)

result, equivalent = connected_component_labeling(data_output)
fig = plt.figure(figsize=(15,9))
rows = 1
cols = 2
FONTSIZE= 25
cmaps = plt.colormaps()

ax = fig.add_subplot(rows, cols, 1)
img = ax.imshow(result, cmap='jet') 
fig.colorbar(img, ax=ax)
ax.set_title('Focused DSC', fontsize=FONTSIZE)

result_label = largest_labels(result)
ax1 = fig.add_subplot(rows, cols, 2)
img2 = ax1.imshow(result_label, cmap='jet') 
fig.colorbar(img2, ax=ax1)
ax1.set_title('Focused DSC', fontsize=FONTSIZE)

#%%
# nnU-Net 3D detailed DSC
DSCs = []
losses = []
for i in [119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 130]:
    print('10000' + str(i) + '.mat')
    data_label = scipy.io.loadmat('/mnt/intern/code/dataset/test/label/10000' + str(i) + '.mat')
    data_label = data_label['label']
    data_label = data_label[128:384, 128:384, 0:250] # (256, 256, 250)
    data_output = scipy.io.loadmat('/mnt/intern/code/nnUnet_result/3d_fullres/10000' + str(i) + '.mat')
    data_output = data_output['data']
    data_output = data_output[128:384, 128:384, 0:250] # (256, 256, 250)

    for i in range(data_label.shape[2]):
    
        output, _ = connected_component_labeling(data_output[:, :, i])
        label, _ = connected_component_labeling(data_label[:, :, i])
        
        Detailed_output = largest_labels(output) # (258, 258) zero padding
        Detailed_label = largest_labels(label)  # (258, 258) zero padding
        
        Detailed_output = Detailed_output[1:-1, 1:-1] # (256, 256)
        Detailed_label = Detailed_label[1:-1, 1:-1] # (256, 256)
        
        Detailed_output = threshold(Detailed_output)
        Detailed_label = threshold(Detailed_label)
        
        data_label[:, :, i] = Detailed_label
        data_output[:, :, i] = Detailed_output

    data_label = np.expand_dims(data_label, axis=0)
    data_output =  np.expand_dims(data_output, axis=0)
    data_label = torch.from_numpy(data_label)
    data_output = torch.from_numpy(data_output)
    DSC = Dice_coef(data_output, data_label)
    loss = DiceLoss(data_output, data_label)
    
    DSCs.append(DSC)
    losses.append(loss)
    print(DSC)
    print(loss)
    
print(np.mean(DSCs))
print(np.mean(losses))
#%%
# nnU-Net 2D detailed DSC
DSCs = []
losses = []
for i in [119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 130]:
    print('10000' + str(i) + '.mat')
    data_label = scipy.io.loadmat('/mnt/intern/code/dataset/test/label/10000' + str(i) + '.mat')
    data_label = data_label['label']
    data_label = data_label[128:384, 128:384, 0:250] # (256, 256, 250)
    data_output = scipy.io.loadmat('/mnt/intern/code/nnUnet_result/2d/10000' + str(i) + '.mat')
    data_output = data_output['data']
    data_output = data_output[128:384, 128:384, 0:250] # (256, 256, 250)
    
    for i in range(data_label.shape[2]):
    
        output, _ = connected_component_labeling(data_output[:, :, i])
        label, _ = connected_component_labeling(data_label[:, :, i])
        
        Detailed_output = largest_labels(output) # (258, 258) zero padding
        Detailed_label = largest_labels(label)  # (258, 258) zero padding
        
        Detailed_output = Detailed_output[1:-1, 1:-1] # (256, 256)
        Detailed_label = Detailed_label[1:-1, 1:-1] # (256, 256)
        
        Detailed_output = threshold(Detailed_output)
        Detailed_label = threshold(Detailed_label)
        
        data_label[:, :, i] = Detailed_label
        data_output[:, :, i] = Detailed_output

    data_label = np.expand_dims(data_label, axis=0)
    data_output =  np.expand_dims(data_output, axis=0)
    data_label = torch.from_numpy(data_label)
    data_output = torch.from_numpy(data_output)
    DSC = Dice_coef(data_output, data_label)
    loss = DiceLoss(data_output, data_label)

    DSCs.append(DSC)
    losses.append(loss)
    print(DSC)
    print(loss)
    
print(np.mean(DSCs))
print(np.mean(losses))

#%%
#Ensemble detailed DSC
DSCs = []
losses = []
for i in [119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 130]:
    print('10000' + str(i) + '.mat')
    data_label = scipy.io.loadmat('/mnt/intern/code/dataset/test/label/10000' + str(i) + '.mat')
    data_label = data_label['label']
    data_label = data_label[128:384, 128:384, 0:250] # (256, 256, 250)
    data_output = np.squeeze(np.load('/mnt/intern/code/ensemble_result/10000' + str(i) + '.npy')) # (256, 256, 250)
    data_output = data_output[:, :, 0:250]

    for i in range(data_label.shape[2]):
    
        output, _ = connected_component_labeling(data_output[:, :, i])
        label, _ = connected_component_labeling(data_label[:, :, i])
        
        Detailed_output = largest_labels(output) # (258, 258) zero padding
        Detailed_label = largest_labels(label)  # (258, 258) zero padding
        
        Detailed_output = Detailed_output[1:-1, 1:-1] # (256, 256)
        Detailed_label = Detailed_label[1:-1, 1:-1] # (256, 256)
        
        Detailed_output = threshold(Detailed_output)
        Detailed_label = threshold(Detailed_label)
        
        data_label[:, :, i] = Detailed_label
        data_output[:, :, i] = Detailed_output

    data_label = np.expand_dims(data_label, axis=0)
    data_output =  np.expand_dims(data_output, axis=0)
    data_label = torch.from_numpy(data_label)
    data_output = torch.from_numpy(data_output)
    DSC = Dice_coef(data_output, data_label)
    loss = DiceLoss(data_output, data_label)

    DSCs.append(DSC)
    losses.append(loss)
    print(DSC)
    print(loss)
    
print(np.mean(DSCs))
print(np.mean(losses))
