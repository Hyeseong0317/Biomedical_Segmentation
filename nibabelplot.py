#%%
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
#%%
img = nib.load('/mnt/intern/nnUNet/Aorta_output/AORTA_119.nii.gz')

data = img.get_data()

print(np.unique(data))
print(data.shape)
plt.set_cmap('gray')
for i in range(3):
    
    plt.subplot(5, 5,i + 1)
    plt.imshow(data[:,:,50 +50*i])
    plt.gcf().set_size_inches(10, 10)
    
plt.show()

#%%
# nifti -> mat save
import scipy.io
mdic_data = {'data' : data}
scipy.io.savemat('/mnt/intern/nnUNet/Aorta_output/10000119.mat', mdic_data)
