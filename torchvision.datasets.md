```import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.DatasetFolder(root=dataroot,
                           transform=transforms.Compose([
                            #    transforms.Resize(image_size),
                            #    transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
```

---> [torchvision.datasets --> DatasetFolder, ImageFolder, VisionDataset](https://pytorch.org/vision/stable/datasets.html)

#### ImageFolder를 사용하면 .jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif,.tiff,.webp를 불러올때 편하다. 단 데이터루트안에 서브폴더를 만들어줘야한다. ImageFolder함수자체가 classification을 목적으로 한 함수이므로 cat, dog, plane과 같이 폴더를 루트안에 추가로 생성해야 이미지를 불러올 수 있다.
[[Pytorch] Dataloader 다양하게 사용하기 (custom loader)](https://honeyjamtech.tistory.com/38)

#### DatasetFolder, Custom loader로 다른 타입의 데이터를 다룰 수 있다.
```
def custom_loader(path): 
  ret = np.load(path) 
  return ret 
data_path = '/mnt/intern/3D_vessel/demo_code/data/train' 
dataset = dset.DatasetFolder(root=data_path, loader=custom_loader, extensions='.npy') 
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
```

<img src="https://github.com/Hyeseong0317/CT_Segmentation/blob/main/images/torchvisiondatasets.PNG" width="60%">


