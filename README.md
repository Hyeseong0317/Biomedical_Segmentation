# CT_Segmentation

[SimpleITK, DICOM 다루기](http://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html)

[Batach VS Instance Normalization](https://www.baeldung.com/cs/instance-vs-batch-normalization)

[Mask R-CNN Object Detection](https://arxiv.org/pdf/1703.06870.pdf)

[Attention-enabled 3D boosted convolutional neural networks for semantic CT segmentation using deep supervision](https://iopscience.iop.org/article/10.1088/1361-6560/ab2818/pdf)

[Training Deeper Convolutional Networks with Deep Supervision](https://arxiv.org/pdf/1505.02496.pdf)

[Deeply-Supervised Nets](https://arxiv.org/pdf/1409.5185.pdf)

[UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/pdf/1807.10165.pdf)

[Cross Entropy meaning](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)

[Hausdorff Distance meaning](https://structseg2019.grand-challenge.org/Evaluation/)

[Hausdorff Distance for Iris Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4450956)

[ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

The disadvantage of using Hausdorff distance is that the degree of mismatch will be large when even a part of an object is missing due to occlusion, or when there are outliers. This is undesirable in object matching applications. In order to have a match between closely resembling portions of objects, partial Hausdorff distance Hp(A, B) has been proposed in [13] which is a modification of the conventional Hausdorff distance. This is done by taking Kth-ranked maximum value, instead of taking the overall maximum in the directed Hausdorff distance

[R-CNN](https://arxiv.org/pdf/1311.2524.pdf)

R-CNN decomposes the overall detection problem into two subproblems: to first utilize low-level cues such as color and superpixel consistency for potential object proposals in a category-agnostic fashion, and to then use CNN classifiers to identify object categories at those locations. Such a two stage approach leverages the accuracy of bounding box segmentation with low-level cues, as well as the highly powerful classification power of state-of-the-art CNNs.

In order to maintain high spatial resolution, these CNNs typically only have two convolutional and pooling layers.

[Going deeper with convolutions, GoogLeNet, Inception](https://arxiv.org/pdf/1409.4842.pdf)

Inception architecture was especially useful in the context of localization and object detection as the base network for [6] and [5]. Interestingly, while most of the original architectural choices have been questioned and tested thoroughly, they turned out to be at least locally optimal.

#### Object detection Key

Note that assuming translation invariance means that our network will be built from convolutional building blocks. All we need is to find the optimal local construction and to repeat it spatially

[nnUnet Biomedical Segmentation FrameWork 2D, 3D](https://github.com/MIC-DKFZ/nnUNet)

[nnUnet paper](https://arxiv.org/pdf/1809.10486.pdf)

[nnU-Net for Brain Tumor Segmentation](https://arxiv.org/pdf/2011.00848.pdf)

#### BatchNorm + DataAugmentation --> effectively close the domain gap to other scanners.

Batch normalization In our participation in the M&Ms challenge 7 we noticed that more aggressive data augmentation could be used to effectively close the domain gap to other scanners, but only when used in conjunction with batch normalization (instead of instance normalization). 
#### In BraTS, Dice scores for the test cases are often lower than the reported values on the training and validation dataset, which makes us believe that there may be a domain gap between the test set and the training and validation sets. This suggests that pursuing a this strategy for BraTS as well may be beneficial.

#### 보통 minibatch로 dice loss를 계산하나 이는 모델이 잘못예측한 voxel에 대해서는 크게 gradient를 업데이트하는 이점이 있지만, 애초에 주어진 voxel 자체가 에러가 있는 데이터였다면 역으로 모델이 크게 gradient를 최적점이 아닌 방향으로 업데이트 할 수 있다. 따라서 모든 샘플마다 minibatch를 독립적으로 계산한 뒤 배치에 대한 평균손실을 계산하는 것보다는 배치의 모든 샘플에 대해 dice loss를 계산한다.(이를 batch Dice라 정의함.) 이는 annotaated voxels이 거의 없는 샘플을 같은 배치 안의 다른 샘플들로 가려 정규화를 할 수 있어 우리가 원하는 방향으로 모델이 학습하는데 유리하다.
#### Batch dice The standard implementation of the Dice loss computes the loss for every sample in the minibatch independently and then averages the loss over the batch (we call this the sample Dice). Small errors in samples with few annotated voxels can cause large gradients and dominate the parameter updates during training. If these errors are caused by imperfect predictions of the model, these large gradients are desired to push the model towards better predictions. However, if the models predictions are accurate and the reference segmentation is imperfect, these large gradients will be counterproductive during training. We therefore implement a different Dice loss computation: instead of treating the samples in a minibatch independently, we compute the dice loss over all samples in the batch (pretending they are just a single large sample, we call this the batch Dice). This effectively regularizes the Dice loss because samples with previously only few annotated voxels are now overshadowed by other samples in the same batch.

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

[Multi-GPU, VRAM, BottleNeck, NVLink](https://89douner.tistory.com/157)

[Deep supervision](https://arxiv.org/pdf/1801.03399.pdf)

As such, coarse-grained class labels used as intermediate concepts are able to improve fine-grained recognition performance, which further validates our deep supervision strategy.

The more concepts applied, the better chance that the generalization is improved. In conclusion, deep supervision with intermediate concepts regularizes the network training by decreasing the number of incorrect solutions that generalize poorly to the test set.

[The Liver Tumor Segmentation Benchmark (LiTS)](https://arxiv.org/pdf/1901.04056.pdf)

##### Cascaded approach
The common cascaded approaches allowed each network to focus and learn the relevantfeatures for its specific task improving overall performance of both liver and tumor segmentation. In addition, network architecture design can be adapted to the specific network task resulting in a more effective and efficient training process and outcome. Also, earlier networks in the cascaded pipeline might help to standardize the intensity distribution for later networks which has been shown to improve overall performance.


##### Higher Dimensionality
However, Chlebus et al. used a small 3D network as the final network in a cascaded infrastructure to fuse the segmentation results of various previous
networks into a final segmentation mask, while Li implemented a shallow 3D network to further refine the preliminary results of a first network. These good results not only prove the feasibility and usefulness of a 3D network architecture despite the currently limited memory availability when used as a supplementary role but also show their huge potential for future performance increase by being able to capture the whole volume context.

[A Review of Deep-Learning-Based Medical Image Segmentation Methods](https://www.mdpi.com/2071-1050/13/3/1224/htm)

[SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/pdf/1511.00561.pdf)

SegNet on the other hand is more efficient since it only stores the max-pooling indices of the feature maps and uses them in its decoder network to achieve good performance.

[PSPNet: Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105.pdf)

We have proposed an effective pyramid scene parsing network for complex scene understanding. The global pyramid pooling feature provides additional contextual information.

[Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation](https://arxiv.org/pdf/1506.04924.pdf)

Bridging layers play a critical role to reduce output space of segmentation, and facilitate to learn segmentation network using a handful number of segmentation annotations.

[Learning Deconvolution Network for Semantic Segmentation](https://arxiv.org/pdf/1505.04366.pdf)

The proposed deconvolution network is suitable to generate dense and precise object segmentation masks since coarse-to-fine structures of an object is reconstructed progressively through a sequence of deconvolution operations.

[2.5D: Deep learning of the sectional appearances of 3D CT images for anatomical structure segmentation based on an FCN voting method.]()

[Random 2.5D U-net for Fully 3D Segmentation](https://arxiv.org/pdf/1910.10398.pdf)

Although the construction of 3D segmentation masks with the help of 3D U-net [2, 4, 8] delivers very satisfying results for vessel segmentation, generation of artefacts and the restrictions to model’s complexity due to memory issues are hardly sustainable. Therefore, we proposed the random 2.5D Unet structure, that is able to conduct volumetric segmentation of very big biomedical 3D scans so we can train a network without any concern about memory space and input size. For the targeted application, the random 2.5D U-net even outperformed the standard slice-by-slice and 3D convolution approaches and showed more consistent accuracy for test application.

[Projection-Based 2.5D U-net Architecture for Fast Volumetric Segmentation](https://arxiv.org/pdf/1902.00347.pdf)

Vessel 시각화 tool ITK-SNAP

[3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/pdf/1606.06650.pdf)

##### The implementation performs on-the-fly elastic deformations for efficient data augmentation during training
##### Data augmentation is done on-the-fly, which results in as many different images as training iterations.

We also introduce batch normalization (“BN”) before each ReLU. In [4], each batch is normalized during training with its mean and standard deviation and global statistics are updated using these values. This is followed by a layer to learn scale and bias explicitly. At test time, normalization is done via these computed global statistics and the learned scale and bias. However, we have a batch size of one and few samples. In such applications, using the current statistics also at test time works the best.
--> batch size가 작으면 그냥 test에서도 test dataset이 갖는 statistic을 사용하는게 제일 좋다고 한다. 다시말해 batch size가 작으면 test시에 InstanceNorm을 쓴다는 의미. (BatchNorm은 training에서 학습된 mean과 bias를 평균내서 test시에 사용한다.)

The important part of the architecture, which allows us to train on sparse annotations, is the weighted softmax loss function. Setting the weights of unlabeled pixels to zero makes it possible to learn from only the labelled ones and, hence, to generalize to the whole volume.

#### In this experiment BN also improves the result, except for the third setting, where it was counterproductive. We think that the large differences in the data sets are responsible for this effect. The typical use case for the fully-automated segmentation will work on much larger sample sizes, where the same number of sparse labels could be easily distributed over much more data sets to obtain a more representative training data set. (BatchNorm3d가 오히려 성능을 떨어뜨리는 이유)

[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)

[Image Segmentation with Cascaded Hierarchical Models and Logistic Disjunctive Normal Networks](https://ieeexplore.ieee.org/document/6751380)

[Hypercolumns for Object Segmentation and Fine-grained Localization](https://arxiv.org/pdf/1411.5752.pdf)

[Hough-CNN: Deep Learning for Segmentation of Deep Brain Regions in MRI and Ultrasound](https://arxiv.org/pdf/1601.07014.pdf)

[Deep MRI brain extraction: A 3D convolutional neural network for skull stripping](https://pubmed.ncbi.nlm.nih.gov/26808333/)

[3D Convolutional Neural Networks for Brain Tumor Segmentation: a comparison of multi-resolution architectures](https://arxiv.org/pdf/1705.08236.pdf)

This distinction is rather challenging as borders are often fuzzy, and also because tumors vary across patients in size, location and extent. Several imaging modalities can be used to solve this task, individually or combined, including T1, T1-contrasted, T2 and FLAIR, each one providing different biological information.

[Semantic segmentation using adversarial networks](https://arxiv.org/pdf/1611.08408.pdf)

[Automatic liver segmentation using an adversarial image-to-image network](https://arxiv.org/pdf/1707.08037.pdf)

[Edge-aware Fully Convolutional Network with CRF-RNN Layer for Hippocampus Segmentation](https://ieeexplore.ieee.org/abstract/document/8785801)
