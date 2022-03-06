# CT_Segmentation

[Batach VS Instance Normalization](https://www.baeldung.com/cs/instance-vs-batch-normalization)

[Mask R-CNN Object Detection](https://arxiv.org/pdf/1703.06870.pdf)

[Attention-enabled 3D boosted convolutional neural
networks for semantic CT segmentation using
deep supervision](https://iopscience.iop.org/article/10.1088/1361-6560/ab2818/pdf)

[UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/pdf/1807.10165.pdf)

[Cross Entropy meaning](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)

[Hausdorff Distance meaning](https://structseg2019.grand-challenge.org/Evaluation/)

[Hausdorff Distance for Iris Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4450956)

[ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

The disadvantage of using Hausdorff distance is that the degree of mismatch will be large when even a part of an object is missing due to occlusion, or when there are outliers. This is undesirable in object matching applications. In order to have a match between closely resembling portions of objects, partial Hausdorff distance Hp(A, B) has been proposed in [13] which is a modification of the conventional Hausdorff distance. This is done by taking Kth-ranked maximum value, instead of taking the overall maximum in the directed Hausdorff distance

[R-CNN](https://arxiv.org/pdf/1311.2524.pdf)

R-CNN decomposes the overall detection problem into two subproblems: to first utilize low-level cues such as color and superpixel consistency for potential object proposals in a category-agnostic fashion, and to then use CNN classifiers to identify object categories at those locations. Such a two stage approach leverages the accuracy of bounding box segmentation with low-level cues, as well as the highly powerful classification power of state-of-the-art CNNs.

[Going deeper with convolutions, GoogleNet, Inception](https://arxiv.org/pdf/1409.4842.pdf)

Inception architecture was especially useful in the context of localization and object detection as the base network for [6] and [5]. Interestingly, while most of the original architectural choices have been questioned and tested thoroughly, they turned out to be at least locally optimal.

#### Object detection Key

Note that assuming translation invariance means that our network will be built from convolutional building blocks. All we need is to find the optimal local construction and to repeat it spatially
