# Biomedical_Segmentation

### Convolution, Stride, Upsampling, Resolution Idea

[Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)

[Multi-scale context aggregation by dilated convolutions](https://arxiv.org/pdf/1511.07122.pdf)

[Deep generative image models using a laplacian pyramid of adversarial networks](https://arxiv.org/pdf/1506.05751.pdf)

[Image superresolution using deep convolutional networks, 2015](https://arxiv.org/pdf/1501.00092.pdf)

[Texture synthesis and the controlled generation of natural stimuli using convolutional neural networks. In Advances in Neural Information Processing Systems (NIPS)](https://arxiv.org/pdf/1505.07376.pdf)

[Understanding deep image representations by inverting them. In IEEE Conf. on Comp. Vision and Pattern Recognition (CVPR), 2015](https://arxiv.org/pdf/1412.0035.pdf)

[L. Xu, J. S. Ren, C. Liu, and J. Jia. Deep convolutional neural network for image deconvolution. In Advances in Neural Information Processing Systems (NIPS) 27, pages 1790–1798. 2014.](https://proceedings.neurips.cc/paper/2014/file/1c1d4df596d01da60385f0bb17a4a9e0-Paper.pdf)

[M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. In Proc. Europ. Conf. Comp. Vision (ECCV), pages 818–833, 2014.](https://arxiv.org/pdf/1311.2901.pdf)

## 학회

NIPS(Neural Information Processing Systems) 신경정보처리시스템학회

MICCAI(Medical Image Computing Computed Assisted Interventions) 국제의료영상컴퓨팅 및 인터벤션 학술대회

IEEE(Institute of Electrical and Electronics Engineers) 국제전기전자기술협회

MIC(Medical Imaging Conference) 

ISBI(IEEE Symposium on Biomedical Imaging)

ISMRM(International Society for Magnetic Resonance Imaging) 국제자기공명의과학회

NSS(Nuclear Science Symposium)

### 논문 및 아이디어

Filter수를 늘리면 edge와 같이 특성들을 보는 convolution filter가 늘어난다. <-> Convolution block OR pooling수를 늘리면 Receptive field가 커진다.
Batch Size를 늘리면 convergence가 빠르고 generalization이 된다.

[SimpleITK, DICOM 다루기](http://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html)

[Batach VS Instance Normalization](https://www.baeldung.com/cs/instance-vs-batch-normalization)

[Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/pdf/1607.08022.pdf)

### Image Generation에서는 Instance Normalization을 쓰는게 효과적으로 알려져있다.
This approach to batch normalization, when the batch size is set to 1, has been termed “instance normalization” and has been demonstrated to be effective at image generation tasks.

[Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)

#### Leaky ReLU와 ReLU의 큰 차이는 없다. Sigmoidal보다는 Rectified activation function을 쓴다. ReLU를 사용함으로써 Sigmoid가 -1, 1부근에서 gradient가 0으로 수렴하여 hidden layer의 가중치가 업데이트되지 않는 gradient vanishing 문제를 해결할 수 있다.
Both the leaky and standard ReL networks perform similarly, suggesting the leaky rectifiers’ non-zero gradient does not substantially impact training optimization. During training we observed leaky rectifier DNNs converge slightly faster, which is perhaps due to the difference in gradient among the two rectifiers. In addition to performing better overall, rectifier DNNs benefit more from depth as compared with sigmoidal DNNs. Each time we add a hidden layer, rectifier DNNs show a greater absolute reduction in WER than sigmoidal DNNs. We believe this effect results from the lack of vanishing gradients in rectifier networks. 

[Cross Entropy meaning](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)

[Hausdorff Distance meaning](https://structseg2019.grand-challenge.org/Evaluation/)

[Hausdorff Distance for Iris Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4450956)

[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

[Densely connected convolutional networks](https://arxiv.org/pdf/1608.06993.pdf)

[Automatic brain tumor segmentation using cascaded anisotropic convolutional neural networks](https://arxiv.org/pdf/1709.00382.pdf)

[Ensembles of Multiple Models and Architectures for Robust Brain Tumour Segmentation](https://arxiv.org/pdf/1711.01468.pdf)

Regularization can be explicit, such as weight decay that prevents networks from learning rare noisy patterns, or implicit, such as the local connectivity of CNN kernels, which however does not allow the model to learn patterns larger than the its receptive field. Architectural and configuration choices thus introduce bias, altering the behaviour of a network. One route to address the bias/variance dilemma is ensembling. By combining multiple models, ensembling seeks to create a higher performing model with low variance. The most popular combination rule is averaging, which is not sensitive to inconsistent errors of the singletons.

Lack of consistent failures can be interpreted as statistical independency. Thus methods for de-correlating the instances have been developed. The most popular is bagging [17], commonly used for random forests. It uses bootstrap sampling to learn less correlated instances from different subsets of the data. [Bagging predictors](https://link.springer.com/article/10.1007/BF00058655)

#### Ensemble benefit -> Intuitively, only inconsistent errors can be averaged out.

#### Model performance 고려사항 4가지
1.large receptive fields(=improved localisation capbilities but less sensitive to fine texture than models emphasizing local information)

2.handling class imbalance(class-weighted sampling or class-weighted corss entropy)

3.loss function(DSC, IoU, cross entropy)

4.hyper-parameters(learning rate)

[Unsupervised domain adaptation in brain lesion segmentation with adversarial networks](https://arxiv.org/pdf/1612.08894.pdf)

[Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation](https://www.sciencedirect.com/science/article/pii/S1361841516301839)

[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

[Multi-GPU, VRAM, BottleNeck, NVLink](https://89douner.tistory.com/157)

[Mask R-CNN Object Detection](https://arxiv.org/pdf/1703.06870.pdf)

[Fully Convolutional Networks for Semantic Segmentation](https://www.cvfoundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

[The Importance of Skip Connections in Biomedical Image Segmentation](https://arxiv.org/pdf/1608.04117.pdf)

[Attention-enabled 3D boosted convolutional neural networks for semantic CT segmentation using deep supervision](https://iopscience.iop.org/article/10.1088/1361-6560/ab2818/pdf)

[Training Deeper Convolutional Networks with Deep Supervision](https://arxiv.org/pdf/1505.02496.pdf)

[Deeply-Supervised Nets](https://arxiv.org/pdf/1409.5185.pdf)

[UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/pdf/1807.10165.pdf)

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

[Automated Design of Deep Learning Methods for Biomedical Image Segmentation]()

[nnU-Net for Brain Tumor Segmentation](https://arxiv.org/pdf/2011.00848.pdf)

#### BatchNorm + DataAugmentation --> effectively close the domain gap to other scanners.

Batch normalization In our participation in the M&Ms challenge 7 we noticed that more aggressive data augmentation could be used to effectively close the domain gap to other scanners, but only when used in conjunction with batch normalization (instead of instance normalization). 
#### In BraTS, Dice scores for the test cases are often lower than the reported values on the training and validation dataset, which makes us believe that there may be a domain gap between the test set and the training and validation sets. This suggests that pursuing a this strategy for BraTS as well may be beneficial.

nnU-Net의 batchsize를 default 2에서 5로 늘렸으나 성능을 떨어뜨렸다.

#### 보통 minibatch로 dice loss를 계산하나 이는 모델이 잘못 예측한 voxel에 대해서는 크게 gradient를 업데이트하는 이점이 있지만, 애초에 주어진 voxel 자체가 에러가 있는 데이터였다면 역으로 모델이 크게 gradient를 최적점이 아닌 방향으로 업데이트 할 수 있다. 따라서 모든 샘플마다 minibatch를 독립적으로 계산한 뒤 배치에 대한 평균손실을 계산하는 것보다는 배치의 모든 샘플에 대해 dice loss를 계산한다.(이를 batch Dice라 정의함.) 이는 annotaated voxels이 거의 없는 샘플을 같은 배치 안의 다른 샘플들로 가려 정규화를 할 수 있어 우리가 원하는 방향으로 모델이 학습하는데 유리하다.
#### Batch dice The standard implementation of the Dice loss computes the loss for every sample in the minibatch independently and then averages the loss over the batch (we call this the sample Dice). Small errors in samples with few annotated voxels can cause large gradients and dominate the parameter updates during training. If these errors are caused by imperfect predictions of the model, these large gradients are desired to push the model towards better predictions. However, if the models predictions are accurate and the reference segmentation is imperfect, these large gradients will be counterproductive during training. We therefore implement a different Dice loss computation: instead of treating the samples in a minibatch independently, we compute the dice loss over all samples in the batch (pretending they are just a single large sample, we call this the batch Dice). This effectively regularizes the Dice loss because samples with previously only few annotated voxels are now overshadowed by other samples in the same batch.

#### nnU-Net ensemble 기본 원리
Note that ensembling was implemented by first predicting the test cases individually with each configuration, followed by averaging the sigmoid outputs to
obtain the final prediction.

#### nnU-Net postprocessing 주의점, 작은 lesion이 제거되버릴 수 있다.
The enhancing tumor score of 0 in the absence of predicted enhancing tumor voxels indicates either that our model missed a small enhancing tumor lesion or that it was removed as a result of our postprocessing. An inspection of the non-postprocessed segmentation mask reveals that the enhancing tumor lesion was indeed segmented by the model and must have been removed during postprocessing.

#### BraTS platform에 가장 유리한 performance 측정방식을 활용하면 1등할 수 있다.

#### small lesion을 제거해서 false positive의 dice score를 1로 하는 것은 챌린지에서 우승하는데 유리하지만, 임상에서는 small lesion을 찾는게 중요하다. 따라서 postprocessing으로 small lesion을 제거해버리지 않도록 하도록 고려해야한다.
The enhancing tumor class is arguably the most difficult to segment in this dataset. What makes this class particularly challenging is the way its evaluation is handled when the reference segmentation of an image does not contain this class. The BraTS evaluation scheme favors the removal of small enhancing tumor lesions and thus encourages such postprocessing. In a clinical scenario where the accurate detection of small enhancing tumors could be critical, this property is not necessarily desired and we recommend to omit the postprocessing presented in this manuscript.

#### leaderboard의 ranking strategy에 따라 순위가 달라진다.
 Ranking schemes can be differentiated in ’aggregate then rank’ and ’rank then average’ approaches. In the former, some aggregated metric (for example the average) is computed and then used to rank the participants. In the latter, the participants are ranked on each individual training case and then their ranks are accumulated across all cases. Different algorithm characteristics may be desired depending on the ranking scheme that is used to evaluate them. For example, in an ’aggregate then rank’ scheme, median aggregation (as opposed to the mean) would be more forgiving to algorithms that produce badly predicted outliers. 
#### BraTS uses a ’rank then aggregate’ approach, most likely because it is well suited to combine different types of segmentation metrics (such as HD95 and Dice).

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

[Image Segmentation with Cascaded Hierarchical Models and Logistic Disjunctive Normal Networks](https://ieeexplore.ieee.org/document/6751380)

[Hypercolumns for Object Segmentation and Fine-grained Localization](https://arxiv.org/pdf/1411.5752.pdf)

[Hough-CNN: Deep Learning for Segmentation of Deep Brain Regions in MRI and Ultrasound](https://arxiv.org/pdf/1601.07014.pdf)

[Deep MRI brain extraction: A 3D convolutional neural network for skull stripping](https://pubmed.ncbi.nlm.nih.gov/26808333/)

[3D Convolutional Neural Networks for Brain Tumor Segmentation: a comparison of multi-resolution architectures](https://arxiv.org/pdf/1705.08236.pdf)

This distinction is rather challenging as borders are often fuzzy, and also because tumors vary across patients in size, location and extent. Several imaging modalities can be used to solve this task, individually or combined, including T1, T1-contrasted, T2 and FLAIR, each one providing different biological information.

[Semantic segmentation using adversarial networks](https://arxiv.org/pdf/1611.08408.pdf)

[Automatic liver segmentation using an adversarial image-to-image network](https://arxiv.org/pdf/1707.08037.pdf)

[Edge-aware Fully Convolutional Network with CRF-RNN Layer for Hippocampus Segmentation](https://ieeexplore.ieee.org/abstract/document/8785801)

## 생성모델과 잠재공간(임베딩공간) z (Generative Model, Latent Space z)

### 고차원 --> 저차원의 잠재공간 z --> 고차원

### Auto Encoder

[초짜 대학원생의 입장에서 이해하는 Auto-Encoding Variational Bayes (VAE) (1)](https://jaejunyoo.blogspot.com/2017/04/auto-encoding-variational-bayes-vae-1.html)

[Auto-encoding variational bayes](https://arxiv.org/pdf/1312.6114.pdf)

<img src="https://github.com/Hyeseong0317/CT_Segmentation/blob/main/images/Taxonomyofdeepgenerativemodels.PNG" width="60%">

<img src="https://github.com/Hyeseong0317/CT_Segmentation/blob/main/images/autoencoder.PNG" width="60%">

### GAN

### 수학적으로 생성 모델의 목적은 실제 데이터 분포와 근사한 것이라고 말할 수 있다.

GAN: https://arxiv.org/abs/1406.2661
DCGAN: https://arxiv.org/abs/1511.06434
cGAN: https://arxiv.org/abs/1611.07004

WGAN: https://arxiv.org/abs/1701.07875
EBGAN: https://arxiv.org/abs/1609.03126
BEGAN: https://arxiv.org/abs/1703.10717

CycleGAN: https://arxiv.org/abs/1703.10593
DiscoGAN: https://arxiv.org/abs/1703.05192
StarGAN: https://arxiv.org/abs/1711.09020

SRGAN: https://arxiv.org/abs/1609.04802
SEGAN: https://arxiv.org/abs/1703.09452
StyleGAN: https://arxiv.org/pdf/1812.04948.pdf

#### StyleGAN 
#### If the network tried to control, e.g., pose using the noise, that would lead to spatially inconsistent decisions that would then be penalized by the discriminator. Thus the network learns to use the global and local channels appropriately, without explicit guidance.

#### Latent Space -> 직관적으로 less curved latent space가 highly curved latent space보다 지각적으로 더 부드러운 변환을 한다.
As noted by Laine [37], interpolation of latent-space vectors may yield surprisingly non-linear changes in the image. For example, features that are absent in either endpoint may appear in the middle of a linear interpolation path. This is a sign that the latent space is entangled and the factors of variation are not properly separated. To quantify this effect, we can measure how drastic changes the image undergoes as we perform interpolation in the latent space. Intuitively, a less curved latent space should result in perceptually smoother transition than a highly curved latent space.

#### Linear Seperability -> Latent Space가 충분히 disentagled되어있으면 latent space를 linear hyperplane을 통해 2개의 구별되는 subsets으로 나눌 수 있다.
If a latent space is sufficiently disentangled, it should be possible to find direction vectors that consistently correspond to individual factors of variation. We propose another metric that quantifies this effect by measuring how well the latent-space points can be separated into two distinct sets via a linear hyperplane, so that each set corresponds to a specific binary attribute of the image.

#### Conclusion -> high-level attributes의 분리, 확률효과, intermediate latent space의 선형성은 GAN synthesis의 control에 대한 이해를 돕는다.
We further believe that our investigations to the separation of high-level attributes and stochastic effects, as well as the linearity of the intermediate latent space will prove fruitful in improving the understanding and controllability of GAN synthesis.

[Image style transfer using convolutional neural networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

### entangle
서로 얽혀 있는 상태여서 특징 구분이 어려운 상태. 즉, 각 특징들이 서로 얽혀있어서 구분이 안됨
### disentangle
각 style들이 잘 구분 되어있는 상태여서 어느 방향으로 가면 A라는 특징이 변하고 B라는 특징이 변하게 되어서 특징들이 잘 분리가 되어있다는 의미.
선형적으로 변수를 변경했을 때 어떤 결과물의 feature인지 예측할 수 있는 상태.

[GAN 포스트](https://minsuksung-ai.tistory.com/12)

[GAN 포스트2](https://dreamgonfly.github.io/blog/gan-explained/)

[Patch-Based Image Inpainting with Generative Adversarial Networks](https://arxiv.org/pdf/1803.07422.pdf)

#### 유클리디안 거리는 이미지의 정보를 측정하고 비교하는데 쓰이지만 보통 possible intensity value들의 평균에 수렴하므로 흐릿한 이미지를 출력하는 경향이 있다. GAN은 objective function을 새로 정의함으로써 이 문제를 해결했다.
Primitive objective functions like Euclidean Distance assist in measuring and comparing information on the general structure of the images, however, they tend to converge to the mean of possible intensity values that cause blurry outputs. In order to solve this challenging problem, Goodfellow et al. proposed Generative Adversarial Networks (GAN) [7], which is a synthesis model trained based on a comparison of real images with generated outputs.

#### Generator network는 corrupted 이미지에 존재하는 임의의 큰 구멍을 채워 이미지를 복원한다.
We introduce a generative CNN model and a training procedure for the arbitrary and large hole filling problem. The generator network takes the corrupted image and tries to reconstruct the repaired image.

[Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis](https://arxiv.org/pdf/1601.04589.pdf)[20]

Then, the style transfer method of [20] is used, which forces features of the small patches from the masked area to be close to those of the undamaged region to improve texture details.

### Markov Random Fields, dCNNs는 지역적으로 상관된 정보와 입력의 위치가 변해도 출력이 변하지 않는 translational invariance에 의존한다.
Our paper augments their framework by replacing the bag-of-feature-like statistics of Gram-matrix-matching by an MRF regularizer that maintains local patterns of the “style” exemplar: MRFs and dCNNs are a canonical combination — both models crucially rely on the assumption of locally correlated information and translational invariance.

### Intra-class variation VS Inter-class variation
Intra- 클래스 내부의 분산이 어떤가

Inter- 클래스간 분산이 어떠냐

[Texture synthesis and the controlled generation of natural stimuli using convolutional neural networks](https://arxiv.org/pdf/1505.07376.pdf)

[Context encoders:feature learning by inpainting](https://arxiv.org/pdf/1604.07379.pdf)

[Semantic Image Inpainting with Deep Generative Models](https://arxiv.org/pdf/1607.07539.pdf)

[Generative Face Completion](https://arxiv.org/pdf/1704.05838.pdf)[21]

In [21], LGAN pushes the generative network to produce independent textures that are incompatible with the whole image semantics.

[Multimodal Unsupervised Image-to-Image Translation](https://arxiv.org/pdf/1804.04732.pdf)

multimodal이란 다양한 형태의 데이터를 입력 데이터로 사용한다는 의미이며 예를 들어 이미지와 텍스트 데이터를 동시에 사용한다는 것이다.

Unsupervised img2img translation의 목적은 이미지를 보지 않고도 target 도메인에서 그에 상응하는 이미지들의 조건부 분포를 구하는 것이다 조건부 분포는 본질적으로 multimodal인데, 현재 존재하는 기법들은 너무 가정을 단순화해서 결정론적으로 one-to-one mapping한다. (결정론적 알고리즘은 예측한 그대로 동작하는 알고리즘이다. 어떤 특정한 입력이 들어오면 언제나 똑같은 과정을 거쳐서 언제나 똑같은 결과를 내놓는다.) 결론적으로 주어진 source domain image로부터 다양한 outputs을 내는 것에 실패한다.

#### Unsupervised image-to-image translation is an important and challenging problem in computer vision. Given an image in the source domain, the goal is to learn the conditional distribution of corresponding images in the target domain, without seeing any examples of corresponding image pairs. While this conditional distribution is inherently multimodal, existing approaches make an overly simplified assumption, modeling it as a deterministic one-to-one mapping. As a result, they fail to generate diverse outputs from a given source domain image.

#### 이미지 표현을 도메인 불변인 content code와 도메인에 특별한 특성을 갖는 style code로 분해가 가능하다 가정한다. 이미지를 다른 도메인으로 전이시키기위해 content code와 타켓 도메인의 style space에서 샘플된 random style code를 재결합한다.
#### We assume that the image representation can be decomposed into a content code that is domain-invariant, and a style code that captures domain-specific properties. To translate an image to another domain, we recombine its content code with a random style code sampled from the style space of the target domain.

#### 즉 이미지의 잠재영역(latent space)을 content space와 style space로 분해한다 가정한다. 이미지가 다른 도메인들 사이에서 공유하는 content space가 있고, style space는 공유되지 않는다 가정한다. 입력 이미지를 타겟 도메인으로 바꿀때 그 이미지의 content code와 타겟 도메인의 random style code를 재결합한다. content code는 전이되는 동안 보존되지만 style code는 입력 이미지에 포함되지 않았으며 variations를 갖는다. 다른 style code를 샘플링함으로써 다양하고 multimodal인 outputs를 출력할 수 있다.
We first assume that the latent space of images can be decomposed into a content space and a style space. We further assume that images in different domains share a common content space but not the style space. To translate an image to the target domain, we recombine its content code with a random style code in the target style space (Fig. 1 (b)). The content code encodes the information that should be preserved during translation, while the style code represents remaining variations that are not contained in the input image. By sampling different style codes, our model is able to produce diverse and multimodal outputs.

<img src="https://github.com/Hyeseong0317/CT_Segmentation/blob/main/images/latentspace.PNG" width="60%">

### 현재 존재하는 image-to-image methods의 한계는 translated outputs의 다양성 부족이다.
A significant limitation of most existing image-to-image translation methods is the lack of diversity in the translated outputs. To tackle this problem, some works propose to simultaneously generate multiple outputs given the same input and encourage them to be different 

[Unsupervised Image-to-Image Translation Networks](https://arxiv.org/pdf/1703.00848.pdf)

UNIT framework, which assumes a shared latent space such that corresponding images in two domains are mapped to the same latent code.

[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf)

#### GAN의 목적함수에 L2 distance를 섞어주면 좋은 효과가 있다. 구별자의 역할은 변함없고, 단 생성자는 구별자를 속이는 것뿐만 아니라 생성자가 만들어낸 output 가짜이미지를 L2 손실에서 ground truth에 가깝도록 학습한다.

Previous approaches to conditional GANs have found it beneficial to mix the GAN objective with a more traditional loss, such as L2 distance. The discriminator’s job remains unchanged, but the generator is tasked to not only fool the discriminator but also to be near the ground truth output in an L2 sense. We also explore this option, using L1 distance rather than L2 as L1 encourages less blurring:

<img src="https://github.com/Hyeseong0317/CT_Segmentation/blob/main/images/GANobjectiveL1.PNG" width="60%">

#### GAN은 random gaussian noise z가 없는 경우 x를 넣으면 y로 결정적으로 출력하므로(= 입력이 x이면 무조건 출력이 y로 나오므로) overfitting문제가 발생하며 함수로 표현하는 경우 어떠한 입력분포에 대해서라도 출력분포가 delta function으로 매핑되버린다.
Without z, the net could still learn a mapping from x to y, but would produce deterministic outputs, and therefore fail to match any distribution other than a delta function.

#### cGANs을 설계시 확률적인 output을 내도록 디자인하는 것이 중요하다. 그럼으로써 조건분포 전체 entropy를 잡아낼 수 있으며 이는 중요한 문제로 현재 남아있다. 모델의 출력이 샘플로 추출한 표본분포에 편향되어 학습하는 경우, 새로운 표본분포에 대해서는 성능이 떨어진다. 따라서 모델의 출력이 결정적으로 나오는 것이 아니라 모집단의 분포를 따라갈 수 있도록 확률적으로 설계해야만 한다.
Despite the dropout noise, we observe very minor stochasticity in the output of our nets. Designing conditional GANs that produce stochastic output, and thereby capture the full entropy of the conditional distributions they model, is an important question left open by the present work.

#### L2와 L1은 blurry한 image를 생성한다. Highfrequency는 잘 못잡는 반면에 low frequencies는 정확하게 잡아낸다. L1 또는 L2 loss를 사용하면 low frequencies를 잡는데 새로운 프레임워크를 필요로 하지 않는다. GAN의 구별자가 high-frequency structure에만 집중하도록 구성할 수 있다. high-frequencies를 잡는 것에 attention을 줌으로써 local image patches에서 high-frequencies 특성을 잡을 수 있다. 이를 PatchGAN을 통해 수행할 수 있다. (Loss 구성시 매우 좋은 아이디어)
It is well known that the L2 loss – and L1, see Figure 4 – produces blurry results on image generation problems [22]. Although these losses fail to encourage highfrequency crispness, in many cases they nonetheless accurately capture the low frequencies. For problems where this is the case, we do not need an entirely new framework to enforce correctness at the low frequencies. L1 will already do. This motivates restricting the GAN discriminator to only model high-frequency structure, relying on an L1 term to force low-frequency correctness (Eqn. 4). In order to model high-frequencies, it is sufficient to restrict our attention to the structure in local image patches.

#### 합성된 이미지의 퀄리티를 평가하는건 아직 어려운 문제로 남아있다.
Evaluating the quality of synthesized images is an open and difficult problem [36]. Traditional metrics such as perpixel mean-squared error do not assess joint statistics of the result, and therefore do not measure the very structure that structured losses aim to capture.

--> [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf) inception score 

--> [Generative Image Modeling using Style and Structure Adversarial Networks](https://arxiv.org/pdf/1603.05631.pdf) the object detection evaluation in [39]

--> [Colorful image colorization](https://arxiv.org/pdf/1603.08511.pdf) the “semantic interpretability” measure in [46]

#### Encoder Decoder + skip connection = U-Net

<img src="https://github.com/Hyeseong0317/CT_Segmentation/blob/main/images/Unet&encoderdecoder.PNG" width="60%">

--> Color histogram matching is a common problem in image processing [33], and PixelGANs may be a promising lightweight solution.

[The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/pdf/1604.01685.pdf)

[Context Encoders: Feature Learning by Inpainting](https://arxiv.org/pdf/1604.07379.pdf)

[Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/pdf/1512.09300.pdf)

[Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)

<img src="https://github.com/Hyeseong0317/CT_Segmentation/blob/main/images/gan수식.png" width="60%">

pdata(x) 원본 데이터의 distribution에서 한 개의 샘플 x를 샘플링 -> 코드상에서는 하나의 이미지를 뽑는다.

pz(z) 노이즈 distribution에서 하나의 노이즈 z를 샘프링 -> 코드상에서는 하나의 노이즈 이미지를 뽑다.

G는 V(D,G)의 값을 낮추려하고, D는 V(D,G)의 값을 높일려고 한다.

Generator에서는 InstanceNorm을 쓰는게 좋다. Instance normalization: The missing ingredient for fast stylization

Moreover, these samples are uncorrelated because the sampling process does not depend on Markov chain mixing. -> Another advantage of adversarial networks is that they can represent very sharp, even degenerate distributions, while methods based on Markov chains require that the distribution be somewhat blurry in order for the chains to be able to mix between modes. --> Markov chains와 다르게 GAN은 samples이 uncorrelated이므로 더 sharp한 이미지를 갖는다.

#### Semi-supervised learning: features from the discriminator or inference net could improve performance of classifiers when limited labeled data is available

[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf) DCGAN

DCGAN의 특징은 몇 가지로 요약할 수 있다. 먼저, 선형 레이어와 풀링 레이어(Pooling Layer)를 최대한 배제하고 합성곱(Convolution)과 ‘Transposed Convolution(Fractional-Strided Convolution)’으로 네트워크 구조를 만들었다. 풀링 레이어는 여러 딥러닝 모델에서 불필요한 매개변수의 수를 줄이고 중요한 특징만을 골라내는 역할을 하는 레이어지만 이미지의 위치 정보를 잃어버린다는 단점이 있다. 이미지를 생성하기 위해서는 위치 정보가 중요하기 때문에 DCGAN은 풀링 레이어를 배제했다. 선형 레이어 역시 마찬가지로 위치 정보를 잃어버리므로 모델의 깊은 레이어에서는 선형 레이어를 사용하지 않았다.

### Image Reconstruction

[Learning to generate chairs with convolutional neural networks](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf)

[Learning to Generate Chairs, Tables and Cars with Convolutional Networks](https://arxiv.org/pdf/1411.5928.pdf)

We found that adding a convolutional layer after each up-convolution significantly improves the quality of the generated images.
#### Data augmentation leads to worse reconstruction of fine details, but it is expected to lead to better generalization
The network easily deals with extreme color-related transformations, but has more problems representing large spatial changes, especially translations.
The performance of this nearest neighbor method is always worse than that of the network with knowledge transfer, suggesting that the network learns more than just linearly combining the known chairs, especially when many viewpoints are available in the target set. The results are shown in Figure 10. While the network trained only on chairs does not generalize to unseen elevation angles almost at all, the one trained with tables is able to generate unseen views of chairs very well. The only drawback is that the generated images do not always precisely correspond to the desired elevation, for example 0◦. and 10◦ for the second model in Figure 10.The network trained both on chairs and tables can fairly well predict views of tables from previously unseen elevation angles.

<img src="https://github.com/Hyeseong0317/CT_Segmentation/blob/main/images/chair.PNG" width="40%">

<img src="https://github.com/Hyeseong0317/CT_Segmentation/blob/main/images/table.PNG" width="40%">

Our hypothesis is that the network trained on both object classes is forced to not only model one kind of objects, but also the general 3D geometry. This helps generating reasonable views from new elevation angles. We hypothesize that modeling even more object classes with a single network would allow to learn a universal class-independent representation of 3D shapes.

#### Remarkably, the generative network can not only imagine previously unseen views of a given object, but also invent new objects by interpolating between given ones. To obtain such interpolations, we simply linearly change the input label vector from one class to another.

The inter-class difference is larger than the intra-class variance, hence to successfully interpolate between classes the network has to close this large gap between different classes.

#### By simple subtraction and addition in the feature space (FC2 features in this case) we can change an armchair into a chair with similar style, or a chair with a stick back into an identical chair with a solid back. We found that the exact layer where the arithmetic is performed does not matter: the results are basically identical when we manipulate the input style vectors, or the outputs of layers FC1 or FC2.

<img src="https://github.com/Hyeseong0317/CT_Segmentation/blob/main/images/chair2chair.PNG" width="40%">

As mentioned above, there is no principled way to perform sampling using networks trained in a supervised manner. Nonetheless there are some natural heuristics that can be used to obtain “quasi random” chairs. We can first observe that the style input of the network is a probability distribution over styles, which at training time is concentrated on a single style (i.e. c is a one-hot encoding of the chair style). However, in the interpolation experiments we have seen that the network also generates plausible images given inputs with several non-zero entries. This suggests generating random images by using random distributions as input for the network. We tried two families of distributions: (1) we computed the softmax of a Gaussian noise vector with the same size as c, with zero mean and standard deviation σ, and (2) we first randomly selected M styles, then sampled coefficient for each of them from uniform([0, 1]), then normalized to unit sum.

### --> one-hot encoding of the chair style --> random Gaussian noise vector --> We can create the new chair style!
(the second approach is advantageous in that it allows generating images simply from a Gaussian distribution and it is more principled, potentially promising further improvement when better optimized or combined with other kinds of stochastic networks.)

<img src="https://github.com/Hyeseong0317/CT_Segmentation/blob/main/images/chairnetwork.PNG" width="60%">

The difference between (e) and (f) is that in (e) the KL-divergence term in the loss function was weighted 10 times higher than in (f). This leads to much more diverse chairs being generated.

[Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/pdf/1401.4082.pdf)

[Deep Generative Stochastic Networks Trainable by Backprop](https://arxiv.org/pdf/1306.1091.pdf)

Assume the problem we face is to construct a model for some unknown data-generating distribution P(X) given only examples of X drawn from that distribution. In many cases, the unknown distribution P(X) is complicated, and modeling it directly can be difficult. A recently proposed approach using denoising autoencoders transforms the difficult task of modeling P(X) into a supervised learning problem that may be much easier to solve. The basic approach is as follows: given a clean example data point X from P(X), we obtain a corrupted version X˜ by sampling from some corruption distribution C(X˜|X). For example, we might take a clean image, X, and add random white noise to produce X˜. We then use supervised learning methods to train a function to reconstruct, as accurately as possible, any X from the data set given only a noisy version X˜. As shown in Figure 1, the reconstruction distribution P(X|X˜) may often be much easier to learn than the data distribution P(X), because P(X|X˜) tends to be dominated by a single or few major modes (such as the roughly Gaussian shaped density in the figure)

<img src="https://github.com/Hyeseong0317/CT_Segmentation/blob/main/images/corrupted.PNG" width="40%">

[Generative image modeling using style and structure adversarial networks](https://arxiv.org/pdf/1603.05631.pdf)
