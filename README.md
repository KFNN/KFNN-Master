# KFNN-Master
The Code for Multiple Neural Networks Fusion Based on Kalman Filter

The whole system consists of two parts: the noise estimation of neural network and the fusion of multiple neural network based on kfnn.
## DataSet
The noise estimation and fusion performance evaluation of neural network are carried out in [Imagenet](https://image-net.org/).

Apply `./ImageNet/val.py` to process ImageNet and organize the data into the following structure. 
###
    /ImageNet
      /val
        /n01440764
          images
        /n01443537
          images
      /train
      /test
###

## Pretrained Models in ImagNet
Thanks to the open pretrained model provided in [cadene/trained-models.torch](https://github.com/Cadene/pretrained-models.pytorch), we used 16 classic pretrained models as the baselines, including ***NASNetlarge, AlexNet, DenseNet121, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, VGG11, VGG11\_bn, VGG13, VGG13\_bn, VGG16, VGG16\_bn, VGG19, VGG19\_bn***. We evaluated their performance on ImageNet separately.

We followed [cadene/trained-models.torch](https://github.com/Cadene/pretrained-models.pytorch), completed the download and application of the pretrained models.

### Accuracy on validation set (single model)
On our machine, the verification accuracy of the pretrained models is shown in the following table.
|  pretrained model   | Acc@1  |  
|  ----   | ----  |     
| NASNetlarge  | 82.506% |     
| AlexNet  | 56.518% |
| DenseNet121  | 74.434% |
| ResNet18  | 69.758% |
| ResNet34  | 73.314% |
| ResNet50  | 76.130% |
| ResNet101  | 77.374% |
| ResNet152  | 78.312% |
| VGG11  | 69.020% |
| VGG11\_bn  | 70.370% |
| VGG13  | 69.928% |
| VGG13\_bn  | 71.586% |
| VGG16  | 71.592% |
| VGG16\_bn  | 73.360% |
| VGG19  | 72.376% |
| VGG19\_bn  | 74.218% |

## Neural network noise estimation
Neural network noise estimation includes two parts, one is data acquisition, the other is data fitting analysis.

According to the method described in the article, the error data is collected, and the code is `./imagenet_eval.py`, 
for example, run

`python  ./imagenet_eval.py --data ImageNet -a vgg16 -b 100 -e`

Using the above method, the evaluation results (npy file) of all neural networks are obtained. Then the move the results  to `Noise Estimation` folder.

In `./Noise Estimation`,  the detailed code of evaluating the neural network fitting is in the [NNNE_ImageNet100.ipynb](https://github.com/KFNN/KFNN-Master/blob/main/Noise_Estimation/NNNE_ImageNet100.ipynb), and the fitting image is saved in the `./Noise Estimation/save` folder.

The code of normality test are in [Normality test.ipynb](https://github.com/KFNN/KFNN-Master/blob/main/Noise_Estimation/Normality%20test.ipynb)

According to the neural network observer variance determination method, the calculated observation error is as follows.

|  pretrained model   | R |  
|  ----   | ----  |     
| NASNetlarge  | 0.0306 |     
| AlexNet  | 0.1891 |
| DenseNet121  | 0.0654 |
| ResNet18  | 0.0915 |
| ResNet34  | 0.0712 |
| ResNet50  | 0.0570 |
| ResNet101  | 0.0512 |
| ResNet152  | 0.0470 |
| VGG11  | 0.0960 |
| VGG11\_bn  | 0.0878 |
| VGG13  | 0.0904 |
| VGG13\_bn  | 0.0807 |
| VGG16  | 0.0807 |
| VGG16\_bn  | 0.0710 |
| VGG19  | 0.0763 |
| VGG19\_bn  | 0.0665 |

## KFNN FUSION
KFNN fusion also includes two parts, weight calculation and weighted fusion.

In the weight calculation, we use matlab to complete the code and get the corresponding weighting coefficient. To calculate the parameters of different neural networks, we only need to modify the corresponding R value. The relevant code is in [./KFNN_FUSION/FUSE_2_OK.m](https://github.com/KFNN/KFNN-Master/blob/main/KFNN_FUSION/FUSE_2_OK.m). 

Finally, it is weighted fusion. For details, please refer to [KFNN_Evaluate.ipynb](https://github.com/KFNN/KFNN-Master/blob/main/KFNN_FUSION/KFNN_Evaluate.ipynb)

## Other Description
In addition to Imagenet, we also verify the performance of KFNN in CIFAR-10. The specific experimental results can be found in the experiment.

This open-source code can only be used for academic research.

## Reference
[Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)
