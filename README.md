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

According to the method described in the article, the error data is collected, and the code is

After the collection, the corresponding NPY file is generated and analyzed by jupyter notebook

## KFNN FUSION
KFNN fusion also includes two parts, weight calculation and weighted fusion.

In the weight calculation, we use matlab to complete the code and get the corresponding weighting coefficient. To calculate the parameters of different neural networks, we only need to modify the corresponding R value.

Finally, it is weighted fusion. For details, please refer to .py.
