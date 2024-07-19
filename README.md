
# UNET model for Airbus Ship Detection Challenge
## Solution
### Dataset for training and model inputs
 * Large amount of dataset consists of images with no ships. So, were used 9000 images with ships, and 1000 without.
We splitted 80 % for training, 20 % - for validation
 * Due to large input images - 768x768x3, we model are using model with 256x256x3 input shape. This decision came from large model size, which we got from original inputs.
 * To train model, we cutted input train image as 3x3 grid, and selected tile with the largest ship area in it's mask
###  Training
 * Used **BCE+DiceLoss**
 * Used Adam optimizer:
   * 8 epochs with 0.001 lr    
   * 8 epochs with 0.0005 lr
   * 8 epochs with 0.0001 lr
   * 8 epochs with 0.00001 lr
### Results   
We got the best validation dice score  - **0.8627**

![alt text](https://github.com/grinvolod13/airbus-ship-detection-challange-unet/blob/master/Untitled.png?raw=true)

You can checkout Kaggle notebook with solution, where model was trained (Thanks, Kaggle!)

https://www.kaggle.com/code/volodymyrhryniuk/unet-for-airbus-ship-segmentation-challange/notebook
