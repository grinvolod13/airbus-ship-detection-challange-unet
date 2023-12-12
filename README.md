
# (added later) UNET model for Airbus Ship Detection Challenge
## Solution
### Dataset for training and model inputs
 * Large amount of dataset consists of images with no ships. So, were used 9000 images with ships, and 1000 without.
We splitted 80 % for training, 20 % - for validation
 * Due to large input images - 768x768x3, we model with 256x256x3 input shape. This decision came from large model size, which we got from original inputs.
 * To train model, we cutted input train image as 3x3 grid, and selected tile with the largest ship area in it's mask
###  Training
 * Used **BCE+DiceLoss** loss, due to larger training per epoch, compared to **BFCE+DiceLoss** (also tried)
 * Used Adam optimizer:
   * 24 epochs with 1e-4 lr    
   * 6 epochs with 5e-5 lr
   * 6 epochs with 1e-5 lr
### Results   
We got the best validation dice score  - **0.64**

![alt text](https://github.com/grinvolod13/airbus-ship-detection-challange-unet/blob/master/Untitled.png?raw=true)

This model get **0.51248** private score, and **0.34559** public score on **Kaggle**

We also trained same model on **BFCE+DiceLoss** loss on smaller number of epochs and smaller dataset, which gave us **0.40143** and **0.34651**, but its not valid comparsion due to different training.

You can checkout Kaggle notebook with solution, where model was trained (Thanks, Kaggle!)

https://www.kaggle.com/code/volodymyrhryniuk/unet-for-airbus-ship-segmentation-challange/notebook
