This main objective of this project is to separate the handwritten text referred to as foreground from the background in the best possible ways.

Inorder to acheive the goal of the project below steps are done:

1. Created synthetic dataset with rules using IAM dataset:
- downloaded the IAM dataset
- added rules:
-- at bottom 
-- change width of rule can be managed from coding 
- applied ElasticTransformation on both ruled image and original image 

2.Remove the rules using below techniques :
- binarization technique 
- removing the rules with binary image sgmentaion using U-Net and then inpaint them using LAMA technique.
- removing the rules with a U-Net 256 classes

3.Extend Dataset to other artifacts and evaluate performance

## Train

+ Dataset in `dataset/{dataset_name}`:
    
    + `input`: contains downloaded IAM images
    + `cropped_data`: contains cropped input images
    + `ruled_data`: contains ruled images
    + `augmented_data`: contains augmented images 
    + `mask_binary_pred`: contains predicted masks using binary classification
    + `removed_rule`: contains resulted images from binarization technique
    + `masks`: contains masks of input ruled images for training
    + `output_lama`: contains predicted images after applying LAMA technique on predicted masked images from binary classification
    + `output_multiclass`: contains predicted images after applying multiclass image segmentation
    + `artifacts`: contains image with additional artifacts like added noise with multiple techniques
    + `artifacts_masks_pred`: contains predicted masks using binary classification on additional artfiacts images
    
  

Log and checkpoints are automatically saved in `lightning_logs_binary` and 'version_9' for binary and multiclass segmentation respectively.
Early stopping is enable by default by pytorch-lightning.
