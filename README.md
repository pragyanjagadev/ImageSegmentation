## Objective
This main objective of this project is to separate the handwritten text referred to as foreground from the background in the best possible ways.

Inorder to achieve the goal of the project below steps are done:

1. Created synthetic dataset with rules using IAM dataset:
- downloaded the IAM dataset
- added rules:
-- at bottom 
-- change width of rule can be managed from coding 
- applied ElasticTransformation on both ruled image and original image 

2.Remove the rules using below techniques :
- binarization technique 
- removing the rules with binary image segmentation using U-Net and then inpaint them using LAMA technique.
- removing the rules with a U-Net 256 classes

3.Extend Dataset to other artifacts and evaluate performance

## Files and Dataset

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
    
+ File structure:
    
    + `rules.py` is used to create rules using IAM dataset
    + `augment.py` is used to create augmented images
    + `masks.py` is used to create masks on IAM dataset
    + `functions.py` is used to keep all reusable functions
    
    + `binarization_removeline.py` is used to remove rules using binarization technique

    + `unet_binary.py` is used to build the architecture of binary classfication using unet 
    + `train_binary.py` is used to train the model
    + `test_binary.py/binary_model.ipynb`  is used to test data
    + `lama/lama.ipynb` is used to implemet lama on result from test dataset

    + `Unet.py` is used to build the architecture of multiclass classfication using unet 
    + `train.py/multiclass_train.ipynb`  is used to train the model
    + `test.py` is used to test data

    + `eval_dataloader.py` is used to load data of test dataset
    + `evaluator.py is` used to evaluate performance 
    + `qualitative_analysis.ipynb` is used to do some qualitative analysis on predicted result

    +additional_artifacts.py is used to add some additional noise on test dataset
    +test_binary_artifacts.py is used to test the model on newly developed dataset



Log and checkpoints are automatically saved in `lightning_logs_binary` and 'version_9' for binary and multiclass segmentation respectively.
Early stopping is enable by default by pytorch-lightning.
