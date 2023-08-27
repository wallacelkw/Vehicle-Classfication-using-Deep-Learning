# Vehicle Classification
This repository contains Pytorch implementation of our paper titled:  

>In this method we train a convolutional neural network with 6 classes which contain Bus, Cars, Motorcycles, Pickup, Truck and Van. To model it use Inception v1 and EfficientNet b3.

## Source Code Folder
### Main code 
Setup environment
`pip install requirements.txt`

To run the code for training, one can use the following command:  
`python train.py --modelname inception`

To test the model one can use the following command:
`python test.py --modelname inception`

The **Model Architecture of Inception and Efficientnet** will be in 
`efficient.py` and `inception.py`

All the information of function will be in `utils.py`

EarlyStopping class will be stored in `pytorchtools.py`

### Preprocessing File-sub folder
All preprocessing used jupyter lab to perform:
> all the processing will be store in these file which include image augmentation, convert to h5 file, and train test split.
1. Augmentation
`image_augmentation.ipynb`
2. Web Scraping
`scrap_from_website.ipynb`
3. Train, validation, test splitting
`train_test_split.ipynb`
4. Convert into h5 file for machine learning
`convert2h5.ipynb`


### Result sub folder
> The result contain Inception and Efficient classification report, training and validation 's loss and accuracy, confusion matrix and the ***checkpoint file*** of the trained model.

