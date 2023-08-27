import torch
import numpy as np
from torchvision import datasets, transforms
import os
import random
from utils import (initialize_model,
    cuda_checking,
    training,
    visualize_training_results
    )
import shutil
import os
import argparse

import mlflow
import mlflow.pytorch
from datetime import date
from inception import *
from efficient import EfficientNet
today = str(date.today())

if __name__ == '__main__':
    # mlflow.set_tracking_uri('http://127.0.0.1:5000')  # set up connection
    # mlflow.set_experiment('test-experiment')          # set the experiment
    # mlflow.pytorch.autolog()

    
    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", help="resnet, efficient")
    parser.add_argument("--pretrained", help="Transfer learning")
    parser.add_argument("--model", help="efficient, inception")
    args = parser.parse_args()

    DEVICE = cuda_checking()
    data_dir = os.path.join(os.getcwd(),'Dataset')
    # get the classes from train file
    num_classes = len(os.listdir(os.path.join(data_dir,'train')))

    if args.pretrained == "true":
        feature_extract=True
        use_pretrained=True
    elif args.pretrained == "false":
        feature_extract=False
        use_pretrained=False



    # modelName = args.model
    # model_ft, input_size = initialize_model(modelName, num_classes, feature_extract=feature_extract, use_pretrained=use_pretrained)


    if args.model == "inception":
        model_ft = Inception()
        input_size = 224
        modelName = 'Inception'
        use_auxiliary = True
    elif args.model =="efficient":
        version="b3"
        model_ft = EfficientNet(version=version, num_classes=num_classes)
        modelName = 'EfficientNet'
        input_size = 224
        use_auxiliary = False

    train_transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        # transforms.RandomRotation(degrees=(30, 70)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    # the validation transforms
    valid_transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    # initial all the transform and original images into tensor
    image_datasets_train = datasets.ImageFolder(root = os.path.join(data_dir, 'train'),
                                                transform = train_transform
                                            )
    image_datasets_valid = datasets.ImageFolder(root = os.path.join(data_dir, 'valid'),
                                                transform = valid_transform
                                            )
    image_datasets_original_train = datasets.ImageFolder(os.path.join(data_dir, 'train'),transform=valid_transform) 
    image_datasets_original_valid = datasets.ImageFolder(os.path.join(data_dir, 'valid'),transform=valid_transform)


    # combine the transform and original for augmenration
    increased_dataset_train = torch.utils.data.ConcatDataset([image_datasets_train,image_datasets_original_train])
    increased_dataset_valid = torch.utils.data.ConcatDataset([image_datasets_valid,image_datasets_original_valid])



    # Send the model to GPU
    model_ft = model_ft.to(DEVICE)
    # parameter setup
    num_epochs = 100
    batch_size = 32
     # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(increased_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(increased_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=0)

    with mlflow.start_run() as run:
        mlflow.log_param('dataset', data_dir)
        mlflow.log_param('model name', modelName)
        mlflow.log_param('number of classes', num_classes)
        mlflow.log_param('Batch size', batch_size)
        mlflow.log_param('epochs', num_epochs)
        # mlflow.log_param('feature extracted', feature_extract)
        # mlflow.log_param('pre-trained', pre_trained)

        model_training_results = training(model_ft, num_epochs, 
                                    training_loader, validation_loader,
                                    DEVICE,use_auxiliary=use_auxiliary)

        model_ft, train_loss_array, train_acc_array, val_loss_array, val_acc_array = model_training_results

        min_loss = min(val_loss_array)
        min_loss_epoch = val_loss_array.index(min_loss)
        min_loss_accuracy = val_acc_array[min_loss_epoch]

        visualize_training_results(train_loss_array,
                                val_loss_array,
                                train_acc_array,
                                val_acc_array,
                                num_epochs,
                                model_name=modelName,
                                batch_size=batch_size)
        print("\nTraining results:")
        print("\tMin val loss {:.4f} was achieved during epoch #{}".format(min_loss, min_loss_epoch + 1))
        print("\tVal accuracy during min val loss is {:.4f}".format(min_loss_accuracy))

        # move the images and information to particular file
        destination = os.path.join(os.getcwd(),'Result',modelName)
        if not os.path.exists(destination):
            os.makedirs(destination)
        
        # move checkpoint, and visualise_images
        moved_file = ['checkpoint.pt', 'loss_n_accuracy.png']
        for idx,name in enumerate(moved_file):
            moved_path = os.path.join(destination,name)
            original_path = os.path.join(os.getcwd(),name)
            shutil.move(original_path,moved_path)
        
        print("Moved checkpoint and Losses images")

        save_model = r'C:\Users\kimwa\Desktop\MMU\Computer Intellegience\Project\models'
        # mlflow.pytorch.log_model(model_ft,"models")
        # mlflow.pytorch.save_model(model_ft,save_model+today+'/')

    
