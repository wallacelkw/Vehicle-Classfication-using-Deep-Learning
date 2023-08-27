from PIL import Image
import os
import torch
from torch.autograd.variable import Variable
from torchvision import models, transforms,datasets
from utils import (initialize_model,
    plot_confusion_matrix,
    test,
    cuda_checking,
    )
import pandas as pd
import shutil
import argparse
from inception import Inception
from efficient import EfficientNet


if __name__=='__main__':

    if torch.cuda.is_available():
        print('CUDA is available. Working on GPU')
        DEVICE = torch.device('cuda')
    else:
        print('CUDA is not available. Working on CPU')
        DEVICE = torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="resnet, efficient")

    args = parser.parse_args()
    data_dir_test = os.path.join(os.getcwd(),'finalise_dataset','Dataset','test')
    num_classes = len(os.listdir(data_dir_test))
    # modelName = args.model

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


    ### Test the result
    
    # get the classes from train file
    
    # model_ft, input_size = initialize_model(modelName, num_classes, feature_extract=True, use_pretrained=True)
    valid_transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # load the transformation of the images
    image_datasets_test = datasets.ImageFolder(data_dir_test,transform = valid_transform)
    test_loader = torch.utils.data.DataLoader(image_datasets_test, batch_size=4, shuffle=True, num_workers=0)

    # load trained pt file to evaluate the testing
    destination = os.path.join(os.getcwd(),'Result',modelName)
    model_ft = model_ft.to(DEVICE)
    checkpoint1 = os.path.join(destination,'checkpoint.pt')
    model_ft.load_state_dict(torch.load(checkpoint1))
    model_ft.eval()
    y_test, y_predict = test(test_loader, model_ft, DEVICE,modelName)
    print("Done Testing......")


    categories = ['bus', 'car', 'motocycle','pickupTruck', 'truck', 'van']
    plot_confusion_matrix(y_test, y_predict,categories)
    

    # store label and predlabel to csv just in case want to refer
    df = pd.DataFrame({'Labels': y_test,
                   'Predict': y_predict})
    df.to_csv(modelName+'.csv')
    moved_file = [modelName+'.csv', 'Confusion Matrix.png',"Classification_Report.csv"]
    print("Moved all the requirement file.....")
    for idx,name in enumerate(moved_file):
        moved_path = os.path.join(destination,name)
        original_path = os.path.join(os.getcwd(),name)
        shutil.move(original_path,moved_path)

    print("Moved Done")


