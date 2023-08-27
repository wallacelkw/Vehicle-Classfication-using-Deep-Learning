import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_curve,
    classification_report
)
import numpy as np
import seaborn as sns
import numpy as np
import itertools
import torch
from torch.autograd.variable import Variable
from tqdm import tqdm
from pytorchtools import EarlyStopping
import torch.optim as optim
from torchvision.models import EfficientNet_V2_L_Weights, ResNet50_Weights
import copy
import mlflow
import mlflow.pytorch
import time
from datetime import date
import pandas as pd
today = str(date.today())

def test(test_loader, model_ft,DEVICE,modelName):
    result_list = []
    label_list = []
    predicted_list = []
    for input,target in tqdm(test_loader,total=len(test_loader)):
        with torch.no_grad():
            input = Variable(input).float().to(DEVICE)
            if modelName == 'Inception':
                output,_,_ = model_ft(input)
            else:
                output = model_ft(input)

            soft_output = torch.softmax(output,dim=-1)
            preds = soft_output.to('cpu').detach().numpy()
            label = target.to('cpu').detach().numpy()
            _,predicted = torch.max(soft_output.data, 1)
            predicted = predicted.to('cpu').detach().numpy()
            for i_batch in range(preds.shape[0]):
                result_list.append(preds[i_batch,1])
                label_list.append(label[i_batch])
                predicted_list.append(predicted[i_batch])
    return (label_list,predicted_list)

def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    mlflow.log_metric(name, value, step=step)


def training(model, num_epochs, train_dataloader, val_dataloader,DEVICE,use_auxiliary=False):
    since = time.time()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)
    criterion = nn.CrossEntropyLoss()
    patience = 15
    train_loss_array = []
    train_acc_array = []
    val_loss_array = []
    val_acc_array = []
    valid_losses = []
    valid_loss = 0.0
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(num_epochs):

        print('Epoch: {} | Learning rate: {}'.format(epoch + 1, scheduler.get_last_lr()))

        epoch_loss = 0
        epoch_correct_items = 0
        epoch_items = 0

        model.train()
        print('Training')
        for inputs, targets in tqdm(train_dataloader,total=len(train_dataloader)):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            if use_auxiliary:
                outputs, aux1, aux2 = model(inputs)
                loss = criterion(outputs, targets) + 0.3 * criterion(aux1, targets) + 0.3 * criterion(aux2, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1)
            correct_items = (preds == targets).float().sum()
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct_items += correct_items.item()
            epoch_items += len(targets)

        train_loss_array.append(epoch_loss / epoch_items)
        train_acc_array.append(epoch_correct_items / epoch_items)
        # display the accuracy each epoch
        each_train_acc = 100* (epoch_correct_items / epoch_items)
        each_train_loss = epoch_loss / epoch_items
        scheduler.step()
        
        model.eval()
        print('Validation')
        with torch.no_grad():
  
            for inputs, targets in tqdm(val_dataloader,total=len(val_dataloader)):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                if use_auxiliary:
                    outputs, _, _ = model(inputs)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, targets)
                preds = outputs.argmax(dim=1)
                correct_items = (preds == targets).float().sum()

                epoch_loss += loss.item()
                epoch_correct_items += correct_items.item()
                epoch_items += len(targets)
                valid_losses.append(loss.item())

        

        # calculate average losses
        valid_loss = np.average(valid_losses)
        val_loss_array.append(epoch_loss / epoch_items)
        val_acc_array.append(epoch_correct_items / epoch_items)

        # display the accuracy each epoch
        each_valid_acc = 100* (epoch_correct_items / epoch_items)
        each_valid_loss = epoch_loss / epoch_items


        log_scalar('training_loss', each_train_loss, epoch)
        log_scalar('training_accuracy', float(each_train_acc), epoch)
        log_scalar('val_loss', each_valid_loss, epoch)
        log_scalar('val_accuracy', float(each_valid_acc), epoch)

        print(f"Training loss: {each_train_loss:.3f}, training acc: {each_train_acc:.3f}")
        print(f"Validation loss: {each_valid_loss:.3f}, validation acc: {each_valid_acc:.3f}")
        print('-'*50)

        early_stopping(valid_loss, model)

        best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        time_elapsed = time.time() - since


        if early_stopping.early_stop:
            print("Early stopping")
            best_model_wts = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_model_wts)
            return best_model_wts, train_loss_array, train_acc_array, val_loss_array, val_acc_array
        
    return best_model_wts, train_loss_array, train_acc_array, val_loss_array, val_acc_array


def visualize_training_results(train_loss_array,
                               val_loss_array,
                               train_acc_array,
                               val_acc_array,
                               num_epochs,
                               model_name,
                               batch_size):
    fig, axs = plt.subplots(1, 2, figsize=(14,4))
    fig.suptitle("{} training | Batch size: {}".format(model_name, batch_size), fontsize = 16)
    axs[0].plot(list(range(1, len(train_loss_array)+1)), train_loss_array, label="train_loss")
    axs[0].plot(list(range(1, len(val_loss_array)+1)), val_loss_array, label="val_loss")
    axs[0].legend(loc='best')
    axs[0].set(xlabel='epochs', ylabel='loss')
    axs[1].plot(list(range(1, len(train_loss_array)+1)), train_acc_array, label="train_acc")
    axs[1].plot(list(range(1, len(val_loss_array)+1)), val_acc_array, label="val_acc")
    axs[1].legend(loc='best')
    axs[1].set(xlabel='epochs', ylabel='accuracy')
    # plt.savefig("Visualize Training Results")
    fig.savefig('loss_n_accuracy.png', bbox_inches='tight')
    # plt.show()

def set_parameter_requires_grad(model, feature_extracting):
    # not fine tuning
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    """
    FINE TUNING
        for params in model.parameters():
                params.requires_grad = True
    """


def plot_confusion_matrix(y_test,y_pred,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True
                          ):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    https://stackoverflow.com/questions/39033880/plot-confusion-matrix-sklearn-with-multiple-labels
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    cm = confusion_matrix(y_test,y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('cividis')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    thresh = cm2.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig("Confusion Matrix.png",bbox_inches = 'tight')
    # plt.show()

    report =classification_report(y_true=y_test,y_pred=y_pred,target_names=target_names,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv("Classification_Report.csv")




