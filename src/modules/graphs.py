import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def plot_loss(history):
    '''Takes in a cnn model history file with metrics set to AUC, Recall and Accuracy and 
       returns plots of loss, AUC, Recall and Accuracy on training and validation sets'''
    epochs = range(1, len(history['loss']) + 1)

    fig, ax = plt.subplots(2,2, figsize=(15, 12))
    
    ax[0,0].plot(epochs, history['loss'], 'g.', label='Training loss')
    ax[0,0].plot(epochs, history['val_loss'], 'g', label='Validation loss')

    ax[0,0].set_title('Training and validation loss')
    ax[0,0].set_xlabel('Epochs')
    ax[0,0].set_ylabel('Loss')
    ax[0,0].legend()

    ax[0,1].plot(epochs, history['auc'], 'r.', label='Training auc')
    ax[0,1].plot(epochs, history['val_auc'], 'r', label='Validation auc')
    ax[0,1].set_title('Training and validation AUC')
    ax[0,1].set_xlabel('Epochs')
    ax[0,1].set_ylabel('AUC')

    ax[1,0].plot(epochs, history['acc'], 'b.', label='Training Accuracy')
    ax[1,0].plot(epochs, history['val_acc'], 'b', label='Validation Accuracy')
    ax[1,0].set_title('Training and validation Accuracy')
    ax[1,0].set_xlabel('Epochs')
    ax[1,0].set_ylabel('Accuracy')

    ax[1,1].plot(epochs, history['recall'], 'y.', label='Training Recall')
    ax[1,1].plot(epochs, history['val_recall'], 'y', label='Validation Recall')
    ax[1,1].set_title('Training and validation Recall')
    ax[1,1].set_xlabel('Epochs')
    ax[1,1].set_ylabel('Recall')

    plt.legend()
    return plt.show()

def plot_cmatrix(actual, predictions):
    '''Takes in arrays of actual binary values and model predictions and generates and plots a confusion matrix'''
    cmatrix = confusion_matrix(actual, predictions)

    fig, ax = plt.subplots(figsize = (12,6))
    sns.heatmap(cmatrix, annot=True, fmt='g', ax=ax, cmap='Blues')
    ax.set_xticklabels(['Healthy', 'Pneumonia'])
    ax.set_yticklabels(['Healthy', 'Pneumonia'])
    ax.set_ylabel('Actual', size=15)
    ax.set_xlabel('Predicted', size=15)
    ax.set_title('Confusion Matrix for CNN Predictions', size =18)
  
    return plt.show()

def plot_roc_curve(actual, predictions):
    '''Takes in arrays of actual binary values and model predictions and generates and plots an ROC curve'''
    
    fpr, tpr, threshholds = roc_curve(actual, predictions)
    
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
    
    print('AUC: {}'.format(auc(fpr, tpr)))
    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    return plt.show()