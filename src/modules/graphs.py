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




# ========================= Visualizing False Positives ================================== # 


def get_false_positives(predictions, y_test):
    """
    Returns a numpy array of index matched false negatives
    predictions --> binary or bool
    y_test --> binary or bool
    theshold 
    
    returns a np.array
    """
    comparisons = list(zip(y_test, predictions))
    return np.array([1 if (true == 0 and prediction == 1) else 0 for true, prediction in comparisons])


# Create a function to return a list of 10 images from the false positive class

def import_image(PATH, name, shape=(224,224)):
    """
    PATH --> str: Relative path to image directoy
    eg 'src/data/xray/train/''
    image_names --> list -> str: Names of the images to load
    
    Returns:
    List -> PIL images
    
    """
    
    # create path to file
    img_path = PATH + name
    # load file and return pil
    return image.load_img(img_path).resize(shape)



def get_image_names(image_dict, false_positives, key='test'):
    """
    image_dict -> dict
    false_postives -> array
    key -> string
    imbalanc -> bool
    """
    tuple_ = image_dict[key]
    image_names = [tup[0] for tup in tuple_]

    name_plus_flags = list(zip(image_names, false_positives))
    
    
    
    return [i for i,j in name_plus_flags if j == 1]
    

def display_bv_images(image_list, PATH, shape=(224,224)):
    
    fig = plt.figure(figsize = (12,6))
    for i, image in enumerate(image_list):  
        label = f'Image {i+1}'
        ax = fig.add_subplot(2, 3, i+1)
        
        # to plot without rescaling, remove target_size
        plt.imshow(image_list[i].resize(shape), cmap='Greys_r')
        plt.title(label)
        plt.axis('off')
    plt.show()
    

def see_false_positives(image_dict, y_hat, PATH="./src/data/x_ray/NORMAL/", shape=(100,100), key='test', num_images=2):
    """
    Image_dict: dict; str -> tuple(str, matrix, int)
    y_hat -> prediction, list->int
    PATH -> str, directory to the normal xray images
    shape -> tuple-> (int,int)
    key -> str: The set from the dictionary tha you want to visualize. eg val, train, test
    
    
    Need numpy and matplotlib. Will display 6 random images from the false positive class.
    
    """
    # get true positives
    y_true  = [i[2] for i in image_dict[key]]
    
    # get false positives
    false_p = get_false_positives(y_hat, y_true)
    
    # get names of the false positives
    image_names = get_image_names(image_dict, false_p, key=key)
    
    # select 6 random images from the false positive names
    random_images = np.random.choice(a=image_names, size=num_images, replace=False)
    # import the 6 images 
    
    image_list = [import_image(PATH, image_name, shape=shape) for image_name in random_images]
    
    # display 6 random false positives 
    display_bv_images(image_list, PATH, shape=shape)
    
    