##################################################
# Author: Vinicius Bobato                        #
# Date: 03-24-2025                               #
# Description: Convolutional Neural Network      #
#              Using VGG16 pretrained model      #
##################################################


import os
import re
import datetime
import argparse
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

stamp = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M')

class Data_Loader:
    def __init__(self, data_dir, img_size : tuple, batch_size : int, val_split=0.2) -> None:
        """
        Data augmentation with ImageDataGenerator(). 
        Split the data into training and validation set. 
        Perform operations on the dataset to make the mdoel mroe robust.  
        """
        self.data_directory = data_dir
        self.image_size = img_size
        self.batch_size = batch_size

        self.train_datagen = ImageDataGenerator(
            rescale= 1.0 / 255.0,
            rotation_range = 30,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            validation_split = val_split
        )
        
    def load_data(self):
        """
        Loads the traning and validation data.
        Uses categorical since it is a multi-class classification problem.
        """
        train_model_data = self.train_datagen.flow_from_directory(
            self.data_directory,
            target_size = self.image_size,
            batch_size = self.batch_size,
            class_mode = 'categorical',
            subset = 'training',
            shuffle = True
        )
        
        validation_model_data = self.train_datagen.flow_from_directory(
            self.data_directory,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        return train_model_data, validation_model_data


class CNN_VGG16:
    """
    Convolutional Neural Network model using pretrained VGG16 for feature extraxtion.
    """
    def __init__(self, shape : tuple, num_classes : int) -> None:
        self.shape = shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        """
        Builds the model using VGG16 as the base model.
        """
        self.base_model = VGG16(
            weights='imagenet',
            include_top=False, 
            input_shape=self.shape
        )
        
        for layer in self.base_model.layers:
            layer.trainable = False
        
        for layer in self.base_model.layers[-5:]:
            layer.trainable = True
        
        self.model = models.Sequential([
            self.base_model,
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9), 
                           loss='categorical_crossentropy', 
                           metrics=['accuracy'])    
        return self.model

    def save_model(self, model_name, file_type='keras'):
        """
        Method to save the model. Can be saved as a keras or h5 file. Aslo has the option to save as a tf directory.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Please build the model before saving.")
        
        if file_type == 'h5':
            self.model.save(f"{model_name}.h5")
            print(f"Model saved as {model_name}.h5 file.")
        if file_type == 'keras':
            self.model.save(f"{model_name}.keras")
            print(f"Model saved as {model_name}.keras file.")
        elif file_type == 'tf':
            self.model.save(f"{model_name}")
            print(f"Model saved as {model_name} directory.")
        else:
            raise ValueError("Invalid file type. Please use 'h5', 'keras' or 'tf'.")


class Train:
    """
    Class to train the and evaluate the model. 
    Uses the history object to plot the training and validation accuracy using sklearn and matplotlib.
    """
    def __init__(self, model, train_data, val_data, epochs : int) -> None:
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
    
    def train(self):
        self.history = self.model.fit(
            self.train_data, 
            epochs=self.epochs, 
            validation_data=self.val_data
        )
        
        return self.history
    
    def evaluate(self, output_file_name : str, save_plot : bool):
        global stamp
        y_true = self.val_data.classes
        y_pred = self.model.predict(self.val_data)
        y_pred = y_pred.argmax(axis=1)
        print(f"Validation Accuracy: {100 * self.history.history['val_accuracy'][-1]:.2f}%")

        print("Classification Report")
        print(classification_report(y_true, y_pred, target_names=list(self.val_data.class_indices.keys())))
        c_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 10))
        sns.heatmap(c_matrix, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=self.val_data.class_indices.keys(),
                    yticklabels=self.val_data.class_indices.keys())
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        if save_plot:
            plt.savefig(output_file_name)
        plt.show()


def main(dataset_path, save_plot : bool):
    global stamp
    model_name = stamp +'_VGG16'
    cleaned_path = re.sub(r'[\\/]', '', dataset_path)
    output_file_name = f"{stamp}_confusion_matrix_{cleaned_path}.png"
    
    print("\n-------------------------------Model Info----------------------------------")
    print(f"Model Name     : {model_name}")
    print(f"Dataset Path   : {dataset_path}") 
    print(f"Save CM        : {save_plot}") 
    print(f"CM Name        : {output_file_name}")
    print(f"Classes        : {os.listdir(dataset_path)}")
    print("---------------------------------------------------------------------------\n")
    
    num_classes = len(os.listdir(dataset_path))
    image_size = (224, 224)
    batch_size = 32
    data_loader = Data_Loader(dataset_path, image_size, batch_size)
    train_data, val_data = data_loader.load_data()
    cnn_model = CNN_VGG16(shape=image_size + (3,), num_classes=num_classes)
    model = cnn_model.build_model()
    train = Train(model, train_data, val_data, epochs=10)
    history = train.train()
    cnn_model.save_model(model_name, file_type='keras')
    train.evaluate(output_file_name, save_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VGG16 Model")
    parser.add_argument("-d", required=True, type=str, help="Path to the dataset")
    parser.add_argument("-s", action="store_true", help="Save Confusion Matrix")
    args = parser.parse_args()

    main(dataset_path=args.d, save_plot=args.s)