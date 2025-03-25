import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
            rescale=1.0 / 255.0,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=val_split
        )
        
        
    def load_data(self):
        """
        Loads the traning and validation data.
        Uses categorical since it is a multi-class classification problem.
        """
        train_model_data = self.train_datagen.flow_from_directory(
            self.data_directory,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_model_data = self.train_datagen.flow_from_directory(
            self.data_directory,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        return train_model_data, validation_model_data
    
class CNN:
    """
    Convolutional Neural Network model.
    """
    def __init__(self, shape : tuple, num_classes : int) -> None:
        self.shape = shape
        self.num_classes = num_classes

    def build_model(self):
        """
        CNN model with 3 Conv2D layers, 3 MaxPooling2D layers, 2 Dense layers and 1 Dropout layer.
        """
        self.model = keras.models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.shape),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model


class Train:
    def __init__(self, model, train_data, val_data, epochs : int) -> None:
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
    
    def train(self):
        self.history = self.model.fit(
            self.train_data, 
            epochs=self.epochs, 
            validation_data=self.val_data)
        return self.history
    
    def evaluate(self):
        self.model.evaluate(self.val_data)
        print("Validation Accuracy: ", self.history.history['val_accuracy'][-1])

def main():
    data_directory = 'data/'
    num_classes = len(os.listdir(data_directory))
    image_size = (224, 224)
    batch_size = 32
    data_loader = Data_Loader(data_directory, image_size, batch_size)
    train_data, val_data = data_loader.load_data()
    cnn_model = CNN(shape=image_size + (3,), num_classes=num_classes)
    model = cnn_model.build_model()
    train = Train(model, train_data, val_data, epochs=10)
    history = train.train()
    train.evaluate()

if __name__ == '__main__':
    main()