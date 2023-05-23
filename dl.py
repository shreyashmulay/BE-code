'''DL- 1.A  Linear regression by using Deep Neural network: Implement Boston housing
price.prediction problem by Linear regression using Deep Neural network. Use Boston House price'''
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

scaler = StandardScaler()  # Normalize the features
data = scaler.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)     # Split the data into training and testing sets
# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[13])
])
model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1) # Train the model
loss = model.evaluate(X_test, y_test, verbose=0) # Evaluate the model
print('Test loss:', loss)
predictions = model.predict(X_test)  # Make predictions
print('Example predictions:')       # Print example predictions and actual values
for i in range(10):
    print('Predicted:', predictions[i][0])
    print('Actual:', y_test[i])
    print()

#///////////////////////////////////////////
'''DL- 2 A Binary classification using Deep Neural Networks Example: Classify movie
reviews into positive" reviews and "negative" reviews, just based on the text content of the reviews.Use
IMDB dataset'''
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten
from tensorflow.keras.preprocessing import sequence
# Set the parameters
max_features = 5000  # Top most frequent words to consider
max_length = 300  # Maximum review length (in words)
batch_size = 64
epochs = 3
# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# Pad the sequences to the same length
x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)
# Build the model
model = Sequential()
model.add(Embedding(max_features, 32, input_length=max_length))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

#///////////////////////////////////////////
'''DL 2B Multiclass classification using Deep Neural Networks: Example: Use the OCR
letter recognition dataset https://archive.ics.uci.edu/ml/datasets/letter+recognition'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
# Load the OCR letter recognition dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
data = pd.read_csv(url, header=None)
# Split the dataset into features and labels
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
# Encode the labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Convert the labels to one-hot encoded vectors
num_classes = len(np.unique(y))
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
# Define the model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(16,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

#///////////////////////////////////////////
'''  Dl 3b- Use MNIST Fashion Dataset and create a classifier to classify fashion clothing
into categories. '''

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load the MNIST Fashion Dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize pixel values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into a 784-dimensional vector
    Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    Dense(10, activation='softmax')  # Output layer with 10 units for 10 fashion categories and softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print('Test accuracy:', test_accuracy)

'''_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________'''
''' DL 2B:- ORC Letter Recognition'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn import metrics
(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.imshow(x_train[0], cmap='gray')
plt.show()

print(x_train[0])
print("X_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", x_test.shape)
print("y_test shape", y_test.shape)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32') # use 32-bit precision when training a neural network, so at one point the training data will have to be converted to 32 bit floats. Since the dataset fits easily in RAM, we might as well convert to float immediately.
x_test = x_test.astype('float32')
x_train /= 255  # Each image has Intensity from 0 to 255
x_test /= 255

num_classes = 10
y_train = np.eye(num_classes)[y_train]  # Return a 2-D array with ones on the diagonal and zeros elsewhere.
y_test = np.eye(num_classes)[y_test]

# Define the model architecture
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))  # The input_shape argument is passed to the foremost layer. It comprises of a tuple shape,
model.add(Dropout(0.2)) # DROP OUT RATIO 20%
model.add(Dense(512, activation='relu')) #returns a sequence of vectors of dimension 512  
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',  # for a multi-class classification problem  
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Train the model
batch_size = 128 # batch_size argument is passed to the layer to define a batch size for the inputs.
epochs = 20
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, # verbose=1 will show you an animated progress bar eg. [==========]
                    validation_data=(x_test, y_test))
# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

'''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''DL 3B :- Fashion mnist code'''

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
plt.imshow(x_train[1])
plt.imshow(x_train[0])

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_train.shape
x_test.shape
y_train.shape
y_test.shape

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.25),
    
    keras.layers.Conv2D(128, (3,3), activation='relu'),
     keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(10, activation='softmax') ])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)

'''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

'''IMDB review positive or negative '''

import numpy as np
import pandas as pd
from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000) # you may take top 10,000 word frequently review of movies
#consolidating data for EDA(exploratory data analysis: involves gathering all the relevant data into one place and preparing it for analysis )
data = np.concatenate((X_train, X_test), axis=0)
label = np.concatenate((y_train, y_test), axis=0)
print("Review is ",X_train[5])
print("Review is ",y_train[5])
vocab=imdb.get_word_index()
print(vocab) 
data
label
X_train.shape
X_test.shape
y_train
y_test

def vectorize(sequences, dimension = 10000):
  # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

test_x = data[:10000]
test_y = label[:10000]
train_x = data[10000:]
train_y = label[10000:]

test_x
test_y
train_x
train_y
print("Categories:", np.unique(label))
print("Number of unique words:", len(np.unique(np.hstack(data))))
length = [len(i) for i in data]
print("Average Review length:", np.mean(length))
print("Standard Deviation:", round(np.std(length)))
print("Label:", label[0])
print(data[0])
index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()]) 
decoded = " ".join( [reverse_index.get(i - 3, "#") for i in data[0]] ) #The purpose of subtracting 3 from i is to adjust the indice,# to indicate that the index was not found.
print(decoded) 
index
reverse_index
decoded
data = vectorize(data)
label = np.array(label).astype("float32")
data
label
import seaborn as sns #seaborn is a popular Python visualization library that is built on top of Matplotlib and provides a high-level interface for creating informative and attractive statistical graphics.
sns.set(color_codes=True)
import matplotlib.pyplot as plt # %matplotlib to display Matplotlib plots inline with the notebook
%matplotlib inline 
labelDF=pd.DataFrame({'label':label})
sns.countplot(x='label', data=labelDF)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,label, test_size=0.20, random_state=1)
X_train.shape
X_test.shape
from keras.utils import to_categorical
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu")) #ReLU" stands for Rectified Linear Unit, and it is a commonly used activation function in neural networks. 
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid")) #adds another Dense layer to the model, but with a single neuron instead of 50,i.e. out put layer,it produces the output predictions of the model.
model.summary()
import tensorflow as tf #TensorFlow provides a wide range of tools and features for data processing, model building, model training, and model evaluation.
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)
results = model.fit(
 X_train, y_train,
 epochs= 2,
 batch_size = 500,
 validation_data = (X_test, y_test),
 callbacks=[callback]
)
print(np.mean(results.history["val_accuracy"]))
print(results.history.keys())
# summarize history for accuracy
plt.plot(results.history['accuracy']) #Plots the training accuracy of the model at each epoch.
plt.plot(results.history['val_accuracy']) #Plots the validation accuracy of the model at each epoch.
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.predict(X_test)









































