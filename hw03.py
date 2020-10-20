import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import struct
import numpy as np
from keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



tf.__version__

TRAIN_IMAGES = '/workspace/DeepLearning/mnist/train-images.idx3-ubyte'
TRAIN_LABELS = '/workspace/DeepLearning/mnist/train-labels.idx1-ubyte'
TEST_IMAGES = '/workspace/DeepLearning/mnist/t10k-images.idx3-ubyte'
TEST_LABELS = '/workspace/DeepLearning/mnist/t10k-labels.idx1-ubyte'

def load_images(file_name):
    binfile = open(file_name,'rb')
    buffers = binfile.read()
    magic,num,rows,cols = struct.unpack_from('>IIII',buffers,0)
    bits = num*rows*cols
    images = struct.unpack_from('>'+str(bits)+'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images,[-1,rows,cols])
    return images

def load_labels(file_name):
    binfile = open(file_name, 'rb') 
    buffers = binfile.read()
    magic,num = struct.unpack_from('>II', buffers, 0) 
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    binfile.close()
    labels = np.reshape(labels, [num])
    return labels  

train_images = load_images(TRAIN_IMAGES)
train_labels = load_labels(TRAIN_LABELS)
test_images = load_images(TEST_IMAGES)
test_labels = load_labels(TEST_LABELS)

train_images = train_images.reshape(-1,28,28,1)
train_images = train_images.astype('float32')
test_images = test_images.reshape(-1,28,28,1)
test_images = test_images.astype('float32')
train_images = train_images/255.
test_images = test_images/255.
print('The shape of training images is {}'.format(np.shape(train_images)))
print('The shape of testing images is {}'.format(np.shape(test_images)))

batch_size = np.shape(train_images)[0]


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(28, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x=train_images,y=train_labels, epochs=10,validation_split=.2)

test_pred_raw = model.predict(test_images)
test_pred = np.argmax(test_pred_raw,axis=1)
model.evaluate(test_images,test_labels)

TEST_IMAGE_NUMBER = 5
layer_outpus = [layer.output for layer in model.layers]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = model.layers[0].output)
inputs = test_images[TEST_IMAGE_NUMBER]

inputs = inputs.reshape(1,28,28,1)
feature_maps = visualization_model.predict(inputs)

'''-----------------------------------VISUALIZATION----START---------------------------------------'''
feature_map = feature_maps[0]
n_features = feature_map.shape[-1]
size = feature_map.shape[1]
display_grid = np.zeros((size,size*7))
plt.figure(1)
plt.imshow(np.squeeze(test_images[5]))
for i in range(7):
    display_grid[:,i*size:(i+1)*size] = feature_map[:,:,i]

plt.figure(figsize=(14,2))
plt.imshow(display_grid,aspect = 'auto',cmap='viridis')
'''-----------------------------------VISUALIZATION----END---------------------------------------'''


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues): 
    if normalize: 
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
        print("Normalized confusion matrix") 
    else: 
        print('Confusion matrix, without normalization') 
        # print(cm) 
    plt.imshow(cm, interpolation='nearest', cmap = 'Greys') 
    plt.title(title) 
    plt.colorbar() 
    tick_marks = np.arange(len(classes)) 
    plt.xticks(tick_marks, classes, rotation=45) 
    plt.yticks(tick_marks, classes) 
    fmt = '.2f' if normalize else 'd' 
    thresh = cm.max() / 2. 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 
        plt.text(j, i, format(cm[i, j], fmt), 
            horizontalalignment="center", 
            color="white" if cm[i, j] > thresh else "black") 
    plt.tight_layout() 
    plt.ylabel('True label') 
    plt.xlabel('Predicted label') 

test_pred_raw = model.predict(test_images)
test_pred = np.argmax(test_pred_raw,axis=1)
classes = [str(i) for i in range(10)]
confusion_matrix = confusion_matrix(test_labels,test_pred)
plt.figure(3)
plot_confusion_matrix(confusion_matrix,classes,normalize=True,title='Test confusion matrix')



'''--------------------------------LEARNING---CURVE-----------------------------------'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.figure(4)
plt.plot(epochs,acc,'r','Training Accuraccy')
plt.plot(epochs,val_acc,'b','Validation Accuraccy')
plt.title('Training and validation accuraccy')
plt.legend()

plt.figure(5)
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')
plt.legend(['Training Loss','Validation Loss'])
plt.show()