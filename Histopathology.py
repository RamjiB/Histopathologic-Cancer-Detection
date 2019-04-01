import pandas as pd
import numpy as np
import keras
import os
import shutil
import skimage.io as skio
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

file = pd.read_csv("train_labels.csv")

file =file[file['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2'] 
file =file[file['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe'] 

train,valid = train_test_split(file,test_size =0.33, random_state=42)

print(train.shape)
print(valid.shape)

batch_size = 90
epochs = 1
print("----------------------------------------------------")
print(train.head())
print("----------------------------------------------------")
#def create_folder(folderName):
#    if not os.path.exists(folderName):
#        try:
#            os.makedirs(folderName)
#        except OSError as exc:
#            if exc.errno != errno.EEXIST:
#                raise
#base_dir = 'data'
#create_folder(base_dir)
#train_dir = os.path.join(base_dir,'train_dataset')
#create_folder(train_dir)
#valid_dir = os.path.join(base_dir,'valid_dataset')
#create_folder(valid_dir)

#train_tum = os.path.join(train_dir,'0')
#create_folder(train_tum)
#train_notum = os.path.join(train_dir,'1')
#create_folder(train_notum)

#valid_tum = os.path.join(valid_dir,'0')
#create_folder(valid_tum)

#valid_notum = os.path.join(valid_dir,'1')
#create_folder(valid_notum)

#for images in range(0,len(train)):
#    file_name = train.iloc[images].values[0] + '.tif'
#    y_train = train.iloc[images].values[1]
#    if(y_train == 0):
#        train_path = train_tum
#    else:
#        train_path = train_notum
#    src = os.path.join('dataset/train/' , file_name)
#    dest = os.path.join(train_path , file_name)
#    shutil.copyfile(src,dest)

#for images in range(0,len(valid)):
#    file_name = valid.iloc[images].values[0] + '.tif'
#    y_valid = valid.iloc[images].values[1]
#    if(y_valid == 0):
#        valid_path = valid_tum
#    else:
#        valid_path = valid_notum
#    src = os.path.join('dataset/train/' , file_name)
#    dest = os.path.join(valid_path , file_name)
#    shutil.copyfile(src,dest)



datagen = ImageDataGenerator(rescale=1.0/255,
				horizontal_flip=True,
				vertical_flip=True)
train_gen = datagen.flow_from_directory('data/train_dataset/' , 
                                        target_size = (96,96) , 
                                        batch_size = batch_size,
                                       class_mode ='categorical')

def tr(train_gen):
    for x,y in train_gen:
        yield ([x,y],[y,x])

valid_gen = datagen.flow_from_directory('data/valid_dataset/',
					target_size = (96,96),
					batch_size = batch_size,
					class_mode='categorical')
print("valid: ",valid.shape[0])
def patches(mode):
	
	if (mode == 'valid'):
		xy = valid_gen
	else:
		xy = test_gen

	batches = 0
	for x,y in xy:
		s = x.shape
		img = x[:,32:64,32:64,:]
		img = np.resize(img,s)
		batches += 1
		yield ([img,y],[y,img])

from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,SeparableConv2D,Dropout,Flatten,Dense,BatchNormalization,GlobalAveragePooling2D
from keras import layers,models
from keras import initializers

#Capsule Net Architecture
#---------------------------------------------------------------------------
class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss
    inputs: shape=[dim_1, ..., dim_{n-1}, dim_n]
    output: shape=[dim_1, ..., dim_{n-1}]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
    Output shape: [None, d2]
    """
    def call(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of vectors of capsules
            x = inputs
            # Enlarge the range of values in x to make max(new_x)=1 and others < 0
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

        # masked inputs, shape = [batch_size, dim_vector]
        inputs_masked = K.batch_dot(inputs, mask, [1, 1])
        return inputs_masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][-1]])
        else:
            return tuple([None, input_shape[-1]])

class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
    Output shape: [None, d2]
    """
    def call(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of vectors of capsules
            x = inputs
            # Enlarge the range of values in x to make max(new_x)=1 and others < 0
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

        # masked inputs, shape = [batch_size, dim_vector]
        inputs_masked = K.batch_dot(inputs, mask, [1, 1])
        return inputs_masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][-1]])
        else:
            return tuple([None, input_shape[-1]])

def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_vector] and output shape = \
    [None, num_capsule, dim_vector]. For Dense Layer, input_dim_vector = dim_vector = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_vector: dimension of the output vectors of the capsules in this layer
    :param num_routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_vector]"
        
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector],
                                 initializer=self.kernel_initializer,
                                 name='W')
        print(self.W)

        # Coupling coefficient. The redundant dimensions are just to facilitate subsequent matrix calculation.
        self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_vector]
        # Expand dims to [None, input_num_capsule, 1, 1, input_dim_vector]
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now it has shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        """  
        # Compute `inputs * W` by expanding the first dim of W. More time-consuming and need batch_size.
        # Now W has shape  = [batch_size, input_num_capsule, num_capsule, input_dim_vector, dim_vector]
        w_tiled = K.tile(K.expand_dims(self.W, 0), [self.batch_size, 1, 1, 1, 1])
        
        # Transformed vectors, inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = K.batch_dot(inputs_tiled, w_tiled, [4, 3])
        """
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0. This is faster but requires Tensorflow.
        # inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))
        """
        # Routing algorithm V1. Use tf.while_loop in a dynamic way.
        def body(i, b, outputs):
            c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))
            b = b + K.sum(inputs_hat * outputs, -1, keepdims=True)
            return [i-1, b, outputs]

        cond = lambda i, b, inputs_hat: i > 0
        loop_vars = [K.constant(self.num_routing), self.bias, K.sum(inputs_hat, 1, keepdims=True)]
        _, _, outputs = tf.while_loop(cond, body, loop_vars)
        """
        # Routing algorithm V2. Use iteration. V2 and V1 both work without much difference on performance
        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
            # outputs.shape=[None, 1, num_capsule, 1, dim_vector]
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            # last iteration needs not compute bias which will not be passed to the graph any more anyway.
            if i != self.num_routing - 1:
                # self.bias = K.update_add(self.bias, K.sum(inputs_hat * outputs, [0, -1], keepdims=True))
                self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
            # tf.summary.histogram('BigBee', self.bias)  # for debugging
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])


def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_vector: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_vector]
    """
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_vector])(output)
    return layers.Lambda(squash)(outputs)


def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 4d, [None, width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer.
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(784, activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=[96, 96, 3], name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return models.Model([x, y], [out_caps, x_recon])

main_model = CapsNet((96,96,3),2,3)
main_model.summary()

#-----------------------------------------------------------------------------------

#main_model = Sequential()
#main_model.add(Conv2D(filters = 32,kernel_size=(5,5),
#                              activation='relu',input_shape=(96,96,3)))
#main_model.add(BatchNormalization())
#main_model.add(Conv2D(filters = 32, kernel_size = (3,3), 
#                          activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(Conv2D(filters = 32, kernel_size = (3,3), 
#                          activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(MaxPool2D(pool_size=(2, 2)))
#main_model.add(Dropout(0.2))
#main_model.add(Conv2D(filters = 64, kernel_size = (3,3), 
#                 activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(Conv2D(filters = 64, kernel_size = (3,3), 
#                 activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(Conv2D(filters = 64, kernel_size = (3,3), 
#                 activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(MaxPool2D(pool_size=(2, 2)))
#main_model.add(Dropout(0.20))
#main_model.add(Conv2D(filters = 128, kernel_size = (3,3), 
#                 activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(Conv2D(filters = 128, kernel_size = (3,3), 
#                 activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(Conv2D(filters = 128, kernel_size = (3,3), 
#                 activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(MaxPool2D(pool_size=(2, 2)))
#main_model.add(Dropout(0.25))

# main_model.add(SeparableConv2D(filters = 512, kernel_size = (3,3), 
#                  activation='relu'))
# main_model.add(SeparableConv2D(filters = 512, kernel_size = (3,3), 
#                  activation='relu'))
# main_model.add(SeparableConv2D(filters = 512, kernel_size = (3,3), 
#                  activation='relu'))
# main_model.add(MaxPool2D(pool_size=(2, 2)))
# main_model.add(Dropout(0.25))

#main_model.add(Flatten())
#main_model.add(Dense(units = 5, activation = 'relu'))
#main_model.add(Dropout(0.2))
#FC => Output
#main_model.add(Dense(2, activation='softmax'))

#main_model.summary()

#Dense Net Architecture
# from keras.models import Model
# base_model = DenseNet121(include_top=False,weights='imagenet',input_shape = (96,96,3))

# x = base_model.output
# x = Flatten()(x)
# x = Dense(150,activation='relu')(x)
# x = Dropout(0.2)(x)
# predictions = Dense(2,activation='softmax')(x)

# main_model = Model(base_model.input,predictions)

#from keras.models import load_model
#main_model = load_model('check_5_epochs_conv_aug.h5')
# main_model.summary()
# In[63]:


from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from keras.optimizers import Adam,RMSprop


# In[64]:


csv_logger = CSVLogger("results/capsnet/result.csv",separator = ",",append=True)

checkpoint_fp = "results/capsnet/weights.{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(checkpoint_fp,monitor='val_out_caps_acc',
                             verbose=1,
                            save_best_only= True,mode='max')


# In[72]:


learning_rate = ReduceLROnPlateau(monitor='val_out_caps_acc',
                                 factor = 0.1,
                                 patience = 2,
                                 verbose = 1,
                                 mode = 'max',
                                 min_lr = 0.00001)


# In[73]:


callback = [checkpoint,learning_rate,csv_logger]


# In[77]:


steps_p_ep_tr =np.ceil(len(train)/batch_size)
steps_p_ep_va =np.ceil(len(valid)/batch_size)


# In[78]:


main_model.compile(optimizer = Adam(lr=0.0001), 
              loss = 'binary_crossentropy', metrics=['accuracy'])


# In[79]:


my_model = main_model.fit_generator(train_gen,
                                   steps_per_epoch = steps_p_ep_tr,
                                   validation_data = patches('valid'),
                                   validation_steps = steps_p_ep_va,
                                   verbose = 1,
                                   epochs = epochs,
                                   callbacks = callback)

print("---------------------------------------------------------------")
print("model keys: ",my_model.history.keys())
print("---------------------------------------------------------------")
plt.plot(my_model.history['acc'])
plt.plot(my_model.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train','valid'],loc='upper left')
plt.savefig('CapsuleNet_patches.png')
