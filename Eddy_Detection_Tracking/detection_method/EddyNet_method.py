from IPython.core.interactiveshell import InteractiveShell
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Activation, Reshape
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Conv2DTranspose
from keras.layers import BatchNormalization, add
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from keras.utils.training_utils import multi_gpu_model
InteractiveShell.ast_node_interactivity = 'all'

SSH_train = np.load('./eddy_data/filtered_SSH_train_data.npy')[:,10:168+10,302-200:]
SSH_test = np.load('./eddy_data/filtered_SSH_test_data.npy')[:,10:168+10,302-200:]


Seg_train = np.load('./eddy_data/train_groundtruth_Segmentation.npy')[:,10:168+10,302-200:]
Seg_test = np.load('./eddy_data/test_groundtruth_Segmentation.npy')[:,10:168+10,302-200:]

randindex = np.random.randint(0,len(SSH_train))
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.imshow(SSH_train[randindex,:-4,:-10],cmap='viridis')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
#plt.clim(-0.25,0.25)
plt.axis('off')
plt.title('SSH', fontsize=24)

plt.subplot(122)
plt.imshow(Seg_train[randindex,:-4,:-10],cmap='viridis')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
plt.axis('off')
plt.title('groundtruth Segmentation', fontsize=24)
plt.savefig('./eddynet/heatmap.png')
#plt.show()

Seg_train_categor = np_utils.to_categorical(np.reshape(Seg_train[:,:,0],(4750, 168*200)), 3) # our own data
#print(Seg_train_categor.shape)

def ConvBNActi(nf, ker, inputs):
    conv1 = Conv2D(nf, ker, padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    return conv1
def ConvTransBNActi(nf, ker, inputs):
    conv1 = Conv2DTranspose(nf, ker, strides=(2, 2), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    return conv1


height = 168
width = 200
nf = 128
nbClass = 3
ker = 3


############INPUT LAYER############

img_input = Input(shape=(height, width, 1))

############ENCODER################

conv1 = ConvBNActi(nf, ker, img_input)
conv1 = ConvBNActi(nf, ker, conv1)
conv1 = Dropout(0.2)(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = ConvBNActi(nf, ker, pool1)
conv2 = ConvBNActi(nf, ker, conv2)
conv2 = Dropout(0.3)(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = ConvBNActi(nf, ker, pool2)
conv3 = ConvBNActi(nf, ker, conv3)
conv3 = Dropout(0.4)(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

############CENTER#################

convC = ConvBNActi(nf, ker, pool3)
convC = ConvBNActi(nf, ker, convC)
convC = Dropout(0.5)(convC)

############DECODER################

up3 = concatenate([UpSampling2D((2, 2))(convC), conv3])
decod3 = ConvBNActi(nf, ker, up3)
decod3 = ConvBNActi(nf, ker, decod3)
decod3 = Dropout(0.4)(decod3)

up2 = concatenate([UpSampling2D((2, 2))(decod3), conv2])
decod2 = ConvBNActi(nf, ker, up2)
decod2 = ConvBNActi(nf, ker, decod2)
decod2 = Dropout(0.3)(decod2)

up1 = concatenate([UpSampling2D((2, 2))(decod2), conv1])
decod1 = ConvBNActi(nf, ker, up1)
decod1 = ConvBNActi(nf, ker, decod1)
decod1 = Dropout(0.2)(decod1)

##########SEGMENTATION#############

x = Conv2D(nbClass, (1, 1), padding='valid', use_bias=False)(decod1)
x = Reshape((height * width, nbClass))(x)
x = Activation('softmax')(x)
eddynet = Model(img_input, x)
# eddynet = multi_gpu_model(eddynet, gpus=2)
eddynet.summary()


unipue, counts = np.unique(Seg_train, return_counts=True)
dict(zip(unipue, counts))

freq = [np.sum(counts)/j for j in counts]
weightsSeg = [f/np.sum(freq) for f in freq]

###loss function

smooth = 1.    #to avoid zero division

def dice_coef_anti(y_true, y_pred):
    y_true_anti = y_true[:,:,1]
    y_pred_anti = y_pred[:,:,1]
    intersection_anti = K.sum(y_true_anti * y_pred_anti)
    return (2 * intersection_anti + smooth) / (K.sum(y_true_anti) + K.sum(y_pred_anti) + smooth)

def dice_coef_cyc(y_true, y_pred):
    y_true_cyc = y_true[:,:,2]
    y_pred_cyc = y_pred[:,:,2]
    intersection_cyc = K.sum(y_true_cyc * y_pred_cyc)
    return (2 * intersection_cyc + smooth) / (K.sum(y_true_cyc) + K.sum(y_pred_cyc) + smooth)

def dice_coef_nn(y_true, y_pred):
    y_true_nn = y_true[:,:,0]
    y_pred_nn = y_pred[:,:,0]
    intersection_nn = K.sum(y_true_nn * y_pred_nn)
    return (2 * intersection_nn + smooth) / (K.sum(y_true_nn) + K.sum(y_pred_nn) + smooth)

def mean_dice_coef(y_true, y_pred):
    return (dice_coef_anti(y_true, y_pred) + dice_coef_cyc(y_true, y_pred) + dice_coef_nn(y_true, y_pred))/3

def weighted_mean_dice_coef(y_true, y_pred):
    return (0.35 * dice_coef_anti(y_true, y_pred) + 0.62 * dice_coef_cyc(y_true,y_pred) + 0.03 * dice_coef_nn(y_true, y_pred))

def dice_coef_loss(y_true, y_pred):
    return 1 - weighted_mean_dice_coef(y_true, y_pred)

def precision(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positive / (predicted_positive + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positive = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positive / (possible_positive + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta = 1):
    if beta < 0:
        raise ValueError('the lowest choosable beta is zero')
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    b = beta ** 2
    fbeta_score = (1 + b) * (p * r) / (b * p + r + K.epsilon())
    return fbeta_score

#def fmeasure(y_true, y_pred):
#    return fbeta_score(y_true, y_pred, beta=1)

eddynet.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=['categorical_accuracy', mean_dice_coef,
                                                                       dice_coef_anti, dice_coef_cyc, dice_coef_nn,
                                                                       weighted_mean_dice_coef, precision, recall, fbeta_score])

#earl = EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=80, verbose=1, mode='auto')
modelcheck = ModelCheckpoint('./eddynet/eddynet.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
reducecall = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1, mode='auto', min_delta=1e-30, min_lr=1e-30)

hiseddynet = eddynet.fit(SSH_train, Seg_train_categor,
                         batch_size=4,
                         epochs=300,
                         verbose=1,
                         callbacks=[modelcheck, reducecall],
                         validation_split=0.2,
                         shuffle=True)

plt.figure(figsize=(10, 10))
plt.semilogy(eddynet.history.history['loss'])
plt.semilogy(eddynet.history.history['val_loss'])
plt.semilogy(eddynet.history.history['val_categorical_accuracy'])
plt.title('EddyNet Loss and Val_acc', fontsize=20)
plt.ylabel('loss/accuracy')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss', 'val_acc'], loc='center right')
plt.savefig('./eddynet/EddyNetLoss.png')
#plt.show()

#performance on train dataset

randindex = np.random.randint(0, len(SSH_train))
predictedSEGM = eddynet.predict(np.reshape(SSH_train[randindex,:,:], (1, height, width, 1)))
predictedSEGMimage = np.reshape(predictedSEGM.argmax(2), (height, width))

plt.figure(figsize=(20, 10))
plt.subplot(131)
plt.imshow(SSH_train[randindex,:-4,:-10], cmap='viridis')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
#plt.clim(-0.25,0.25)
plt.axis('off')
plt.title('SSH', fontsize=24)

plt.subplot(132)
plt.imshow(predictedSEGMimage[:-4,:-10], cmap='viridis')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
#plt.clim(-0.25,0.25)
plt.axis('off')
plt.title('EddyNet Method', fontsize=24)

plt.subplot(133)
plt.imshow(Seg_train[randindex,:-4,:-10], cmap='viridis')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
#plt.clim(-0.25,0.25)
plt.axis('off')
plt.title('PET Method', fontsize=24)
plt.savefig('./eddynet/traindata.png')
#plt.show()
#performance on test dataset

randindex = np.random.randint(0, len(SSH_test))
predictedSEGM = eddynet.predict(np.reshape(SSH_test[randindex,:,:], (1, height, width, 1)))
predictedSEGMimage = np.reshape(predictedSEGM.argmax(2), (height, width))

plt.figure(figsize=(20, 10))

plt.subplot(131)
plt.imshow(SSH_test[randindex,:-4,:-10], cmap='viridis')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
#plt.clim(-0.25,0.25)
plt.axis('off')
plt.title('SSH', fontsize=24)

plt.subplot(132)
plt.imshow(predictedSEGMimage[:-4,:-10], cmap='viridis')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
#plt.clim(-0.25,0.25)
plt.axis('off')
plt.title('EddyNet Method', fontsize=24)

plt.subplot(133)
plt.imshow(Seg_test[randindex,:-4,:-10], cmap='viridis')
plt.colorbar(extend='both', fraction=0.042, pad=0.04)
#plt.clim(-0.25,0.25)
plt.axis('off')
plt.title('PET Method', fontsize=24)
plt.savefig('./eddynet/testdata.png')

#plt.show()
#metrics on test dataset


Seg_test_categor = np_utils.to_categorical(np.reshape(Seg_test[:,:,0], (730, 168*200)), 3) # our own data
#print(Seg_test_categor.shape)
preds = eddynet.evaluate(SSH_test, Seg_test_categor)
print('loss: %s,'%preds[0], 'accuracy: %s,'%preds[1], 'mean_dice_coef: %s,'%preds[2], 'weighted_mean_dice_coef: %s,'%preds[3],
      'MD cyc: %s'%preds[4], 'MD NE: %s'%preds[5], 'MD anti: %s'%preds[6],
      'precision: %s,'%preds[7], 'recall: %s,'%preds[8], 'f1_score: %s'%preds[9])

