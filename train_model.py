from keras import backend as K
from keras.layers import Conv2D, MaxPool2D, Input, Dense, Lambda, Activation,BatchNormalization,LSTM, Bidirectional
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam
from data_loader import num_class, maxlen
import numpy as np
from data_loader import x_train, y_train, x_train_length, y_train_length, x_val, y_val, x_val_length, y_val_length
#import matplotlib.pyplot as plt

def ctc_lambda_func(args):
    '''
    
    '''
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ocr_model(batch_size=2, epoch=5, rnn_size=128, img_row=32,img_col=128):
    '''
    if you use the full datset you could increase the batch_size and epo
    '''
    inputs = Input(shape=(img_row, img_col, 1))
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
    conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 1))(conv_2)
    conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)
    conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
    pool_3 = MaxPool2D(pool_size=(2, 1))(conv_4)
    conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_3)
    batch_norm_1 = BatchNormalization()(conv_5)
    conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_1)
    batch_norm_2 = BatchNormalization()(conv_6)
    pool_4 = MaxPool2D(pool_size=(2, 1))(batch_norm_2)
    conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_4)
    # the output here, called the time step, should be at least greater than the maximum input length of the GT.
    reshape = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
    #[ sample, timesteps, features]
    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(rnn_size, return_sequences=True, dropout=0.25))(reshape)
    blstm_2 = Bidirectional(LSTM(rnn_size, return_sequences=True, dropout=0.25))(blstm_1)

    outputs = Dense(num_class + 1, activation='softmax')(blstm_2)

    pred_model = Model(inputs, outputs)

    labels = Input(name='the_labels', shape=[maxlen], dtype='float32')# 32 is the max size of text length
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    checkpoint = ModelCheckpoint('model_30{epoch:01d}.hdf5', period=1)

    hist=model.fit(x=[x_train, y_train, x_train_length, y_train_length], y=np.zeros(len(y_train)),
          batch_size=batch_size, epochs =epoch,validation_data = ([x_val, y_val, x_val_length, y_val_length], [np.zeros(len(y_val))]),
          verbose = 1, callbacks = [checkpoint])
   
    return pred_model
if __name__=="__main__":
     
  model=ocr_model()

  model.save('model_test.hdf5')
  print("Training is scussfully completed and now your model is stored to your disk"



