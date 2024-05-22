import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.optimizers import Adam

def set_seed(seed=7):
    """
    乱数シードseedで乱数固定
    """
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# ネットワークの構築
def build_network(IMAGE_SIZE,input_channel=3, output_channel=1, seed=7):
    initializer=GlorotUniform(seed=seed)
    input = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,input_channel))

    # enc1
    conv1 = Conv2D(filters=64, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(input)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(conv1)
    conv2 = BatchNormalization()(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2),padding='same')(conv2)

    # enc2
    conv3 = Conv2D(filters=128, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(pool1)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(filters=128, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(conv3)
    conv4 = BatchNormalization()(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2),padding='same')(conv4)

    # enc3
    conv5 = Conv2D(filters=256, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(pool2)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(filters=256, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(conv5)
    conv6 = BatchNormalization()(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2),padding='same')(conv6)

    # enc4
    conv7 = Conv2D(filters=512, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(pool3)
    conv7 = BatchNormalization()(conv7)
    conv8 = Conv2D(filters=512, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(conv7)
    conv8 = BatchNormalization()(conv8)
    pool4 = MaxPooling2D(pool_size=(2, 2),padding='same')(conv8)

    # enc5
    conv9 = Conv2D(filters=1024, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(pool4)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(filters=1024, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(conv9)
    conv10 = BatchNormalization()(conv10)

    # dec1
    upconv1 = UpSampling2D(size=(2,2))(conv10)
    merge1 = concatenate([conv8,upconv1],axis=-1)
    #merge1 = Dropout(0.5)(merge1)
    conv11 = Conv2D(filters=512, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(merge1)
    conv11 = BatchNormalization()(conv11)
    conv12 = Conv2D(filters=512, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(conv11)
    conv12 = BatchNormalization()(conv12)

    # dec2
    upconv2 = UpSampling2D(size=(2, 2))(conv12)
    merge2 = concatenate([conv6,upconv2],axis=-1)
    #merge2 = Dropout(0.5)(merge2)
    conv13 = Conv2D(filters=256, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(merge2)
    conv13 = BatchNormalization()(conv13)
    conv14 = Conv2D(filters=256, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(conv13)
    conv14 = BatchNormalization()(conv14)

    # dec3
    upconv3 = UpSampling2D(size=(2, 2))(conv14)
    merge3 = concatenate([conv4,upconv3],axis=-1)
    #merge3 = Dropout(0.5)(merge3)
    conv15 = Conv2D(filters=128, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(merge3)
    conv15 = BatchNormalization()(conv15)
    conv16 = Conv2D(filters=128, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(conv15)
    conv16 = BatchNormalization()(conv16)

    # dec4
    upconv4 = UpSampling2D(size=(2, 2))(conv16)
    merge4 = concatenate([conv2,upconv4],axis=-1)
    #merge4 = Dropout(0.5)(merge4)
    conv17 = Conv2D(filters=64, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(merge4)
    conv17 = BatchNormalization()(conv17)
    conv18 = Conv2D(filters=64, kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=initializer)(conv17)
    conv18 = BatchNormalization()(conv18)

    conv19 = Conv2D(filters=output_channel, kernel_size=(1,1),padding='same',activation='sigmoid',kernel_initializer=initializer)(conv18)

    model = Model(input,conv19)

    adam_optimizer = Adam(lr=0.0001)
    model.compile(
    loss='binary_crossentropy',
    optimizer=adam_optimizer,
    metrics=['accuracy']
    )   # 損失関数と最適化関数の設定

    return model