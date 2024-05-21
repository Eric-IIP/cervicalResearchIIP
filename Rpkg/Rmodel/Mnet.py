# from keras.models import Model
# from keras.layers import Input
# from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, average
# from keras.layers.normalization import BatchNormalization
# from keras.layers import concatenate

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, average
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate

def build_network(IMAGE_SIZE, input_channel=3, output_channel=1):
    FILTERS = 32
    # FILTERS = 64
    input_img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, input_channel))

    ms1 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_img)
    # ms1 = BatchNormalization()(ms1)
    ms2 = AveragePooling2D(pool_size=(2, 2), strides=2)(ms1)
    ms2 = Conv2D(FILTERS*1, kernel_size=3, strides=1, activation="relu", padding="same")(ms2)
    ms3 = AveragePooling2D(pool_size=(2, 2), strides=2)(ms2)
    ms3 = Conv2D(FILTERS*1, kernel_size=3, strides=1, activation="relu", padding="same")(ms3)
    # ms3 = BatchNormalization()(ms3)

    enc1 = Conv2D(FILTERS*1, kernel_size=3, strides=1, activation="relu", padding="same")(input_img)
    enc1 = BatchNormalization()(enc1)
    enc1 = Conv2D(FILTERS*1, kernel_size=3, strides=1, activation="relu", padding="same")(enc1)
    enc1 = BatchNormalization()(enc1)
    down1 = MaxPooling2D(pool_size=2, strides=2)(enc1)
    
    ms1 =  Conv2D(FILTERS*2, kernel_size=3, strides=1, activation="relu", padding="same")(ms1)
    enc2 = concatenate([ms1, down1], axis=-1) #
    enc2 = Conv2D(FILTERS*2+FILTERS*1, kernel_size=3, strides=1, activation="relu", padding="same")(enc2)
    enc2 = BatchNormalization()(enc2)
    enc2 = Conv2D(FILTERS*2, kernel_size=3, strides=1, activation="relu", padding="same")(enc2)
    enc2 = BatchNormalization()(enc2)
    down2 = MaxPooling2D(pool_size=2, strides=2)(enc2)

    ms2 =  Conv2D(FILTERS*4, kernel_size=3, strides=1, activation="relu", padding="same")(ms2)
    enc3 = concatenate([ms2, down2], axis=-1) #
    enc3 = Conv2D(FILTERS*4+FILTERS*2, kernel_size=3, strides=1, activation="relu", padding="same")(enc3)
    enc3 = BatchNormalization()(enc3)
    enc3 = Conv2D(FILTERS*4, kernel_size=3, strides=1, activation="relu", padding="same")(enc3)
    enc3 = BatchNormalization()(enc3)
    down3 = MaxPooling2D(pool_size=2, strides=2)(enc3)
    
    ms3 =  Conv2D(FILTERS*8, kernel_size=3, strides=1, activation="relu", padding="same")(ms3)
    enc4 = concatenate([ms3, down3], axis=-1) #
    enc4 = Conv2D(FILTERS*8+FILTERS*4, kernel_size=3, strides=1, activation="relu", padding="same")(enc4)
    enc4 = BatchNormalization()(enc4)
    enc4 = Conv2D(FILTERS*8, kernel_size=3, strides=1, activation="relu", padding="same")(enc4)
    enc4 = BatchNormalization()(enc4)
    down4 = MaxPooling2D(pool_size=2, strides=2)(enc4)
    
    enc5 = Conv2D(FILTERS*16, kernel_size=3, strides=1, activation="relu", padding="same")(down4)
    enc5 = BatchNormalization()(enc5)
    enc5 = Conv2D(FILTERS*16, kernel_size=3, strides=1, activation="relu", padding="same")(enc5)
    enc5 = BatchNormalization()(enc5)

    up4 = UpSampling2D(size=2)(enc5)
    # up4 = Conv2DTranspose(FILTERS*8, kernel_size=3, strides=2, activation="relu", padding="same")(enc5)
    dec4 = concatenate([up4, enc4], axis=-1)
    dec4 = Conv2D(FILTERS*8, kernel_size=3, strides=1, activation="relu", padding="same")(dec4)
    dec4 = BatchNormalization()(dec4)
    dec4 = Conv2D(FILTERS*8, kernel_size=3, strides=1, activation="relu", padding="same")(dec4)
    dec4 = BatchNormalization()(dec4)
    
    up3 = UpSampling2D(size=2)(dec4)
    # up3 = Conv2DTranspose(FILTERS*4, kernel_size=3, strides=2, activation="relu", padding="same")(dec4)
    dec3 = concatenate([up3, enc3], axis=-1)
    dec3 = Conv2D(FILTERS*4, kernel_size=3, strides=1, activation="relu", padding="same")(dec3)
    dec3 = BatchNormalization()(dec3)
    dec3 = Conv2D(FILTERS*4, kernel_size=3, strides=1, activation="relu", padding="same")(dec3)
    dec3 = BatchNormalization()(dec3)

    up2 = UpSampling2D(size=2)(dec3)
    # up2 = Conv2DTranspose(FILTERS*2, kernel_size=3, strides=2, activation="relu", padding="same")(dec3)
    dec2 = concatenate([up2, enc2], axis=-1)
    dec2 = Conv2D(FILTERS*2, kernel_size=3, strides=1, activation="relu", padding="same")(dec2)
    dec2 = BatchNormalization()(dec2)
    dec2 = Conv2D(FILTERS*2, kernel_size=3, strides=1, activation="relu", padding="same")(dec2)
    dec2 = BatchNormalization()(dec2)
    
    up1 = UpSampling2D(size=2)(dec2)
    # up1 = Conv2DTranspose(FILTERS*1, kernel_size=3, strides=2, activation="relu", padding="same")(dec2)
    dec1 = concatenate([up1, enc1], axis=-1)
    dec1 = Conv2D(FILTERS*1, kernel_size=3, strides=1, activation="relu", padding="same")(dec1)
    dec1 = BatchNormalization()(dec1)
    dec1 = Conv2D(FILTERS*1, kernel_size=3, strides=1, activation="relu", padding="same")(dec1)
    dec1 = BatchNormalization()(dec1)
    
    dec1 = Conv2D(output_channel, kernel_size=1, strides=1, activation="sigmoid", padding="same")(dec1)

    out1 = dec1
    out2 = UpSampling2D(size=2)(dec2)
    out2 = Conv2D(output_channel, kernel_size=1, strides=1, activation="sigmoid", padding="same")(out2)
    out3 = UpSampling2D(size=4)(dec3)
    out3 = Conv2D(output_channel, kernel_size=1, strides=1, activation="sigmoid", padding="same")(out3)
    out4 = UpSampling2D(size=8)(dec4)
    out4 = Conv2D(output_channel, kernel_size=1, strides=1, activation="sigmoid", padding="same")(out4)
    avg = average([out1, out2, out3, out4])
    
    model = Model(input_img, avg)
    
    return model