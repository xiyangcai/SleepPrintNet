from keras import backend as K
from keras.regularizers import l2
import keras
import logging
logger = logging.getLogger(__name__)
K.set_image_data_format("channels_last")


def se_slice_psd(x):
    return x[:, :, :, :, 0]


def deepsleepnet(intput, Fs, time_filters_nums, bn_mom, version):
    ######### CNNs with small filter size at the first layer #########
    y1 = keras.layers.Conv1D(name='conv1_small{}'.format(version), kernel_size=Fs//2, strides=Fs//16, filters=time_filters_nums, padding='same',
                             use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(intput)
    y1 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y1)
    y1 = keras.layers.LeakyReLU()(y1)

    y1 = keras.layers.MaxPooling1D(pool_size=8, strides=8, padding='same')(y1)
    y1 = keras.layers.Dropout(0.5)(y1)

    y1 = keras.layers.Conv1D(name='conv2_small{}'.format(version), kernel_size=8, strides=1, filters=time_filters_nums*2, padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(y1)
    y1 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y1)
    y1 = keras.layers.LeakyReLU()(y1)

    y1 = keras.layers.Conv1D(name='conv3_small{}'.format(version), kernel_size=8, strides=1, filters=time_filters_nums*2, padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(y1)
    y1 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y1)
    y1 = keras.layers.LeakyReLU()(y1)

    y1 = keras.layers.Conv1D(name='conv4_small{}'.format(version), kernel_size=8, strides=1, filters=time_filters_nums*2, padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(y1)
    y1 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y1)
    y1 = keras.layers.LeakyReLU()(y1)
    y1 = keras.layers.MaxPooling1D(pool_size=4, strides=4, padding='same')(y1)
    y1 = keras.layers.Flatten()(y1)

    ######### CNNs with big filter size at the first layer #########
    y2 = keras.layers.Conv1D(name='conv1_big{}'.format(version), kernel_size=Fs*4, strides=Fs//2, filters=time_filters_nums, padding='same',
                             use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(intput)
    y2 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y2)
    y2 = keras.layers.LeakyReLU()(y2)

    y2 = keras.layers.MaxPooling1D(pool_size=4, strides=4, padding='same')(y2)
    y2 = keras.layers.Dropout(0.5)(y2)

    y2 = keras.layers.Conv1D(name='conv2_big{}'.format(version), kernel_size=6, strides=1, filters=time_filters_nums*2, padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(y2)
    y2 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y2)
    y2 = keras.layers.LeakyReLU()(y2)

    y2 = keras.layers.Conv1D(name='conv3_big{}'.format(version), kernel_size=6, strides=1, filters=time_filters_nums*2, padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(y2)
    y2 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y2)
    y2 = keras.layers.LeakyReLU()(y2)

    y2 = keras.layers.Conv1D(name='conv4_big{}'.format(version), kernel_size=6, strides=1, filters=time_filters_nums*2, padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(y2)
    y2 = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                         beta_initializer='zeros', gamma_initializer='ones',
                                         moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                         beta_regularizer=None, gamma_regularizer=None,
                                         beta_constraint=None, gamma_constraint=None)(y2)
    y2 = keras.layers.LeakyReLU()(y2)

    y2 = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(y2)
    y2 = keras.layers.Flatten()(y2)
    y = keras.layers.concatenate([y1, y2], axis=-1)
    return y


def create_SleepPrintNet(
    num_class,
    seq_len=100,
    width=16,
    height=16,
    use_bias=True,
    bn_mom=0.9,
    times=7680,
    Fs=128,
    time_filters_nums=64,
    psd_filter_nums=32
):

    # Begin Layers
    # (Samples,5,16,16,1)
    input_layer = keras.layers.Input(
        name='input_layer_psd', shape=(seq_len, width, height, 1))
    input_psd = keras.layers.Lambda(
        se_slice_psd)(input_layer)
    input_psd = keras.layers.core.Permute((2, 3, 1))(input_psd)

    # Residual 16*16
    x_psd = keras.layers.Conv2D(name='conv1_middle_psd', kernel_size=(1, 1), strides=(1, 1), filters=psd_filter_nums, padding='same',
                                use_bias=use_bias, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input_psd)
    x_psd = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                            beta_initializer='zeros', gamma_initializer='ones',
                                            moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                            beta_regularizer=None, gamma_regularizer=None,
                                            beta_constraint=None, gamma_constraint=None)(x_psd)
    x_psd = keras.layers.ReLU()(x_psd)

    x = keras.layers.Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=psd_filter_nums*2, padding='same',
                            use_bias=True, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_psd)
    x = keras.layers.BatchNormalization(axis=-1, momentum=bn_mom, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                        beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None)(x)

    x_psd = keras.layers.Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=psd_filter_nums*2, padding='same',
                                use_bias=True, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_psd)
    x = keras.layers.add([x_psd, x])
    x = keras.layers.MaxPooling2D(pool_size=(
        4, 4), strides=(2, 2), padding='valid')(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)

    # DeepSleepNet for EEG, EMG, and EOG
    input_layer_time = keras.layers.Input(
        name='input_layer_time', shape=(6, times))
    layer_time = keras.layers.Reshape((times, 6))(input_layer_time)
    y1 = deepsleepnet(layer_time, Fs, time_filters_nums, bn_mom, '1')

    input_layer_emg = keras.layers.Input(
        name='input_layer_emg', shape=(3, times))
    layer_emg = keras.layers.Reshape((times, 3))(input_layer_emg)
    y2 = deepsleepnet(layer_emg, Fs, time_filters_nums//2, bn_mom, '2')

    input_layer_eog = keras.layers.Input(
        name='input_layer_time_eog', shape=(2, times))
    layer_eog = keras.layers.Reshape((times, 2))(input_layer_eog)
    y3 = deepsleepnet(layer_eog, Fs, time_filters_nums//2, bn_mom, '3')

    y = keras.layers.concatenate([x, y1, y2, y3], axis=-1)

    y = keras.layers.Dense(128, activation='relu',
                           kernel_regularizer=l2(0.1))(y)
    y = keras.layers.Dense(num_class, activation='softmax',
                           kernel_regularizer=l2(0.1))(y)
    model = keras.models.Model(
        [input_layer, input_layer_time, input_layer_emg, input_layer_eog], y)
    return model
