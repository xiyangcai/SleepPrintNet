import numpy as np
import os
import model as SleepPrintNet

import time
import argparse
import keras
from keras import callbacks
from keras import metrics
from Utils.two_stream_dataloader import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import multi_gpu_model

gpunums = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # allocate dynamically
config.gpu_options.per_process_gpu_memory_fraction = 0.9
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

# Global vars
fold_string = None
best_acc = 0
model = None
X_test_6 = None
y_test_6 = None
X_test_16 = None
y_test_16 = None
X_test_psd = None
X_test_eog = None
y_test_psd = None
file_dir = None
filename = None
model_dir = None
real_fold = 1
n_oversampling = 0
over_cm = []


class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_epoch_end(self, epoch, logs={}):

        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        self.draw_p2in1(
            self.losses['epoch'], self.val_loss['epoch'], 'loss', 'train_epoch', 'val_epoch')
        self.draw_p2in1(
            self.accuracy['epoch'], self.val_acc['epoch'], 'acc', 'train_epoch', 'val_epoch')
        global best_acc, over_cm

        # Output best
        if best_acc < max(self.val_acc['epoch']):
            best_acc = max(self.val_acc['epoch'])
            predict_test = np.argmax(model.predict(
                [X_test_psd, X_test_16, X_test_6, X_test_eog]), axis=1)
            cm = confusion_matrix(y_test_6, predict_test,
                                  labels=[0, 1, 2, 3, 4])
            print(cm)
            over_cm[-1] = cm
            np.savetxt(filename+'.txt', cm, "%d")
            f = open(filename + '_best_acc.txt', "w")
            print(best_acc, file=f)
            f.close()
        print("acc", best_acc)

    def draw_p2in1(self, lists1, lists2, label, type1, type2):
        plt.figure()
        plt.plot(range(len(lists1)), lists1, 'r', label=type1)
        plt.plot(range(len(lists2)), lists2, 'b', label=type2)
        plt.ylabel(label)
        plt.xlabel(type1.split('_')[0]+'_'+type2.split('_')[0])
        plt.legend(loc="upper right")
        global filename
        filename = file_dir+label+'_fold'+fold_string
        plt.savefig(filename+'.jpg')
        plt.close()

    def draw_p(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(filename+'.jpg')
        plt.close()

    def end_draw(self):
        self.draw_p2in1(
            self.losses['epoch'], self.val_loss['epoch'], 'loss', 'train_epoch', 'val_epoch')
        self.draw_p2in1(
            self.accuracy['epoch'], self.val_acc['epoch'], 'acc', 'train_epoch', 'val_epoch')


def train(args):
    global fold_string, best_acc, model, file_dir, over_cm, X_test_16, X_test_6, y_test_16, y_test_6, X_test_psd, X_test_eog, y_test_psd

    classes = [0, 1, 2, 3, 4]
    num_classes = len(classes)
    num_folds = args.num_fold
    data_dir = args.data_dir1
    data_dir2 = args.data_dir2
    data_dir3 = args.data_dir3
    n_files = args.n_files
    model_dir = args.model_dir
    file_dir = args.result_dir

    seq_len = args.seqLen
    width = args.height
    height = args.width
    save_model = True if args.save_model else False

    all_time = 0
    acc = np.zeros(real_fold)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    for fold_idx in range(real_fold):
        over_cm.append([])
        best_acc = 0
        fold_string = str(fold_idx)
        start_time_fold_i = time.time()
        logs_loss = LossHistory()
        print('train start time of fold{} is {}'.format(
            fold_idx, start_time_fold_i))

        # Reading Data
        data_loader_16 = SeqDataLoader(
            data_dir, num_folds, fold_idx, classes, n_files)
        X_train_16, y_train_16, X_test_16, y_test_16 = data_loader_16.load_data()

        X_train_eog = X_train_16[:, 6:, :]
        X_train_16 = X_train_16[:, :6, :]

        X_test_eog = X_test_16[:, 6:, :]
        X_test_16 = X_test_16[:, :6, :]

        data_loader_6 = SeqDataLoader(
            data_dir2, num_folds, fold_idx, classes, n_files)
        X_train_6, y_train_6, X_test_6, y_test_6 = data_loader_6.load_data()

        data_loader_psd = SeqDataLoader(
            data_dir3, num_folds, fold_idx, classes, n_files)
        X_train_psd, y_train_psd, X_test_psd, y_test_psd = data_loader_psd.load_data()

        model_name = "model_fold{:02d}_in{:02d}of{:02d}.h5".format(
            fold_idx, num_folds, n_files)
        model = SleepPrintNet.create_SleepPrintNet(
            num_classes, seq_len, width, height, psd_filter_nums=args.num_filters, times=11520, Fs=128)

        if gpunums > 1:
            parallel_model = multi_gpu_model(model, gpus=gpunums)
            adam = keras.optimizers.Adam(
                lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            parallel_model.compile(
                optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['acc'])
        else:
            adam = keras.optimizers.Adam(
                lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            model.compile(
                optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['acc'])

        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20
            ),
            logs_loss
        ]

        if save_model:
            callbacks_list.append(callbacks.ModelCheckpoint(
                filepath=model_dir + model_name,
                monitor='val_acc',
                save_best_only=True,
            ))

        if gpunums > 1:
            parallel_model.fit([X_train_psd, X_train_16, X_train_6, X_train_eog], y_train_6, validation_data=(
                [X_test_psd, X_test_16, X_test_6, X_test_eog], y_test_6), epochs=args.epoch, batch_size=args.batch_size, callbacks=callbacks_list, verbose=2, shuffle=True)
        else:
            model.fit([X_train_psd, X_train_16, X_train_6, X_train_eog], y_train_6, validation_data=([X_test_psd, X_test_16, X_test_6,
                                                                                                      X_test_eog], y_test_6), epochs=args.epoch, batch_size=args.batch_size, callbacks=callbacks_list, verbose=2, shuffle=True)

        del X_train_16, y_train_16, X_test_16, y_test_16, X_train_6, y_train_6, X_test_6, y_test_6, model, data_loader_16, data_loader_6, X_train_psd, y_train_psd, X_test_psd, y_test_psd, X_train_eog, X_test_eog, data_loader_psd

        end_time_fold_i = time.time()
        train_time_fold_i = end_time_fold_i - start_time_fold_i
        all_time += train_time_fold_i
        logs_loss.end_draw()
        acc[fold_idx] = max(logs_loss.val_acc['epoch'])
        print('train time of fold{} is {}'.format(fold_idx, train_time_fold_i))

    for index in range(1, len(over_cm)):
        over_cm[0] += over_cm[index]
    print('train_time:', all_time)
    print("over_cm:")
    print(over_cm[0])
    np.savetxt(file_dir+"over_cm.txt", acc)


def main():
    parser = argparse.ArgumentParser(
        description='SleepPrintNet - MASS-SS3 - K fold')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, metavar='N',
                        help='epoch (default: 100)')
    parser.add_argument('--num_fold', type=int, default=31, metavar='N',
                        help='fold num (default:31)')

    parser.add_argument('--seqLen', type=int, default=5, metavar='N',
                        help='Seq length (default: 5)')
    parser.add_argument('--height', type=int, default=16, metavar='N',
                        help='Height of 2D Map (default: 16)')
    parser.add_argument('--width', type=int, default=16, metavar='N',
                        help='Width of 2D Map (default: 16)')
    parser.add_argument('--num_filters', type=int, default=16, metavar='N',
                        help='num_filters (default: 16)')
    parser.add_argument('--save_model', type=int, default=1, metavar='N',
                        help='save_model (default: 0)')

    parser.add_argument('--model_dir', type=str, default='./output_model/', metavar='N',
                        help='output dir (default: ./output_model/)')
    parser.add_argument('--data_dir1', type=str, default='./EEG_EOG', metavar='N',
                        help='data_dir1 (default: ./EEG_EOG)')
    parser.add_argument('--data_dir2', type=str, default='./EMG', metavar='N',
                        help='data_dir2 (default: ./EMG)')
    parser.add_argument('--data_dir3', type=str, default='./fre_spa', metavar='N',
                        help='data_dir3 (default:./fre_spa)')
    parser.add_argument('--n_files', type=int, default=62, metavar='N',
                        help='n_files (default: 62)')
    parser.add_argument('--result_dir', type=str, default='./result/', metavar='N',
                        help='result_dir (default: ./result/)')

    args = parser.parse_args()

    print("SleepPrintNet")
    train(args)


if __name__ == "__main__":
    main()
