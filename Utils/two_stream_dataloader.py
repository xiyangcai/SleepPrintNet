import os
import numpy as np


class SeqDataLoader():
    def __init__(self, data_dir, n_folds, fold_idx, classes, n_files):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.classes = classes
        self.n_files = n_files

    def _load_npy_list_files(self, data_files, label_files):
        data = []
        labels = []
        for data_name, label_name in zip(data_files, label_files):
            #print ("Loading {} {} ...".format(data_name,label_name))
            tmp_data = np.load(data_name)
            tmp_labels = np.load(label_name)
            tmp_labels = tmp_labels.astype(int)
            data.append(tmp_data)
            labels.append(tmp_labels)
        return data, labels

    def print_n_samples_each_class(self, labels, classes):
        class_dict = dict(zip(range(len(classes)), classes))
        unique_labels = np.unique(labels)
        for c in unique_labels:
            n_samples = len(np.where(labels == c)[0])
            print("{}: {}".format(class_dict[c], n_samples))

    def load_data(self, shuffle=False):

        allfiles = os.listdir(self.data_dir)
        npyfiles = []
        for f in allfiles:
            if ".npy" in f:
                npyfiles.append(os.path.join(self.data_dir, f))

        npyfiles.sort(key=lambda x: (len(x), x))

        datafiles = npyfiles[:len(npyfiles)//2]
        labelfiles = npyfiles[len(npyfiles)//2:]
        datafiles = datafiles[:self.n_files]
        labelfiles = labelfiles[:self.n_files]

        # Divide Training & Testing Sets
        r_permute = np.random.permutation(len(datafiles))
        filename = os.path.join("r_permute{}.npz".format(len(datafiles)))
        if (os.path.isfile(filename)):
            with np.load(filename) as f:
                print("already exist")
                r_permute = f["inds"]
        else:
            save_dict = {
                "inds": r_permute,
            }
            np.savez(filename, **save_dict)

        datafiles = np.asarray(datafiles)[r_permute]
        labelfiles = np.asarray(labelfiles)[r_permute]
        traindata_files = np.array_split(datafiles, self.n_folds)
        trainlabel_files = np.array_split(labelfiles, self.n_folds)
        subjectdata_files = traindata_files[self.fold_idx]
        subjectlabel_files = trainlabel_files[self.fold_idx]
        traindata_files = list(set(datafiles) - set(subjectdata_files))
        trainlabel_files = list(set(labelfiles) - set(subjectlabel_files))
        traindata_files.sort(key=lambda x: (len(x), x))
        trainlabel_files.sort(key=lambda x: (len(x), x))

        # Load training and validation sets
        print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print("Load training set:")
        data_train, label_train = self._load_npy_list_files(
            traindata_files, trainlabel_files)
        print(" ")
        print("Load Test set:")
        data_test, label_test = self._load_npy_list_files(
            subjectdata_files, subjectlabel_files)
        print(" ")
        print("Training set: n_subjects={}".format(len(data_train)))
        n_train_examples = 0
        for d in data_train:
            n_train_examples += d.shape[0]
        print("Number of examples = {}".format(n_train_examples))
        self.print_n_samples_each_class(np.hstack(label_train), self.classes)
        print(" ")
        print("Test set: n_subjects = {}".format(len(data_test)))
        n_test_examples = 0
        for d in data_test:
            n_test_examples += d.shape[0]
        print("Number of examples = {}".format(n_test_examples))
        self.print_n_samples_each_class(np.hstack(label_test), self.classes)
        print(" ")

        data_train = np.vstack(data_train)
        label_train = np.hstack(label_train)

        data_test = np.vstack(data_test)
        label_test = np.hstack(label_test)

        if shuffle is True:
            # training data
            permute = np.random.permutation(len(label_train))
            data_train = np.asarray(data_train)
            data_train = data_train[permute]
            label_train = label_train[permute]

        return data_train, label_train, data_test, label_test
