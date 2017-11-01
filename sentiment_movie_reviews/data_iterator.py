import os
import numpy as np

batching_seed = np.random.RandomState(1234)


class LengthAwareDataIterator:
    def __init__(self, x, mask, y, batch_size, padded=False):

        self.x = x
        self.mask = mask
        self.y = y

        self._batch_size = batch_size

        if padded:
            # for evaluation, pad until multiple of batchsize
            if self.x.shape[0] % self._batch_size != 0:
                additional_rows = self._batch_size - self.x.shape[0] % self._batch_size
                x_additional = np.zeros((additional_rows, self.x.shape[1]), dtype=np.int32)
                self.x = np.concatenate([x_additional, self.x])

                mask_additional = np.zeros((additional_rows, self.mask.shape[1]), dtype=np.int32)
                self.mask = np.concatenate([mask_additional, self.mask])

                y_additional = np.zeros((additional_rows,), dtype=np.int32)
                self.y = np.concatenate([y_additional, self.y])

        self.lengths = np.sum(self.mask, axis=1)
        self.sorted_indexes = sorted(range(self.lengths.shape[0]), key=lambda k: self.lengths[k])

    def get_batch(self, train=True):

        n_batches = self.x.shape[0] / self._batch_size

        if train:
            batches = batching_seed.permutation(range(n_batches))
        else:
            batches = range(n_batches)

        for i in batches:
            batch_indexes = self.sorted_indexes[i * self._batch_size:(i + 1) * self._batch_size]
            max_len = self.lengths[batch_indexes[-1]]

            x_batch = self.x[batch_indexes, :max_len]
            mask = self.mask[batch_indexes, :max_len]
            y_batch = self.y[batch_indexes]

            yield x_batch, y_batch, mask


def load_model(paras):

    print("Loading data")

    x_train = np.load(os.path.join(paras["data_dir"], paras["load_file"] + ".x_train.npy"))
    x_mask_train = np.load(os.path.join(paras["data_dir"], paras["load_file"] + ".x_mask_train.npy"))
    y_train = np.load(os.path.join(paras["data_dir"], paras["load_file"] + ".y_train.npy"))
    x_test = np.load(os.path.join(paras["data_dir"], paras["load_file"] + ".x_test.npy"))
    x_mask_test = np.load(os.path.join(paras["data_dir"], paras["load_file"] + ".x_mask_test.npy"))
    y_test = np.load(os.path.join(paras["data_dir"], paras["load_file"] + ".y_test.npy"))
    matrix = np.load(os.path.join(paras["data_dir"], paras["load_file"] + ".matrix.npy"))

    print("Loaded data")
    print(x_train.shape)
    np.random.seed(1234)
    indexes = np.random.permutation(range(x_train.shape[0]))

    fold_start = int((paras["heldout_index"] - 1) * paras["heldout"] * x_train.shape[0])
    fold_end = int((paras["heldout_index"]) * paras["heldout"] * x_train.shape[0])

    valid_indexes_mask = np.zeros((x_train.shape[0],), dtype=np.bool)
    valid_indexes_mask[fold_start:fold_end] = True
    valid_indexes = indexes[valid_indexes_mask]
    train_indexes_mask = np.bitwise_not(valid_indexes_mask)
    train_indexes = indexes[train_indexes_mask]

    train_it = LengthAwareDataIterator(x_train[train_indexes],
                                       x_mask_train[train_indexes],
                                       y_train[train_indexes],
                                       paras["batch_size"])
    valid_it = LengthAwareDataIterator(x_train[valid_indexes],
                                       x_mask_train[valid_indexes],
                                       y_train[valid_indexes],
                                       paras["batch_size"], padded=True)
    test_it = LengthAwareDataIterator(x_test, x_mask_test, y_test, paras["batch_size"], padded=True)

    paras["input_vocab_size"] = matrix.shape[0]
    paras["embedding_size"] = matrix.shape[1]
    paras["model_seq_len"] = None

    return train_it, valid_it, test_it, matrix
