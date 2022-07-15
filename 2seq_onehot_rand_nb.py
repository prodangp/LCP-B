import numpy as np
import os
import time
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU

import tensorflow as tf
import ltc_network as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse
import json

from nni_functions import interaction_score

# NNI method for inputs - George


AA_MAX = 0
ZERO_COUNT_TRAINING = 0
ZERO_COUNT_VALIDATION = 0
ZERO_COUNT_TEST = 0
AA_TRAINING = None
AA_TEST = None
AA_VALIDATION = None
N = 2000


def std_acc(acc, N):
    return (acc*(1-acc)/N)**0.5


def load_trace(nni_function, window_size, seq_len, hydro_enabled, k):
    global ZERO_COUNT
    global N
    sequences = []
    miss = 0
    labels_dict = {'#': 0, 'H': 1, 'E': 2, 'S': 3, 'T': 3, 'B': 2, 'G': 1, 'p': 3, 'I': 3, '-': 3, 'C': 3}
    aa_codes = [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24]
    # ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']
    hydro = [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 0, 1]

    f = open("../data/proteins/struct-seq250.json", "r")
    data = np.array(json.load(f))
    permutation = np.random.RandomState(23489).permutation(len(data))
    
    global AA_MAX
    aa_nbs = []
    for protein in data[permutation][:N]:
        aa_nbs.append(len(protein["aminoacids"]))
    
    print('TOTAL #PROTEINS:', len(data))
    print('AA_MAX = ', max(aa_nbs))
    print('AA_MIN = ', min(aa_nbs))
    print('AA_AVG = ', np.mean(aa_nbs))
    print('AA_STD = ', np.std(aa_nbs))
    
    AA_MAX = max(aa_nbs)
    total_ = 0
    for protein in data[permutation][:N]:
        one_hot_vect = np.zeros(21, dtype=np.float32)
        x = []
        y = []
        for _ in range(window_size):
            x.append(one_hot_vect)
            y.append(0)
        aa_arr = [int(ord(aa) - 65) for aa in protein["aminoacids"]]
        for t in range(len(aa_arr)):
            one_hot_vect = np.zeros(21, dtype=np.float32)
            aa_code = aa_arr[t]
            label = protein["labels"][t]
            try:
                idx = aa_codes.index(aa_code)
                one_hot_vect[idx] = 1
            except ValueError:
                miss += 1
            for pos in range(1, int((window_size-1)/2) + 1):
                try:
                    idx_l = aa_codes.index(aa_arr[t - pos])
                    idx_r = aa_codes.index(aa_arr[t + pos])
                    if hydro_enabled:
                        one_hot_vect[idx_l] += hydro[idx_l] * hydro[idx] * interaction_score[nni_function](pos, k)
                        one_hot_vect[idx_r] += hydro[idx_l] * hydro[idx] * interaction_score[nni_function](pos, k)
                    else:
                        one_hot_vect[idx_l] += interaction_score[nni_function](pos, k)
                        one_hot_vect[idx_r] += interaction_score[nni_function](pos, k)
                except ValueError:
                    miss += 1
                except IndexError:
                    pass
            x.append(one_hot_vect)
            y.append(int(labels_dict[label]))
        total_ += len(protein["aminoacids"])
        for aa in range(len(protein["aminoacids"])):
            sequence = np.array(x[aa: seq_len + aa]).astype(np.float32)
            label = y[aa: seq_len + aa]
            sequences.append((sequence, label))
    print("Unrecognized aminoacids in {} out of {} samples ({:0.2f})".format(miss, total_, 100*miss/total_))
    print("Read {} lines".format(len(x)))

    return sequences


class SeqData:
    def __init__(self, nni_function=False, window=1, seq_len=1, hydro=0, k=0.5):
        self.nni_function = nni_function
        self.window = window
        self.seq_len = seq_len
        self.hydro_enabled = hydro
        self.k = k
        train_x, train_y = list(zip(*load_trace(nni_function, window, seq_len, hydro, self.k)))
        self.train_x = np.stack(train_x, axis=1)
        self.train_y = np.stack(train_y, axis=1)

        global N 
        print("Total number of proteins: {}".format(N))
        # permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.2*N)
        test_size = int(0.2*N)

        self.valid_x = np.array(self.train_x)[:, :valid_size]
        self.valid_y = np.array(self.train_y)[:, :valid_size]
        global ZERO_COUNT_VALIDATION
        global AA_VALIDATION
        ZERO_COUNT_VALIDATION = np.count_nonzero(self.valid_y == 0)
        AA_VALIDATION = self.valid_y.shape[0] * self.valid_y.shape[1]
        print(self.valid_y.shape)
        print(AA_VALIDATION)
        print('ZERO COUNT VALIDATION =', ZERO_COUNT_VALIDATION)
        self.test_x = np.array(self.train_x)[:, valid_size:valid_size + test_size]
        self.test_y = np.array(self.train_y)[:, valid_size:valid_size + test_size]
        global ZERO_COUNT_TEST
        global AA_TEST
        ZERO_COUNT_TEST = np.count_nonzero(self.test_y == 0)
        AA_TEST = self.test_y.shape[0] * self.test_y.shape[1]
        print(AA_TEST)
        print('ZERO COUNT TEST =', ZERO_COUNT_TEST)
        self.train_x = np.array(self.train_x)[:, valid_size + test_size:]
        self.train_y = np.array(self.train_y)[:, valid_size + test_size:]
        global ZERO_COUNT_TRAINING
        global AA_TRAINING
        ZERO_COUNT_TRAINING = np.count_nonzero(self.train_y == 0)
        AA_TRAINING = self.train_y.shape[0] * self.train_y.shape[1]
        print(AA_TRAINING)
        print('ZERO COUNT TRAINING =', ZERO_COUNT_TRAINING)

    def iterate_train(self, batch_size=32):
        total_seqs = self.train_x.shape[1]
        # permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            # batch_x = self.train_x[:,permutation[start:end]]
            # batch_y = self.train_y[:,permutation[start:end]]
            batch_x = self.train_x[:, start:end]
            batch_y = self.train_y[:, start:end]
            yield (batch_x, batch_y)


class SeqModel:
    def __init__(self, model_type, model_size, epochs, opt='Adam', learning_rate=0.01, activation='sigmoid'):
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 21])
        self.target_y = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.epochs = epochs
        self.model_size = model_size
        self.opt = opt
        self.lr = learning_rate
        self.activation = activation

        head = self.x
        if model_type == "lstm":
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)

            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        elif (model_type.startswith("ltc")):
            self.wm = ltc.LTCCell(model_size, activation)
            if (model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif (model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head, _ = tf.nn.dynamic_rnn(self.wm, head, dtype=tf.float32, time_major=True)
            self.constrain_op = self.wm.get_param_constrain_op()
        elif (model_type == "node"):
            self.fused_cell = NODE(model_size, cell_clip=-1)
            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        elif (model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size, cell_clip=-1)
            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        elif (model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size, cell_clip=-1, global_feedback=True)
            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))

        #tf.keras.layers.Flatten() 
        self.y = tf.layers.Dense(4, activation=None)(head)
        print("logit shape: ", str(self.y.shape))
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels=self.target_y,
            logits=self.y,
        ))
        
        optimizer_dict = {'Adam': tf.train.AdamOptimizer,
                          'Adagrad': tf.train.AdagradOptimizer,
                          'RMSProp': tf.train.RMSPropOptimizer,
                          'GD': tf.train.GradientDescentOptimizer,
                          'Adadelta': tf.train.AdadeltaOptimizer} 
                                 
        optimizer = optimizer_dict[self.opt](learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=2)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(model_prediction, tf.cast(self.target_y, tf.int64)), tf.float32))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.result_file = "{}_{}_{}_{}".format(model_type, model_size, epochs, opt)
        
        if (not os.path.exists("results/seq2")):
            os.makedirs("results/seq2")
        if not os.path.isfile('results/seq2/experiments'):
            with open('results/seq2/experiments', "w") as f:
                f.write(
                    "model,size,lr,opt,activation,nni,window,seq_len,hydro,k,best_epoch,train_loss,train_accuracy,train_std,valid_loss,valid_accuracy,val_std,test_loss,test_accuracy,test_std\n")

        self.checkpoint_path = os.path.join("../tf_sessions", "seq", "{}".format(model_type))
        if (not os.path.exists("../tf_sessions/seq")):
            os.makedirs("../tf_sessions/seq")

        self.saver = tf.train.Saver()
  

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)


    def fit(self,seq_data,verbose=True,log_period=50):
        best_valid_acc = 0
        n_train = AA_TRAINING - ZERO_COUNT_TRAINING
        zf = ZERO_COUNT_TRAINING / n_train
        n_val = AA_VALIDATION - ZERO_COUNT_VALIDATION
        zf_v = ZERO_COUNT_VALIDATION / n_val
        n_test = AA_TEST - ZERO_COUNT_TEST
        zf_t = ZERO_COUNT_TEST / n_test
        epochs = self.epochs
        best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
        self.save()
        ts = int(time.time()) % 1000000
        for e in range(epochs):
            if (verbose and e % log_period == 0):

                test_acc, test_loss = self.sess.run([self.accuracy, self.loss],
                                                    {self.x: np.transpose(seq_data.test_x, (0, 1, 2)),
                                                     self.target_y: seq_data.test_y})
                valid_acc, valid_loss = self.sess.run([self.accuracy, self.loss],
                                                      {self.x: np.transpose(seq_data.valid_x, (0, 1, 2)),
                                                       self.target_y: seq_data.valid_y})

                # F1 metric -> higher is better
                if ((valid_acc > best_valid_acc and e > 0) or e == 1):
                    best_valid_acc = valid_acc
                    best_valid_stats = (
                        e,
                        np.mean(losses), (np.mean(accs) - (zf / (1 + zf))) * (1 + zf),
                        valid_loss, (valid_acc - (zf_v / (1 + zf_v))) * (1 + zf_v),
                        test_loss, (test_acc - (zf_t / (1 + zf_t))) * (1 + zf_t)
                    )
                    self.save()
            # (TP+ZEROS/N+ZEROS - zf/1+zf)) * (1+ZEROS/N)
            losses = []
            accs = []
            for batch_x, batch_y in seq_data.iterate_train(batch_size=64):
                acc, loss, _ = self.sess.run([self.accuracy, self.loss, self.train_step],
                                             {self.x: batch_x, self.target_y: batch_y})
                if self.constrain_op is not None:
                    if len(self.constrain_op) > 0:
                        self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if (verbose and e % log_period == 0):
                acc_train = (np.mean(accs) - (zf / (1 + zf))) * (1 + zf)
                acc_val = (valid_acc - (zf_v / (1 + zf_v))) * (1 + zf_v)
                acc_test = (test_acc - (zf_t / (1 + zf_t))) * (1 + zf_t)
                epoch_result = "Epochs {:03d}, train loss: {:0.2f}, train acc: {:0.2f} (std {:0.2f}), valid loss: {:0.2f}, valid acc: {:0.2f} (std {:0.2f}), test loss: {:0.2f}, test acc: {:0.2f} (std {:0.2f})".format(
                        e,
                        np.mean(losses), acc_train, std_acc(acc_train, n_train),
                        valid_loss, acc_val, std_acc(acc_val, n_val),
                        test_loss, acc_test, std_acc(acc_test, n_test)
                    )
                print(epoch_result)

                with open('./results/seq2/' + str(ts) + '_' + self.result_file + '_epochs.csv', "a+") as f:
                    f.write(epoch_result + '\n')

            if (e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = best_valid_stats
        print(
            "Best epoch {:03d}, train loss: {:0.2f}, train acc: {:0.2f}, valid loss: {:0.2f}, valid acc: {:0.2f}, test loss: {:0.2f}, test acc: {:0.2f}".format(
                best_epoch,
                train_loss, train_acc,
                valid_loss, valid_acc,
                test_loss, test_acc
            ))
        with open('./results/seq2/' + str(ts) + '_' + self.result_file + '.csv', "a+") as f:
            f.write("{:08d}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}\n".format(
                best_epoch,
                train_loss, train_acc,
                valid_loss, valid_acc,
                test_loss, test_acc
            ))
        with open('results/seq2/experiments', "a") as f:
            f.write(
                f"{self.model_type},{self.model_size},{self.lr},{self.opt},{self.activation},{seq_data.nni_function},"
                f"{seq_data.window}, {seq_data.seq_len}, {seq_data.hydro_enabled}, {seq_data.k}, {best_epoch},"
                f"{train_loss},{train_acc},{std_acc(train_acc, n_train)},"
                f"{valid_loss},{valid_acc},{std_acc(valid_acc, n_val)},"
                f"{test_loss},{test_acc}, {std_acc(test_acc, n_test)}\n")
        self.save()
if __name__ == "__main__":
    # https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="ltc")
    parser.add_argument('--log', default=1, type=int)
    parser.add_argument('--size', default=32, type=int)
    parser.add_argument('--opt', default='Adam', type=str)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--activation', default='sigmoid', type=str)
    parser.add_argument('--window', default=1, type=int)
    parser.add_argument('--seq_len', default=1, type=int)
    parser.add_argument('--nni_func', default=None, type=str)
    parser.add_argument('--hydro', default=0, type=int)
    parser.add_argument('--k', default=0.5, type=float)
    args = parser.parse_args()

    seq_data = SeqData(nni_function=args.nni_func, window=args.window, seq_len=args.seq_len, hydro=args.hydro, k=args.k)
    model = SeqModel(model_type=args.model, model_size=args.size, opt=args.opt, epochs=args.epochs,
                     learning_rate=args.lr, activation=args.activation)
    model.fit(seq_data, log_period=args.log)

