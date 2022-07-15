import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Run on CPU

import tensorflow as tf
import ltc_network as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse

# LNN for one amino acid - George

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


def load_trace():
    total = 0
    labels = []
    x = []
    y = []
    proteins_stats = []
    total_ck = 0
    labels_dict = {'#': 0, 'H': 1, 'E': 2, 'S': 3, 'T': 3, 'B': 2, 'G': 1, 'p': 3, 'I': 3, '-': 3}

    with open("/home/prody/Desktop/struct-seq.dat", "r") as f:
        miss = 0
        f.readline()
        line = f.readline()
        N = 1000000
        #x = np.empty(N, dtype=np.float32)
        #y = np.empty(N, dtype=np.int32)
        while total < N:
            if line[0] == '#':
                proteins_stats.append(total - total_ck)
                total_ck = total
                x.append([0,])
                x.append([0,])
                x.append([0,])
                y.append(0)        
                y.append(0)        
                y.append(0)        
                    
                #total += 1
                line = f.readline()
                continue
            aa = line[0]
            label = line[-2]
            x.append([float(ord(aa) - 64),])
            #y[total] = int(labels_dict[label])
            y.append(int(labels_dict[label]))
            total += 1
            #if label not in labels:
            #    labels.append(label)
            line = f.readline()



    # found '!' in the data
    print("Missing features in {} out of {} samples ({:0.2f})".format(miss,total,100*miss/total))
    print("Read {} lines".format(len(x)))
    #all_x = np.stack(all_x,axis=0)
    x = np.array(x).astype(np.float32)

    #print("Imbalance: {:0.2f}%".format(100*np.mean(all_y)))
    #all_y -= np.mean(all_y)  # normalize
    #all_y /= np.std(all_y)   # normalize
    x -= np.mean(x)   # normalize
    x /= np.std(x)    # normalize

    return x, y


def cut_in_sequences(x,y,seq_len,interleaved=1):

    num_sequences = x.shape[0]//seq_len
    sequences = []

    for s in range(num_sequences):
        start = seq_len*s
        end = start+seq_len
        sequences.append((x[start:end],y[start:end]))

        if(interleaved and s < num_sequences - 1):
            start += seq_len//2
            end = start+seq_len
            sequences.append((x[start:end],y[start:end]))

    return sequences



class SeqData:
    def __init__(self,seq_len=32):

        x,y = load_trace()
        
        train_traces = []
        valid_traces = []
        test_traces = []
        
        train_x, train_y = list(zip(*cut_in_sequences(x, y, seq_len, interleaved=True)))
        self.train_x = np.stack(train_x,axis=1)
        print(self.train_x.shape)
        self.train_y = np.stack(train_y,axis=1)
        print(self.train_y.shape)

        total_seqs = self.train_x.shape[1]
        print("Total number of training sequences: {}".format(total_seqs))
        #permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.10*total_seqs)
        test_size =  int(0.15*total_seqs)

        # self.valid_x = self.train_x[:, permutation[:valid_size]]
        # self.valid_y = self.train_y[:, permutation[:valid_size]]
        # self.test_x = self.train_x[:,  permutation[valid_size:valid_size+test_size]]
        # self.test_y = self.train_y[:,  permutation[valid_size:valid_size+test_size]]
        # self.train_x = self.train_x[:, permutation[valid_size+test_size:]]
        # self.train_y = self.train_y[:, permutation[valid_size+test_size:]]
        self.valid_x = self.train_x[:, :valid_size]
        self.valid_y = self.train_y[:, :valid_size]
        self.test_x = self.train_x[:,  valid_size:valid_size+test_size]
        print(self.test_x.shape)
        self.test_y = self.train_y[:,  valid_size:valid_size+test_size]
        print(self.test_y.shape)
        self.train_x = self.train_x[:, valid_size+test_size:]
        self.train_y = self.train_y[:, valid_size+test_size:]

    def iterate_train(self, batch_size=16):
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
    def __init__(self,model_type,model_size,epochs, learning_rate = 0.001):
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None,1])
        self.target_y = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.epochs = epochs
        self.model_size = model_size
        head = self.x
        if (model_type == "lstm"):
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)

            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        elif (model_type.startswith("ltc")):
            learning_rate = 0.01  # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
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
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=2)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(model_prediction, tf.cast(self.target_y, tf.int64)), tf.float32))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join("results", "seq", "{}_{}_{}.csv".format(model_type, model_size, epochs))
        if (not os.path.exists("results/seq")):
            os.makedirs("results/seq")
        if (not os.path.isfile(self.result_file)):
            with open(self.result_file, "w") as f:
                f.write(
                    "best epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        self.checkpoint_path = os.path.join("tf_sessions", "seq", "{}".format(model_type))
        if (not os.path.exists("tf_sessions/seq")):
            os.makedirs("tf_sessions/seq")

        self.saver = tf.train.Saver()
  

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)


    def fit(self,gesture_data,verbose=True,log_period=50):
        best_valid_acc = 0
        epochs = self.epochs
        best_valid_stats = (0,0,0,0,0,0,0)
        self.save()
        print(gesture_data.test_x.shape)
        print(gesture_data.test_y.shape)
        for e in range(epochs):
            if(verbose and e%log_period == 0):
                
                test_acc,test_loss = self.sess.run([self.accuracy,self.loss],{self.x:np.transpose(gesture_data.test_x, (0,1,2)),self.target_y: gesture_data.test_y})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.x:np.transpose(gesture_data.valid_x, (0,1,2)),self.target_y: gesture_data.valid_y})

                # F1 metric -> higher is better
                if((valid_acc > best_valid_acc and e > 0) or e==1):
                    best_valid_acc = valid_acc
                    best_valid_stats = (
                        e,
                        np.mean(losses),np.mean(accs),
                        valid_loss,valid_acc,
                        test_loss,test_acc
                    )
                    self.save()

            losses = []
            accs = []
            for batch_x,batch_y in gesture_data.iterate_train(batch_size=64):
                acc,loss,_ = self.sess.run([self.accuracy,self.loss,self.train_step],{self.x:batch_x,self.target_y: batch_y})
                if(len(self.constrain_op) > 0):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if(verbose and e%log_period == 0):
                print("Epochs {:03d}, train loss: {:0.2f}, train acc: {:0.2f}, valid loss: {:0.2f}, valid acc: {:0.2f}, test loss: {:0.2f}, test acc: {:0.2f}".format(
                    e,
                    np.mean(losses),np.mean(accs),
                    valid_loss,valid_acc,
                    test_loss,test_acc
                ))
            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train acc: {:0.2f}, valid loss: {:0.2f}, valid acc: {:0.2f}, test loss: {:0.2f}, test acc: {:0.2f}".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))
        with open(self.result_file,"a") as f:
            f.write("{:08d}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}\n".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))


if __name__ == "__main__":
    # https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="lstm")
    parser.add_argument('--log',default=1,type=int)
    parser.add_argument('--size',default=32,type=int)
    parser.add_argument('--epochs',default=200,type=int)

    args = parser.parse_args()

    seq_data = SeqData()
    model = SeqModel(model_type=args.model, model_size=args.size, epochs=args.epochs)
    model.fit(seq_data, log_period=args.log)

