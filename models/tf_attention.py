from attention_model import *
import attention_config as cfg
import cv2
import pickle
import time
import os
from sklearn.utils import shuffle
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# sys.exit(0)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)

with open("../data/sequences_20_all_same_train_new.txt", "r") as f1:
    seq_labels = pickle.load(f1)
with open("../data/sequences_20_all_same_test_new.txt", "r") as f2:
    seq_val_labels = pickle.load(f2)

#Call loss and gradient descent
loss,train_decode_result,pred_decode_result=build_network(is_training=True)
optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9, use_nesterov=True)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op=optimizer.minimize(loss)

#Save parameters
var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list,max_to_keep=5)

#Session run
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
with tf.name_scope('summaries'):
    tf.summary.scalar("cost", loss)
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter(cfg.LOGS_PATH)

#Load model
saved_model = tf.train.latest_checkpoint(cfg.SAVE_PATH)
if saved_model:
    saver.restore(sess,saved_model)
    print('restore from the checkpoint{0}'.format(saved_model))

val_img, val_label = cfg.get_data(cfg.val_dir,seq_val_labels,0,30000)
for cur_epoch in range(cfg.EPOCH):
    for j in range(0,32):
        img, label = cfg.get_data(cfg.train_dir, seq_labels,30000*j,30000*(j+1))
        num_train_samples = img.shape[0]
        num_batches_per_epoch = int(num_train_samples / cfg.BATCH_SIZE)
        target_in, target_out = cfg.label2int(label)
        sys.exit(0)
        shuffle_idx = np.random.permutation(num_train_samples)
        train_cost = 0
        for cur_batch in range(num_batches_per_epoch):
            val_img,val_label=shuffle(val_img,val_label)
            indexs = [shuffle_idx[i % num_train_samples] for i in
                      range(cur_batch * cfg.BATCH_SIZE, (cur_batch + 1) * cfg.BATCH_SIZE)]
            batch_inputs,batch_target_in,batch_target_out=img[indexs],target_in[indexs],target_out[indexs]
            sess.run( train_op,feed_dict={image: batch_inputs,train_output: batch_target_in,target_output: batch_target_out,sample_rate:np.min([1.,0.2*cur_epoch+0.2])})
            if cur_batch%cfg.DISPLAY_STEPS==0:
                summary_loss, loss_result = sess.run([summary_op, loss],feed_dict={image: batch_inputs,train_output: batch_target_in,target_output: batch_target_out,
                                                                                   sample_rate: np.min([1., 1.])})
                writer.add_summary(summary_loss, cur_epoch*num_batches_per_epoch+cur_batch)
                val_predict = sess.run(pred_decode_result,feed_dict={image: val_img[0:cfg.BATCH_SIZE]})
                train_predict = sess.run(pred_decode_result, feed_dict={image: batch_inputs, train_output: batch_target_in,
                                                                         target_output: batch_target_out,sample_rate:np.min([1., 1.])})
                predit = cfg.int2label(np.argmax(val_predict, axis=2))
                train_pre = cfg.int2label(np.argmax(train_predict, axis=2))
                gt = val_label[0:cfg.BATCH_SIZE]
                acc = cfg.cal_acc_char(predit, gt)
                train_acc=cfg.cal_acc_char(train_pre,cfg.int2label(batch_target_out))
                print("epoch:{}, data_part:{},batch:{}, loss:{}, val_acc:{},train_acc:{}".
                      format(cur_epoch,j, cur_batch,
                             loss_result, acc,train_acc
                             ))
                if not os.path.exists(cfg.SAVE_PATH):
                    os.makedirs(cfg.SAVE_PATH)
                saver.save(sess, os.path.join(cfg.SAVE_PATH, 'attention_ocr.model'), global_step=cur_epoch*num_batches_per_epoch+cur_batch)
