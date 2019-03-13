from model import *
import config2 as cfg
import os
import cv2
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

val_img = cv2.imread("test/21.png",0)
val_img = cv2.resize(val_img,(240,32))
val_img = val_img.swapaxes(0, 1)
val_img = np.zeros(( 512,val_img.shape[0], val_img.shape[1], 1))+np.array(val_img[ np.newaxis,:, :, np.newaxis])
# print val_img.shape
# sys.exit(0)

loss,train_decode_result,pred_decode_result=build_network(is_training=True)
saver = tf.train.Saver()

sess = tf.Session()

ckpt = tf.train.latest_checkpoint(cfg.CKPT_DIR)
if ckpt:
    saver.restore(sess,ckpt)
    print('restore from the checkpoint{0}'.format(ckpt))
else:
    print('failed to load ckpt')

# val_img,_=cfg.read_data_train(cfg.val_dir,'Synthetic Chinese String Dataset_2/train.txt',0,1000)
val_predict = sess.run(pred_decode_result,feed_dict={image: val_img[0:cfg.BATCH_SIZE]})
predit = cfg.int2label(np.argmax(val_predict, axis=2))
for i in predit[0]:
    print i


