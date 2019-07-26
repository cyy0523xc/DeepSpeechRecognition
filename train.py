import os
from time import time
import tensorflow as tf
from utils import get_data, data_hparams
from keras.callbacks import ModelCheckpoint
from model_speech.cnn_ctc import Am, am_hparams
from model_language.transformer import Lm, lm_hparams

# TF1.14.0会报错：
# AttributeError: '_TfDeviceCaptureOp' object has no attribute '_set_device_from_string'

# 开始时间
start_time = time()

# 0.准备训练所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'train'
data_args.data_path = '/corpus/'
data_args.thchs30 = True
data_args.aishell = False
data_args.prime = False
data_args.stcmd = False
# 系统空余内存15G
# 设置为32时：
# Allocation of 897843200 exceeds 10% of system memory.
# 使用cpu训练：设置为16时，一步耗时约12秒
# 2080TI可以设置为32，显存几乎100%，272s/epoch 872ms/step
data_args.batch_size = 32
data_args.data_length = 10000 // 32
# data_args.data_length = None
data_args.shuffle = True
train_data = get_data(data_args)

# 0.准备验证所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'dev'
data_args.data_path = '/corpus/'
data_args.thchs30 = True
data_args.aishell = False
data_args.prime = False
data_args.stcmd = False
data_args.batch_size = 32
# data_args.data_length = None
# data_args.data_length = 10
data_args.data_length = 10000 // 32
data_args.shuffle = True
dev_data = get_data(data_args)

# 1.声学模型训练-----------------------------------
print("*"*80)
print("训练声学模型...")
print("*"*80)
am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am_args.gpu_nums = 2
am_args.lr = 0.0008
am_args.is_training = True
am = Am(am_args)

if os.path.exists('logs_am/model.h5'):
    print('load acoustic model...')
    am.ctc_model.load_weights('logs_am/model.h5')

epochs = 10
batch_num = len(train_data.wav_lst) // train_data.batch_size

# checkpoint
ckpt = "model_{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join('./checkpoint', ckpt),
                             monitor='val_loss', save_weights_only=False,
                             verbose=1, save_best_only=True)

#
# for k in range(epochs):
#     print('this is the', k+1, 'th epochs trainning !!!')
#     batch = train_data.get_am_batch()
#     dev_batch = dev_data.get_am_batch()
#     am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=10, callbacks=[checkpoint], workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=200)

batch = train_data.get_am_batch()
dev_batch = dev_data.get_am_batch()

am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs,
                           callbacks=[checkpoint], workers=1,
                           use_multiprocessing=False,
                           validation_data=dev_batch, validation_steps=200)
am.ctc_model.save_weights('logs_am/model.h5')

speech_time = time() - start_time
print("训练声学模型: ", speech_time)


# 2.语言模型训练-------------------------------------------
start_time = time()
print("*"*80)
print("训练语言模型...")
print("*"*80)
lm_args = lm_hparams()
lm_args.num_heads = 8
lm_args.num_blocks = 6
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)
lm_args.max_length = 100
lm_args.hidden_units = 512
lm_args.dropout_rate = 0.2
lm_args.lr = 0.0003
lm_args.is_training = True
lm = Lm(lm_args)

epochs = 10
with lm.graph.as_default():
    saver = tf.train.Saver()
with tf.Session(graph=lm.graph) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    add_num = 0
    if os.path.exists('logs_lm/checkpoint'):
        print('loading language model...')
        latest = tf.train.latest_checkpoint('logs_lm')
        add_num = int(latest.split('_')[-1])
        saver.restore(sess, latest)
    writer = tf.summary.FileWriter('logs_lm/tensorboard', tf.get_default_graph())
    for k in range(epochs):
        total_loss = 0
        batch = train_data.get_lm_batch()
        for i in range(batch_num):
            input_batch, label_batch = next(batch)
            feed = {lm.x: input_batch, lm.y: label_batch}
            cost, _ = sess.run([lm.mean_loss, lm.train_op], feed_dict=feed)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs = sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)
        print('epochs', k+1, ': average loss = ', total_loss/batch_num)
    saver.save(sess, 'logs_lm/model_%d' % (epochs + add_num))
    writer.close()

print("*"*80)
print("训练声学模型: ", speech_time)
print("训练语言模型: ", time()-start_time)
