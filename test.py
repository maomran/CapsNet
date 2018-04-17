"""
License: Apache-2.0
Author: Suofei Zhang | Hang Yu
E-mail: zhangsuofei at njupt.edu.cn | hangyu5 at illinois.edu
"""
import warnings
import sys
import tensorflow as tf
from config import cfg, get_coord_add, get_dataset_size_train, get_dataset_size_test, get_num_classes, get_create_inputs
import time
import os
import capsnet_em as net
import tensorflow.contrib.slim as slim

import logging
import daiquiri
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

warnings.filterwarnings("ignore")
rundata = 1
def main(args):
    """Get dataset hyperparameters."""
    assert len(args) == 3 and isinstance(args[1], str) and isinstance(args[2], str)
    dataset_name = args[1]
    model_name = args[2]
    coord_add = get_coord_add(dataset_name)
    dataset_size_train = get_dataset_size_train(dataset_name)
    dataset_size_test = get_dataset_size_test(dataset_name)
    num_classes = get_num_classes(dataset_name)
    create_inputs = get_create_inputs(
        dataset_name, is_train=False, epochs=cfg.epoch)

    """Set reproduciable random seed"""
    tf.set_random_seed(1234)

    with tf.Graph().as_default():
        num_batches_per_epoch_train = int(dataset_size_train / cfg.batch_size)
        num_batches_test = int(dataset_size_test / cfg.batch_size * 0.1)

        batch_x, batch_labels = create_inputs()
        batch_x = slim.batch_norm(batch_x, center=False, is_training=False, trainable=False)
        if model_name == "caps":
            output, _, hist = net.build_arch(batch_x, coord_add,
                                       is_train=False, num_classes=num_classes)
        elif model_name == "cnn_baseline":
            output, hist = net.build_arch_baseline(batch_x,
                                             is_train=False, num_classes=num_classes)
        else:
            raise "Please select model from 'caps' or 'cnn_baseline' as the secondary argument of eval.py!"
        batch_acc = net.test_accuracy(output, batch_labels)
        saver = tf.train.Saver()

        step = 0

        summaries = []
        summaries.append(hist)
        summaries.append(tf.summary.scalar('accuracy', batch_acc))
        summary_op = tf.summary.merge(summaries)

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)) as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            if not os.path.exists(cfg.test_logdir + '/{}/{}/'.format(model_name, dataset_name)):
                os.makedirs(cfg.test_logdir + '/{}/{}/'.format(model_name, dataset_name))
            summary_writer = tf.summary.FileWriter(
                cfg.test_logdir + '/{}/{}/'.format(model_name, dataset_name), graph=sess.graph)  # graph=sess.graph, huge!

            files = os.listdir(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name))
            for epoch in range(1, cfg.epoch):
               # requires a regex to adapt the loss value in the file name here
#                ckpt_re = ".ckpt-%d" % (num_batches_per_epoch_train * epoch)
#                for __file in files:
#                   if __file.endswith(ckpt_re + ".index"):
#                       ckpt = os.path.join(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name), __file[:-6])
                #ckpt = os.path.join(cfg.logdir, "model.ckpt-%d" % (num_batches_per_epoch_train * epoch))
#                saver.restore(sess, ckpt)
                if (model_name == 'cnn_baseline'):
                    saver = tf.train.import_meta_graph(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name)+"model.ckpt-46800.meta")
                else:
                    if (dataset_name == 'mnist'):
                        saver = tf.train.import_meta_graph(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name)+"model-0.0630.ckpt-55000.meta")
                    elif (dataset_name == 'fashion_mnist'):
                        saver = tf.train.import_meta_graph(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name)+"model-55000.ckpt-55000.meta")
                    else:
                        saver = tf.train.import_meta_graph(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name)+"model-0.4319.ckpt-46800.meta")

                saver.restore(sess,tf.train.latest_checkpoint(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name)))
                accuracy_sum = 0
                for i in range(num_batches_test):
                    if (rundata == 1):
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        batch_acc_v, summary_str = sess.run([batch_acc, summary_op], options=run_options,run_metadata=run_metadata)
                    else:
                        batch_acc_v, summary_str = sess.run([batch_acc, summary_op])

                    print('%d batches are tested.' % step)
#                    if (i == num_batches_test % 500 ):
                    if(rundata ==1):
                        summary_writer.add_run_metadata(run_metadata, 'step%d' % step)

                    summary_writer.add_summary(summary_str, step)

                    accuracy_sum += batch_acc_v

                    step += 1

                ave_acc = accuracy_sum / num_batches_test
                print('the average accuracy is %f' % ave_acc)

            coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
