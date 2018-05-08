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
import model as net
import tensorflow.contrib.slim as slim
from tensorflow.core.profiler import tfprof_log_pb2
import cudaprofile
from tensorflow.python.profiler import option_builder

import logging
import daiquiri
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

warnings.filterwarnings("ignore")
tfprof = 0
rundata =0

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
    for d in ['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']:
	with tf.Graph().as_default(),tf.device(d):
	    num_batches_per_epoch_train = int(dataset_size_train / cfg.batch_size)
	    num_batches_test = int(dataset_size_test / cfg.batch_size * 0.1)
    #        num_batches_test = 10


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
		    if (model_name == 'cnn_baseline'):
			if(dataset_name == 'smallNORB'):
			    saver = tf.train.import_meta_graph(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name)+"model.ckpt-46800.meta")
			else:
			    saver = tf.train.import_meta_graph(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name)+"model.ckpt-46800.meta")

		    else:
			if (dataset_name == 'mnist'):
			    saver = tf.train.import_meta_graph(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name)+"model-0.0630.ckpt-55000.meta")
			elif (dataset_name == 'fashion_mnist'):
			    saver = tf.train.import_meta_graph(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name)+"model-55000.ckpt-55000.meta")
			else:
			    saver = tf.train.import_meta_graph(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name)+"model.ckpt.meta")

		    saver.restore(sess,tf.train.latest_checkpoint(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name)))
		    accuracy_sum = 0
		    profiler = tf.profiler.Profiler(sess.graph)

		    for i in range(num_batches_test):
			if (tfprof):
			    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			    run_metadata = tf.RunMetadata()
			    batch_acc_v, summary_str = sess.run([batch_acc, summary_op], options=run_options,run_metadata=run_metadata)
			    profiler.add_step(i,run_metadata)
    #                        options = tf.profiler.ProfileOptionBuilder.time_and_memory()
    #                        options["min_bytes"] = 0
    #                        options["select"] = ("bytes", "peak_bytes", "output_bytes",
    #                                                        "residual_bytes")
    #                        options["output"] = "file"
    #                        profiler.add_step(1, run_metadata)
    #                        profiler.advise(options=options)
    #                        tf.profiler.advise(sess.graph, run_meta=run_metadata)
			    if(model_name == 'caps'):
				opts = (option_builder.ProfileOptionBuilder(
					   option_builder.ProfileOptionBuilder.time_and_memory())
					    .with_step(i)
					    .with_timeline_output("/data/omran/chrome_b1").build())
			    else:
				opts = (option_builder.ProfileOptionBuilder(
					   option_builder.ProfileOptionBuilder.time_and_memory())
					    .with_step(i)
					    .with_timeline_output("/data/omran/chrome_cnn_b1").build())


			    profiler.profile_graph(options=opts)
	       #             tf.profiler.profile(tf.get_default_graph(), run_meta=run_metadata, cmd="scope",  options=options)
			elif (rundata == 1):
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
