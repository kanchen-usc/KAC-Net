from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

def msr_init(shape):
	n = 1.0
	for dim in shape:
		n *= float(dim)
	std = np.sqrt(2.0/n)
	init = tf.truncated_normal_initializer(mean=0.0, stddev=std)
	return init