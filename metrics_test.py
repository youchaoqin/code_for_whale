import tensorflow as tf
import numpy as np
import os
from cls_eval import make_metric_dict
import sklearn

if __name__ == '__main__':
    print('')
    """
    prob  = np.load('/home/westwell/Desktop/mode_training/0005_tiny_image_net/03_xception39/new_ckpt/eval/prob.npy')
    label = np.load('/home/westwell/Desktop/mode_training/0005_tiny_image_net/03_xception39/new_ckpt/eval/label.npy')
    
    eq = np.equal(np.argmax(prob,axis=-1),
                  np.argmax(label,axis=-1))
    print(np.mean(eq))

    with tf.Session() as sess:
        topk = sess.run(
            tf.to_float(tf.math.in_top_k(prob, np.argmax(label,axis=-1), 5)))
    print(np.mean(topk))
    """
    """
    pred=[[0.1, 0.2, 0.6,  0.05, 0.05],
          [0.7, 0.1, 0.05, 0.1,  0.05],
          [0.4, 0.4, 0.05, 0.1,  0.05],
          [0.3, 0.1, 0.1,  0.4,  0.1],
          [0.1, 0.5, 0.05, 0.3,  0.05],
          [0.2, 0.2, 0.05, 0.05, 0.5],
          ]

    label=[[0, 0, 1, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1],]

    pred = np.array(pred)
    label = np.array(label).astype(np.int)

    # tf
    prob = tf.placeholder(tf.float32, shape=[1, 5])
    lb = tf.placeholder(tf.int32, shape=[1, 5])

    # _, up_ops = tf.metrics.accuracy(
    #     labels=lb,
    #     predictions=prob,)

    metric_dict = make_metric_dict(prob=prob, label=lb,
                                   metric_opt={'eval_metrics': ['accuracy'],
                                               'acc_k': 2})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for i in range(6):
            print('')
            print("##")
            print(pred[i])
            print(label[i])
            md = sess.run(metric_dict, feed_dict={prob: np.expand_dims(pred[i],0),
                                                  lb: np.expand_dims(label[i],0)})
            for k in md:
                print('%s: %s'%(k, md[k]))
    """