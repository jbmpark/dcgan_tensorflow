import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import random
import scipy

gpu=3   #  1: use_bias True,  2 : use_bias Fals
#################################
# parameters
width  = 64
height = 64
channel = 3
epoch = 100
latent_len = 100
data_dir = '../data/celebA'
sample_dir = './samples_%d'%gpu
checkpoint_dir = './checkpoint_%d'%gpu
conv_kernel_size = 5
kernel_init = tf.truncated_normal_initializer(stddev=0.02)
#kernel_init=None
use_bias=False
padding='same'
bn_momentum=0.9
bn_eps = 0.00001
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)



global_step = tf.Variable(0, trainable=False, name='global_step')

################################# 
# input
Z = tf.placeholder(tf.float32, (None, latent_len), name='noise')
X = tf.placeholder(tf.float32, (None, height, width, channel), name='input')
PHASE = tf.placeholder(tf.bool, name='is_training')



#################################
# generator
#with tf.name_scope('Generator'):
def G(noise, is_training):
    with tf.variable_scope('G_variables'):
        latent = tf.reshape(noise, [-1, 1, 1, latent_len])
        l1 = tf.layers.conv2d_transpose(latent, 1024, 4, activation=None , use_bias=use_bias, kernel_initializer=kernel_init)
        #l1 = tf.layers.dense(latent, 4*4*1024)
        #l1 = tf.reshape(l1, [-1, 4, 4, 1024])   # 4x4x1024
        l1 = tf.layers.batch_normalization(l1, training=is_training, momentum=bn_momentum, epsilon=bn_eps)
        l1 = tf.nn.relu(l1)
        l2 = tf.layers.conv2d_transpose(l1, 512, conv_kernel_size, strides=(2,2), padding=padding, activation=None , use_bias=use_bias, kernel_initializer=kernel_init)
        l2 = tf.layers.batch_normalization(l2, training=is_training, momentum=bn_momentum, epsilon=bn_eps)
        l2 = tf.nn.relu(l2) # 8x8x512
        l3 = tf.layers.conv2d_transpose(l2, 256, conv_kernel_size, strides=(2,2), padding=padding, activation=None , use_bias=use_bias, kernel_initializer=kernel_init)
        l3 = tf.layers.batch_normalization(l3, training=is_training, momentum=bn_momentum, epsilon=bn_eps)
        l3 = tf.nn.relu(l3) # 16x16x256
        l4 = tf.layers.conv2d_transpose(l3, 128, conv_kernel_size, strides=(2,2), padding=padding, activation=None , use_bias=use_bias, kernel_initializer=kernel_init)
        l4 = tf.layers.batch_normalization(l4, training=is_training, momentum=bn_momentum, epsilon=bn_eps)
        l4 = tf.nn.relu(l4) # 32x32x512
        l5 = tf.layers.conv2d_transpose(l4, 3,   conv_kernel_size, strides=(2,2), padding=padding, activation=None , use_bias=True, kernel_initializer=kernel_init)
        #l5 = tf.layers.batch_normalization(l5, training=is_training, momentum=bn_momentum, epsilon=bn_eps)
        l5 = tf.nn.tanh(l5) # 64x64x3
        
    return l5



#################################
# discriminator
#with tf.name_scope('Discriminator'):
def D(input_image, is_training, reuse=False):
    with tf.variable_scope('D_variables') as scope:
        if reuse:
            scope.reuse_variables()
        input = input_image    # 64x64x3
        l1 = tf.layers.conv2d(input, 128, conv_kernel_size, strides=(2,2), padding=padding, activation=None, use_bias=True, kernel_initializer=kernel_init)
        #l1 = tf.layers.batch_normalization(l1, training=is_training, momentum=bn_momentum, epsilon=bn_eps)    #32x32x128
        l1 = tf.nn.leaky_relu(l1)
        l2 = tf.layers.conv2d(l1,    256, conv_kernel_size, strides=(2,2), padding=padding, activation=None, use_bias=use_bias, kernel_initializer=kernel_init)
        l2 = tf.layers.batch_normalization(l2, training=is_training, momentum=bn_momentum, epsilon=bn_eps)    #16x16x256
        l2 = tf.nn.leaky_relu(l2)
        l3 = tf.layers.conv2d(l2,    512, conv_kernel_size, strides=(2,2), padding=padding, activation=None, use_bias=use_bias, kernel_initializer=kernel_init)
        l3 = tf.layers.batch_normalization(l3, training=is_training, momentum=bn_momentum, epsilon=bn_eps)    #8x8x512
        l3 = tf.nn.leaky_relu(l3)
        l4 = tf.layers.conv2d(l3,    1024, conv_kernel_size, strides=(2,2), padding=padding, activation=None, use_bias=use_bias, kernel_initializer=kernel_init)
        l4 = tf.layers.batch_normalization(l4, training=is_training, momentum=bn_momentum, epsilon=bn_eps)    #4x4x1024
        l4 = tf.nn.leaky_relu(l4)
        l5 = tf.layers.flatten(l4)
        #l5 = tf.layers.dense(l5,     256,  activation=None, use_bias=True, kernel_initializer=kernel_init)
        #l5 = tf.layers.batch_normalization(l5, training=is_training, momentum=bn_momentum, epsilon=bn_eps)    
        #l5 = tf.nn.leaky_relu(l5)
        l6 = tf.layers.dense(l5,     1,    activation=None, use_bias=True, kernel_initializer=kernel_init)
            
    return l6


#with tf.name_scope('Optimizer'):

#################################
# gan network

with tf.device("/device:GPU:%d"%gpu):   # Wanna run on specified GPU only
    G = G(Z, PHASE)
    D_real = D(X, PHASE)
    D_fake = D(G, PHASE, reuse=True)

    #################################
    # train step
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

    D_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_variables')
    G_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_variables')

    # Two lines below are for updating 'moving_average/moving_variance' of batch norm layers, 
    # see : https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        D_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(D_loss, var_list=D_var_list, global_step = global_step)
        G_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(G_loss, var_list=G_var_list, global_step = global_step)


#tensorboard
tf.summary.scalar('D_Loss', D_loss)
tf.summary.scalar('G_Loss', G_loss)


#################################
# miscs
def get_batch_file_list(n):
    start = n*batch_size
    return data_file_list[start:(start+batch_size)]

def img_squeeze(img):   # 0~255 -->  -1 ~ 1
    return ((img*2.0)/256.0) -1.

def img_recover(img):
    img =((img+1.)*256.0)/2.0
    return img.astype(int)

def read_image(file, scale_w, scale_h, crop=True):
    img = scipy.misc.imread(file, mode='RGB').astype(np.float)
    if crop:
        ysize, xsize, chan = img.shape
        xoff = (xsize - 128) // 2
        yoff = (ysize - 128) // 2
        img= img[yoff:-yoff,xoff:-xoff]

    img = scipy.misc.imresize(img, [scale_w, scale_h])
    img = img_squeeze(img)
    return img

def read_batch(batch_file_list):
    return [read_image(file , width, height) for file in batch_file_list]

def gen_noise(batch, noise_len):
    return np.random.uniform(-1.,1.,size=(batch, noise_len))


#################################
# training

# data 
batch_size = 128
data_file_list = glob.glob(os.path.join(data_dir, '*.jpg'))
num_data = len(data_file_list)
num_batch = int(num_data/batch_size)

test_size = 10  #num images to test
test_noise = gen_noise(test_size, latent_len)

#session
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
gpu_options = tf.GPUOptions(allow_growth=True)  # Without this, the process occupies whole area of memory in the all GPUs.
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

#if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#    saver.restore(sess, ckpt.model_checkpoint_path)
#else:
#    sess.run(tf.global_variables_initializer())
sess.run(tf.global_variables_initializer())

#tensorboard
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs_%d'%gpu, sess.graph)


#These are just for debugging
#epoch =1
#num_batch=10
 
for e in range(epoch):  
    random.shuffle(data_file_list)  #to avoid overfitting
    for b in range(num_batch):
        this_batch = get_batch_file_list(b)
        #print('File : ', this_batch)
        x_input = read_batch(this_batch)
        z_input = gen_noise(batch_size, latent_len)

        #train 
        is_training=True
        _, D_batch_loss = sess.run([D_train, D_loss], feed_dict={Z:z_input, X:x_input, PHASE:is_training})
        _, G_batch_loss = sess.run([G_train, G_loss], feed_dict={Z:z_input, X:x_input, PHASE:is_training})

        print("epoch: %04d"%e, "batch: %05d"%b, "D_loss: {:.04}".format(D_batch_loss),"G_loss: {:.04}".format(G_batch_loss) )
        
        
        #save input X
        save_input=False
        if save_input:
            if not os.path.exists('input'):
                os.makedirs('input')
            fig, ax = plt.subplots(1, 10, figsize=(10,1))
            for k in range(10):
                ax[k].set_axis_off()
                ax[k].imshow(((x_input[k]+1.0)/2.0))
            plt.savefig('./input/input_{}'.format(str(e).zfill(3)) + '_{}.png'.format(str(b).zfill(5)), bbox_inches='tight')
            plt.savefig('samples/x.png', bbox_inches='tight')
            plt.close(fig)
        
        # testing
        if not b%100: 
            is_training=False
            samples = sess.run(G, feed_dict={Z:test_noise, PHASE:is_training})
            samples = (samples+1.0)/2.0
            #print (samples)  
            fig, ax = plt.subplots(1, test_size, figsize=(test_size,1))
            for k in range(test_size):
                ax[k].set_axis_off()
                ax[k].imshow(samples[k])
            plt.savefig(sample_dir+'/dcgan_{}'.format(str(e).zfill(3)) + '_{}.png'.format(str(b).zfill(5)), bbox_inches='tight')
            plt.close(fig)
            
    saver.save(sess, checkpoint_dir+'/dcgan.ckpt', global_step=global_step)

    #tensorboard
    is_training=False
    summary = sess.run(merged, feed_dict={Z:z_input, X:x_input, PHASE:is_training})
    writer.add_summary(summary, global_step=sess.run(global_step))


