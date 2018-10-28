import tensorflow as tf

class Alexnet(object):
    def __init__(self,x,keep_prob,num_classes,skip_layer,is_training,weights_path = 'DEFAULT'):
        self.X =x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.is_training = is_training

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        self.create()

    def create(self):

        conv1 = conv(self.X,11,11,96,4,4,paddding='VALID',name='conv1')
        norm1 = lrn(conv1,2,1e-05,0.75,name='norm1')
        pool1 = max_pool(norm1,3,3,2,2,padding='VALID',name='pool1')

        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        conv3 = conv(pool2,3,3,384,1,1,name='conv3')

        conv4 = conv(conv3,3,3,384,1,1,name='conv4')

        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        flattened = tf.reshape(pool5,[-1,6*6*256])

        fc6 = fc(flattened,6*6*256,4096,name='fc6')
        dropout6 = dropout(fc6,self.KEEP_PROB)

        fc7 = fc(dropout6,4096,4096,name='fc7')
        dropout7 = dropout(fc7,self.KEEP_PROB)
        self.fc8 = fc(dropout7,4096,self.NUM_CLASSES,name='fc8',relu=False)


        pass
    def load_init_weights(self):
        pass

    




def conv(x, filter_height,filter_width,num_filters,stride_y,stride_x,name,paddding = 'SAME',groups = 1):
    input_channels = int(x.get_shape().as_list()[-1])

    convolve = lambda i,k:tf.nn.conv2d(i,k,strides = [1,stride_y,stride_x,1],paddding = paddding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights',[filter_height,filter_width], input_channels/groups ,num_filters)
        biases = tf.get_variable('bias',[num_filters])

        if groups == 1:
            conv = convolve(x,weights)
        else:
            input_groups = tf.split(axis = 3,num_or_size_splits = groups,value = x)
            weights_group = tf.split(axis = 3,num_or_sie_splits = groups,value = weights)
            output_group = [convolve(i,k) for i,k in zip(input_groups,weights_group)]


            conv = tf.concat(axis = 3,value = output_group)

        bias = tf.reshape(tf.nn.bias_add(conv,biases),conv.get_shape().as_list())
        relu = tf.nn.relu(bias,name = scope.name)

        return relu




def fc(x,num_in,num_out,name,relu = True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights',shape = [num_in,num_out],trainable = True)
        biases = tf.get_variable('biases',shape = [num_out],trainable = True)

        act = tf.nn.xw_plus_b(x,weights,biases,name = scope.name)

        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act



def max_pool(x,filter_height,filter_width,stride_x,stride_y,name,padding='SAME'):
    return tf.nn.max_pooling(x,ksize = [1,filter_height,filter_width,1],stride = [1,stride_y,stride_x,1],padding = padding,name = name)

def lrn(x,radius,alpha,beta,name,bias = 1.0):
    return tf.nn.local_response_normalization(x,depth_radius = radius,alpha = alpha,beta = beta,bias = bias,name = name)

def dropout(x,keep_prob):
    return tf.nn.dropout(x,keep_prob)























