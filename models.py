from layers import *
from metrics import *
import tensorflow as tf

from utils import _scale_l2

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        # print("output: " + str(self.outputs))
        # print("embedding: " + str(self.layers[0].embedding))
        # if FLAGS.adversarial_loss:
        #     self.loss += self._adversarial_loss() * tf.constant(FLAGS.adv_reg_coeff)

        if FLAGS.vat_loss:
            self.loss += self.virtual_adversarial_loss(self.inputs, self.outputs)

    def _adversarial_loss(self):
        """Adds gradient to embedding and recomputes classification loss."""
        grad, = tf.gradients(
            self.loss,
            self.layers[0].embedding,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)
        print("embedding: " + str(grad))
        perturb = _scale_l2(grad, FLAGS.perturb_norm_length)
        output = self.layers[1](self.layers[0].embedding + perturb)
        return masked_softmax_cross_entropy(output, self.placeholders['labels'],
                                            self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.pred = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            featureless=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x, #
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

    def generate_virtual_adversarial_perturbation(self, input, logit):
        # d = tf.random_normal(shape=(self.input_dim, FLAGS.hidden1))

        # for _ in range(FLAGS.vat_num_power_iterations):
        #     d = _scale_l2(d, FLAGS.vat_random_eps) # normalization
        #     logit_p = logit
        #     logit_m = self.layers[1](h1 + d)
        #     dist = kl_divergence_with_logit(logit_p, logit_m)
        #     grad = tf.gradients(dist, d, aggregation_method=2)[0]
        #     d = tf.stop_gradient(grad)

        # return FLAGS.vat_adv_eps * _scale_l2(d, 3.0)

        d = tf.random_normal(shape=(self.input_dim, self.input_dim))

        for _ in range(FLAGS.vat_num_power_iterations):
            d = _scale_l2(d, FLAGS.vat_random_eps) # normalization
            logit_p = logit
            self.layers[0].support[0] += d
            logit_m = self.layers[1](self.layers[0](input))
            dist = kl_divergence_with_logit(logit_p, logit_m)
            grad = tf.gradients(dist, self.layers[0].support[0], aggregation_method=2)[0]
            d = tf.stop_gradient(grad)
            self.layers[0].support[0] -= d

        return _scale_l2(d, FLAGS.vat_adv_eps)

    def virtual_adversarial_loss(self, input, logit):
        logit = tf.stop_gradient(logit)
        r_vadv = self.generate_virtual_adversarial_perturbation(input, logit)
        logit_p = logit
        self.layers[0].support[0] += r_vadv
        logit_m = self.layers[1](self.layers[0](input))
        loss = kl_divergence_with_logit(logit_p, logit_m)
        self.layers[0].support[0] -= r_vadv
        return loss



def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keepdims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keepdims=True))
    return lsm


def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp