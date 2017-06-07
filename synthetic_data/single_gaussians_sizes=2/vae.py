import numpy as np
import tensorflow as tf
from IPython import embed
import sys

np.random.seed(0)
tf.set_random_seed(0)

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100,
                 reconstruct_cost="bernoulli"):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reconstruct_cost = reconstruct_cost # must be bernoulli or gaussian

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        self.y = tf.placeholder(tf.float32, [None,
            network_architecture["n_input"]]) # For denoising
        self.denoising = False

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()
        #init = tf.initialize_all_variables()

        self.saver = tf.train.Saver()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        # Initialize autoencode network weights and biases
        self.network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(self.network_weights["weights_recog"],
                                      self.network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        #n_z = self.network_architecture["n_z"]
        #eps = tf.random_normal((self.batch_size, n_z), 0, 1,
        #                       dtype=tf.float32)
        eps = tf.random_normal(tf.shape(self.z_log_sigma_sq), 0, 1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean, self.x_reconstr_sigma = \
                self._generator_network(self.network_weights["weights_gener"],
                                        self.network_weights["biases_gener"])


    def _initialize_weights(self, n_hidden_recog, n_hidden_gener,
                            n_input, n_z):
        all_weights = dict()

        # Recognition Network (Encoder)
        all_weights['weights_recog'] = {}
        all_weights['weights_recog']['hidden_layers'] = []
        recogW = all_weights['weights_recog']['hidden_layers']
        all_weights['biases_recog'] = {}
        all_weights['biases_recog']['hidden_layers'] = []
        recogB = all_weights['biases_recog']['hidden_layers']
        prevNodes = n_input
        for numNodes in n_hidden_recog:
            recogW.append(tf.Variable(xavier_init(prevNodes, numNodes)))
            recogB.append(tf.Variable(tf.zeros([numNodes], dtype=tf.float32)))
            prevNodes = numNodes
        all_weights['weights_recog']['out_mean'] = tf.Variable(xavier_init(n_hidden_recog[-1], n_z))
        all_weights['weights_recog']['out_log_sigma'] = tf.Variable(xavier_init(n_hidden_recog[-1], n_z))
        all_weights['biases_recog']['out_mean'] = tf.Variable(tf.zeros([n_z], dtype=tf.float32))
        all_weights['biases_recog']['out_log_sigma'] = tf.Variable(tf.zeros([n_z], dtype=tf.float32))

        # Generator Network (Decoder)
        all_weights['weights_gener'] = {}
        all_weights['weights_gener']['hidden_layers'] = []
        generW = all_weights['weights_gener']['hidden_layers']
        all_weights['biases_gener'] = {}
        all_weights['biases_gener']['hidden_layers'] = []
        generB = all_weights['biases_gener']['hidden_layers']
        prevNodes = n_z
        for layerNum,numNodes in enumerate(n_hidden_gener):
            generW.append(tf.Variable(xavier_init(prevNodes, numNodes)))
            generB.append(tf.Variable(tf.zeros([numNodes], dtype=tf.float32)))
            prevNodes = numNodes
        all_weights['weights_gener']['out_mean'] = tf.Variable(xavier_init(n_hidden_gener[-1], n_input))
        all_weights['weights_gener']['out_sigma'] = tf.Variable(xavier_init(n_hidden_gener[-1], n_input))
        all_weights['biases_gener']['out_mean'] = tf.Variable(tf.zeros([n_input], dtype=tf.float32))
        all_weights['biases_gener']['out_sigma'] = tf.Variable(tf.zeros([n_input], dtype=tf.float32))

        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        currentInput = [self.x]
        for layerNum,weight in enumerate(weights['hidden_layers']):
            currentInput.append(self.transfer_fct(tf.add(tf.matmul(currentInput[-1], weight),
                                            biases['hidden_layers'][layerNum])))
        #print "currentInput[-1].shape() = ", tf.shape(currentInput[-1])
        z_mean = tf.add(tf.matmul(currentInput[-1], weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(currentInput[-1], weights['out_log_sigma']),
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.

        currentInput = [self.z]
        for layerNum,weight in enumerate(weights['hidden_layers']):
            currentInput.append(self.transfer_fct(tf.add(tf.matmul(currentInput[-1], weight),
                                            biases['hidden_layers'][layerNum])))

        x_reconstr_mean = \
            tf.add(tf.matmul(currentInput[-1], weights['out_mean']),
                                 biases['out_mean'])
        x_reconstr_sigma = \
            tf.add(tf.matmul(currentInput[-1], weights['out_sigma']),
                   biases['out_sigma'])

        if self.reconstruct_cost == "bernoulli":
            return (tf.nn.sigmoid(x_reconstr_mean), x_reconstr_sigma)
        if self.reconstruct_cost == "gaussian":
            return (x_reconstr_mean, x_reconstr_sigma)

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        if self.reconstruct_cost == "bernoulli":
            self.reconstr_loss = \
                -tf.reduce_sum(self.y * tf.log(1e-10 + self.x_reconstr_mean)
                               + (1-self.y) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                               1)
        elif self.reconstruct_cost == "gaussian":
            #eps = tf.random_normal(tf.shape(self.x_reconstr_sigma), name="epsilon")
            #self.x_reconstruct = self.x_reconstr_mean + tf.multiply(self.x_reconstr_sigma, eps)
            #normed_x = tf.subtract(self.x_reconstr, self.x_reconstr_mean)
            #normed_x = tf.subtract(self.y, self.x_reconstr_mean)
            #reconstr_loss = \
            #    tf.reduce_sum(tf.log(tf.abs(self.x_reconstr_sigma)) + \
            #    0.5*tf.pow(tf.div(normed_x, self.x_reconstr_sigma), 2))
            self.reconstr_loss = tf.reduce_sum(tf.pow(tf.subtract(self.y,
                self.x_reconstr_mean), 2))
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(self.reconstr_loss + self.latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X, Y):
        """Train model based on mini-batch of input data.
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X, self.y: Y})

        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """

        if self.reconstruct_cost == "bernoulli":
            return self.sess.run(self.x_reconstr_mean,
                                 feed_dict={self.x: X})
        elif self.reconstruct_cost == "gaussian":
            return self.sess.run(self.x_reconstr_mean,
                                 feed_dict={self.x: X})

    def latentEntropy(self, X):
        """ Pass the entire dataset, X, through the network and keep
        track of the mean and standard deviation of each node in
        the latent space. At the end, use these to approximate the entropy of
        the posterior distribution.
        """
        z_sigma = tf.sqrt(tf.exp(self.z_log_sigma_sq), name="z_sigma")
        #dists = tf.contrib.distributions.MultivariateNormalDiag(self.z_mean, z_sigma)
        #epsilons = tf.random_normal(tf.shape(self.z_log_sigma_sq), name="epsilon")
        #samples = self.z_mean + epsilons*z_sigma_sq
        #samples = dists.sample()
        #print "samples.shape() = ", tf.shape(samples)
        #values = dists.pdf(samples)
        #print "values.shape() = ", tf.shape(values)
        #logValues = tf.log(values)
        #entropy = -tf.reduce_mean(logValues)
        entropy = (1+np.log(2*np.pi))/2.0 + tf.reduce_mean(tf.reduce_mean(z_sigma, 1))
        #s, v, lv, e = self.sess.run((samples, values, logValues, entropy), feed_dict={self.x: X})
        #embed()
        #sys.exit()
        return self.sess.run(entropy, feed_dict={self.x: X})

    def getLatentParams(self, X):

        z_sigma = tf.sqrt(tf.exp(self.z_log_sigma_sq))
        return self.sess.run((self.z_mean, z_sigma), feed_dict={self.x: X})

    def getReconParams(self, X):

        return self.sess.run((self.x_reconstr_mean, self.x_reconstr_sigma), feed_dict={self.x: X})
