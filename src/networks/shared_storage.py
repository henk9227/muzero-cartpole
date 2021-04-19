import tensorflow as tf

from src.networks.network import *


class SharedStorage(object):
    """Save the different versions of the network."""

    def __init__(self, network: BaseNetwork, uniform_network: UniformNetwork, optimizer: tf.keras.optimizers):
        self._networks = {}
        self.current_network = network
        self.uniform_network = uniform_network
        self.optimizer = optimizer

    def latest_network(self) -> AbstractNetwork:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return self.uniform_network

    def save_network(self, step: int, network: BaseNetwork):
        self._networks[step] = network
    
    def save_network_dir(self, step: int):
        self.current_network.representation_network.save('./model/trainstep_{}/representation'.format(step))
        self.current_network.value_network.save('./model/trainstep_{}/value'.format(step))
        self.current_network.policy_network.save('./model/trainstep_{}/policy'.format(step))
        self.current_network.dynamic_network.save('./model/trainstep_{}/dynamic'.format(step))
        self.current_network.reward_network.save('./model/trainstep_{}/reward'.format(step))

        self.current_network.initial_model.save('./model/trainstep_{}/initial'.format(step))
        self.current_network.recurrent_model.save('./model/trainstep_{}/recurrent'.format(step))

    def load_network_dir(self, step: int):
        self.current_network.representation_network = tf.keras.models.load_model('./model/trainstep_{}/representation'.format(step))
        self.current_network.value_network = tf.keras.models.load_model('./model/trainstep_{}/value'.format(step))
        self.current_network.policy_network = tf.keras.models.load_model('./model/trainstep_{}/policy'.format(step))
        self.current_network.dynamic_network = tf.keras.models.load_model('./model/trainstep_{}/dynamic'.format(step))
        self.current_network.reward_network = tf.keras.models.load_model('./model/trainstep_{}/reward'.format(step))

        self.current_network.initial_model = InitialModel(self.current_network.representation_network,
                                                          self.current_network.value_network,
                                                          self.current_network.policy_network)
        self.current_network.recurrent_model = RecurrentModel(self.current_network.dynamic_network,
                                                              self.current_network.reward_network,
                                                              self.current_network.value_network,
                                                              self.current_network.policy_network)

        self.save_network(step, self.current_network)   # 경우에 따라 step 수정 
