import collections
from typing import Optional, Dict
import tensorflow as tf

from src.game.cartpole import CartPole
from src.game.game import AbstractGame
from src.networks.cartpole_network import CartPoleNetwork
from src.networks.network import BaseNetwork, UniformNetwork

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MuZeroConfig(object):

  def __init__(self,
               game,
               nb_training_loop: int,
               nb_episodes: int,
               nb_epochs: int,
               network_args: Dict,
               network,
               action_space_size: int,
               max_moves: int,
               discount: float,
               dirichlet_alpha: float,
               num_simulations: int,
               batch_size: int,
               td_steps: int,
               # num_actors: int,
               # lr_init: float,
               # lr_decay_steps: float,
               visit_softmax_temperature_fn,
               lr: float,
               known_bounds: Optional[KnownBounds] = None):
    ### Environment
    self.game = game

    ### Self-Play
    self.action_space_size = action_space_size
    # self.num_actors = num_actors

    self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
    self.max_moves = max_moves
    self.num_simulations = num_simulations
    self.discount = discount

    # Root prior exploration noise.
    self.root_dirichlet_alpha = dirichlet_alpha
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    self.known_bounds = known_bounds

    ### Training
    self.nb_training_loop = nb_training_loop
    self.nb_episodes = nb_episodes  # Nb of episodes per training loop
    self.nb_epochs = nb_epochs      # Nb of epochs per training loop

    # self.training_steps = int(1000e3)
    # self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = batch_size
    self.num_unroll_steps = 5
    self.td_steps = td_steps

    self.weight_decay = 1e-4
    self.momentum = 0.9

    self.network_args = network_args
    self.network = network
    self.lr = lr

    # Exponential learning rate schedule
    # self.lr_init = lr_init
    # self.lr_decay_rate = 0.1
    # self.lr_decay_steps = lr_decay_steps

  def new_game(self) -> AbstractGame:
    # return Game(self.action_space_size, self.discount)
    return self.game(self.discount)

  def new_network(self) -> BaseNetwork:
      return self.network(**self.network_args)

  def uniform_network(self) -> UniformNetwork:
      return UniformNetwork(self.action_space_size)

  def new_optimizer(self) -> tf.keras.optimizers:
      return tf.keras.optimizers.SGD(lr=self.lr, momentum=self.momentum)


def make_cartpole_config() -> MuZeroConfig:

  def visit_softmax_temperature(num_moves, training_steps): # num_moves, training_steps?
    # if num_moves < 30:
      return 1.0
    # else:
    #   return 0.0  # Play according to the max.

  return MuZeroConfig(
      game=CartPole,
      nb_training_loop=50,
      nb_episodes=20,
      nb_epochs=20,
      network_args={'action_size': 2,
                    'state_size': 4,
                    'representation_size': 4,
                    'max_value': 500},
      network=CartPoleNetwork,
      action_space_size=2,
      max_moves=1000,
      discount=0.99,
      dirichlet_alpha=0.25,
      num_simulations=11,   # Odd number perform better in eval mode
      batch_size=512,
      td_steps=10,
      visit_softmax_temperature_fn=visit_softmax_temperature,
      lr=0.05)