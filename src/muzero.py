from src.config import *
from src.networks.shared_storage import SharedStorage
from src.self_play.self_play import run_selfplay, run_eval
from src.training.replay_buffer import ReplayBuffer
from src.training.training import train_network


def muzero(config: MuZeroConfig):
    storage = SharedStorage(config.new_network(), config.uniform_network(), config.new_optimizer())
    replay_buffer = ReplayBuffer(config)

    for loop in range(config.nb_training_loop):
        print("Training loop", loop)
        score_train = run_selfplay(config, storage, replay_buffer, config.nb_episodes)
        train_network(config, storage, replay_buffer, config.nb_epochs)

        print("Train score:", score_train)
        print("Eval score:", run_eval(config, storage, 50))
        print(f"MuZero played {config.nb_episodes * (loop + 1)} "
              f"episodes and trained for {config.nb_epochs * (loop + 1)} epochs.\n")

    storage.save_network_dir(config.nb_training_loop)

    return storage.latest_network()


def muzero_load_train(config: MuZeroConfig, step: int):
    storage = SharedStorage(config.new_network(), config.uniform_network(), config.new_optimizer())
    replay_buffer = ReplayBuffer(config)
    storage.load_network_dir(step)
    
    for loop in range(config.nb_training_loop):
        print("Training loop", loop)
        score_train = run_selfplay(config, storage, replay_buffer, config.nb_episodes)
        train_network(config, storage, replay_buffer, config.nb_epochs)

        print("Train score:", score_train)
        print("Eval score:", run_eval(config, storage, 50))
        print(f"MuZero played {config.nb_episodes * (loop + 1)} "
              f"episodes and trained for {config.nb_epochs * (loop + 1)} epochs.\n")

    # storage.save_network_dir(config.nb_training_loop + step)

    return storage.latest_network()


def muzero_recording(config: MuZeroConfig, step: int):
    storage = SharedStorage(config.new_network(), config.uniform_network(), config.new_optimizer())
    storage.load_network_dir(step)
    print("Eval score:", run_eval(config, storage, 50))

    storage.save_network_dir(config.nb_training_loop)