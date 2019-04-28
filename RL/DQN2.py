'''
    Deep Q Network for flappybird
    using TF 2.0 alpha
'''
import numpy as np
import tensorflow as tf
import cv2
import sys
import matplotlib.pylab as plt
import argparse

sys.path.append('game')
import wrapped_flappy_bird as env

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=2000000)
parser.add_argument('--check', type=str, default='model/dqn2/')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--mode', type=str, default='train')

params = parser.parse_args()
np.random.seed(1)
tf.random.set_seed(1)

STDDEV = 0.01
CONST = 0.01
WIDTH, HEIGHT = 80, 80
ACTION_DIM = 2
ACTION = np.array([[1, 0], [0, 1]], dtype=np.int32)
INITIAL_EPSILON = 0.2
EPSILON = 1.0
ALPHA = 0.1
LAMBDA = 0.99
OBSERVE = 100
EXPLORE = 10000
MEMORY_SIZE = 20000


def convert(S):
    S = cv2.cvtColor(cv2.resize(S, (HEIGHT, WIDTH)), cv2.COLOR_BGR2GRAY)
    _, S = cv2.threshold(S, 1, 255, cv2.THRESH_BINARY)
    return S / 255.


class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.dqn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 8, 4, 'same', activation='relu',
                                   kernel_initializer=tf.initializers.TruncatedNormal(stddev=STDDEV),
                                   bias_initializer=tf.initializers.Constant(CONST)
                                   ),
            tf.keras.layers.MaxPool2D(2, 2, 'same'),
            tf.keras.layers.Conv2D(64, 4, 2, 'same', activation='relu',
                                   kernel_initializer=tf.initializers.TruncatedNormal(stddev=STDDEV),
                                   bias_initializer=tf.initializers.Constant(CONST)
                                   ),
            tf.keras.layers.Conv2D(64, 3, 1, 'same', activation='relu',
                                   kernel_initializer=tf.initializers.TruncatedNormal(stddev=STDDEV),
                                   bias_initializer=tf.initializers.Constant(CONST)
                                   ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu',
                                  kernel_initializer=tf.initializers.TruncatedNormal(stddev=STDDEV),
                                  bias_initializer=tf.initializers.Constant(CONST)
                                  ),
            tf.keras.layers.Dense(ACTION_DIM,
                                  kernel_initializer=tf.initializers.TruncatedNormal(stddev=STDDEV),
                                  bias_initializer=tf.initializers.Constant(CONST)
                                  )
        ])

    def __call__(self, s):
        return self.dqn(s)


class QLAgent():
    def __init__(self):
        if params.mode == 'train':
            self.epsilon = INITIAL_EPSILON
            self.epsilon_increment = (EPSILON - INITIAL_EPSILON) / EXPLORE
            self.store = []

            self.cost_hist = []
            self.Reward = []

    def choose_action(self, state, model):
        if np.random.uniform() <= self.epsilon:
            state = tf.reshape(state, (1, HEIGHT, WIDTH, 4))
            action_value = model(state)[0]
            action = tf.argmax(action_value, output_type=tf.int32)
        else:
            action = tf.constant(0, dtype=tf.int32)
        return action

    def epsilon_change(self, t):
        if t >= OBSERVE:
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < EPSILON else EPSILON

    def choose_action_deterministic(self, state, model):
        state = tf.reshape(state, (1, HEIGHT, WIDTH, 4))
        action_value = model(state)[0]
        action = tf.argmax(action_value)
        return action

    def store_transition(self, S, A, R, S_next, is_terminal):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = [S, A, R, S_next, is_terminal]
        self.store.append(transition)
        self.memory_counter += 1
        if self.memory_counter > MEMORY_SIZE:
            self.store.pop(0)

    def learn(self, model, optim):
        index = np.random.permutation(np.minimum(self.memory_counter, MEMORY_SIZE))[:params.batch_size]
        batch_s = np.array([self.store[id][0] for id in index])
        batch_a = np.array([self.store[id][1] for id in index])
        batch_r = np.array([self.store[id][2] for id in index])
        batch_s_next = np.array([self.store[id][3] for id in index])
        batch_is_terminal = np.array([self.store[id][4] for id in index])

        self.cost_hist.append(train_step(batch_s, batch_a, batch_r, batch_s_next, batch_is_terminal, model, optim))

    def plot_cost_reward(self):
        plt.subplot(131)
        plt.plot(self.cost_hist)
        plt.title('cost')
        plt.subplot(132)
        plt.plot(self.Reward)
        plt.title('Reward')
        plt.subplot(133)
        self.average_reward = [np.mean(self.Reward[i * 10:(i + 1) * 10]) for i in range(len(self.Reward) // 10)]
        plt.plot(self.average_reward)
        plt.title('Average Reward')
        plt.show()


@tf.function
def train_step(s, a, r, s_next, is_terminal, model, optim):
    with tf.GradientTape() as tape:
        q_eval = tf.reduce_sum(tf.multiply(model(s), tf.cast(a, tf.float64)), axis=-1)
        q_next = model(s_next)
        q_target = tf.where(is_terminal, r, r + LAMBDA * tf.reduce_max(q_next, axis=-1))

        loss = tf.reduce_mean(tf.math.squared_difference(q_target, q_eval))

    trainable_variables = model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optim.apply_gradients(zip(gradients, trainable_variables))

    return loss


class USR():
    @staticmethod
    def train():
        model = DQN()
        agent = QLAgent()
        optimizer = tf.keras.optimizers.Adam(learning_rate=params.lr)

        checkpoint_dir = params.check
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=model
                                         )
        manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_dir, max_to_keep=1)

        step = 0
        counter_max = 0

        for epoch in range(1, params.epochs + 1):
            step_counter = 0
            game_state = env.GameState()
            S, R, is_terminal = game_state.frame_step(ACTION[0])
            S = convert(S)
            S = np.stack((S, S, S, S), axis=2)

            while not is_terminal:
                A = agent.choose_action(S, model)
                S_next, R, is_terminal = game_state.frame_step(ACTION[A])

                S_next = np.append(np.expand_dims(convert(S_next), 2), S[:, :, :3], axis=2)

                agent.store_transition(S, ACTION[A], R, S_next, is_terminal)

                S = S_next
                step_counter += 1

                step += 1
                agent.epsilon_change(step)

            print('Epoch %d/%d Reward:%d Epsilon:%f' % (epoch, params.epochs, step_counter, agent.epsilon))
            agent.Reward.append(step_counter)
            if step_counter > counter_max and agent.epsilon == 1.0:
                counter_max = step_counter
                manager.save()
                print('model saved successfully!')
            if step > OBSERVE:
                for l in range(min(step_counter, 50)):
                    agent.learn(model, optimizer)

        agent.plot_cost_reward()

    @staticmethod
    def play():
        model = DQN()
        agent = QLAgent()

        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint_dir = params.check
        manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_dir, max_to_keep=1)
        checkpoint.restore(manager.latest_checkpoint)

        for test_step in range(1, 1001):
            step_counter = 0
            game_state = env.GameState()
            S, _, is_terminal = game_state.frame_step(ACTION[0])
            S = convert(S)
            S = np.stack((S, S, S, S), axis=2)

            while not is_terminal:
                A = agent.choose_action_deterministic(S, model)
                S_next, R, is_terminal = game_state.frame_step(ACTION[A])

                S_next = np.append(np.expand_dims(convert(S_next), 2), S[:, :, :3], axis=2)

                S = S_next
                step_counter += 1
            print('Test %d Reward:%d' % (test_step, step_counter))


def main():
    usr = USR()

    if params.mode == 'train':
        usr.train()
    elif params.mode == 'play':
        usr.play()


if __name__ == '__main__':
    main()
