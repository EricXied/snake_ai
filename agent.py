import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


def load_model(model, file_name='model.pth'):
    model_folder_path = './model'
    model.load_state_dict(torch.load(os.path.join(model_folder_path, file_name)))
    model.eval()
    return model


class Agent:

    def __init__(self):
        self.n_game = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(15, 256, 3)
        # if os.path.exists('./model/model.pth'):
        #     self.model = load_model(self.model)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        tail = game.snake[-1]
        tail2 = game.snake[-2]
        distance = np.sqrt((head.x - tail.x) ** 2 + (head.y - tail.y) ** 2)

        distance_l = np.sqrt((point_l.x - tail2.x) ** 2 + (point_l.y - tail2.y) ** 2)
        distance_r = np.sqrt((point_r.x - tail2.x) ** 2 + (point_r.y - tail2.y) ** 2)
        distance_u = np.sqrt((point_u.x - tail2.x) ** 2 + (point_u.y - tail2.y) ** 2)
        distance_d = np.sqrt((point_d.x - tail2.x) ** 2 + (point_d.y - tail2.y) ** 2)

        state = [
            # Danger straight

            (dir_r and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_l)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d)),

            # Danger right
            (dir_r and game._is_collision(point_d)) or
            (dir_l and game._is_collision(point_u)) or
            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)),

            # Danger left
            (dir_r and game._is_collision(point_u)) or
            (dir_l and game._is_collision(point_d)) or
            (dir_u and game._is_collision(point_l)) or
            (dir_d and game._is_collision(point_r)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            distance > distance_l,
            distance > distance_r,
            distance > distance_u,
            distance > distance_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,

        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self, state, action, reward, next_state, done):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration/ exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)
        # get move
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # remeber
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_game += 1
            agent.train_long_memory(state_old, final_move, reward, state_new, done)
            if score > record:
                record = score
                agent.model.save()
            print('Game', agent.n_game, 'Score', score, 'Record', record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
