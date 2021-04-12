import csv
import os
import random
import shutil

import numpy as np
import pygame
from dataclasses import dataclass
from GameObjects import Ball, AIPlayer_v1, HumanPlayer, Opponent, RandomPlayer, AIPlayer_v2
from Global import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from models_pytorch import train_pytorch
from data_loader import load
from train import train_model

PLAYER_SPEED = 1
BALL_SPEED = 2
NOISE_PERCENTAGE = 0


@dataclass
class Colors:
    light_gray = (200, 200, 200)
    dark_gray = pygame.Color('grey12')


class Game:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SCALED)
        pygame.display.set_caption('Pong')

        self.opponent = RandomPlayer(x=0, y=SCREEN_HEIGHT / 2, speed=PLAYER_SPEED)
        self.ai = AIPlayer_v2(SCREEN_WIDTH - 1, SCREEN_HEIGHT / 2, speed=PLAYER_SPEED)
        self.ai.load_model_snapshot('./pong_net_y_v0.mdl', self.ai.net_y.name)
        self.ai.load_model_snapshot('./pong_net_x_v0.mdl', self.ai.net_x.name) #only for AIPlayer_v2
        ball_collider_group = pygame.sprite.Group()
        ball_collider_group.add(self.opponent, self.ai)
        self.ball = Ball(x=SCREEN_WIDTH / 2, y=SCREEN_HEIGHT / 2, speed=BALL_SPEED, collider_group=ball_collider_group)

        self.draw_group = pygame.sprite.Group()
        self.draw_group.add(self.opponent, self.ai, self.ball)

        self.round = 1
        self.scores = {'Opponent': 0, 'AI': 0}
        self.previous_ballx = self.ball.rect.x
        self.previous_bally = self.ball.rect.y

    def update(self):
        screen = pygame.surfarray.pixels3d(self.screen)
        meta = {'prev_ballx': self.previous_ballx,
                'prev_bally': self.previous_bally,
                'ballx': self.ball.rect.x,
                'bally': self.ball.rect.y}

        self.ai.update(screen, meta)
        self.opponent.update(screen, meta)
        self.previous_ballx = self.ball.rect.x
        self.previous_bally = self.ball.rect.y
        self.ball.update(screen, None)

        if self.ball.rect.right > SCREEN_WIDTH or self.ball.rect.left < 0:
            if self.ball.rect.right > SCREEN_WIDTH:
                self.scores['Opponent'] += 1
            else:
                self.scores['AI'] += 1
            self.round += 1
            print(self.scores)
            self.ball.reset()
            self.ai.reset()
            self.opponent.reset()


    def draw(self):
        def apply_noise():
            pixels = pygame.surfarray.array3d(self.screen) / 255
            noise = np.random.uniform(low=0, high=1, size=pixels.shape)
            noise_filter = np.random.uniform(low=0, high=1, size=SCREEN_WIDTH * SCREEN_HEIGHT)
            noise_filter = noise_filter < NOISE_PERCENTAGE
            for col in range(0, SCREEN_WIDTH):
                for row in range(0, SCREEN_HEIGHT):
                    idx = col * SCREEN_HEIGHT + row
                    if noise_filter[idx]:
                        pixels[col][row] = noise[col][row]
            pixels = pixels * 255
            pygame.surfarray.blit_array(self.screen, pixels)

        self.screen.fill(Colors.dark_gray)
        self.draw_group.draw(self.screen)
        apply_noise()
        pygame.display.flip()
        self.clock.tick(FPS)


    def run(self, rounds=1):
        while self.round <= rounds:
            self.draw()
            self.update()
        print(f"Results after {rounds} rounds:")
        print(f"Opponent: {self.scores['Opponent']} vs AI: {self.scores['AI']}")

    def generate_data(self, data_dir_output, metadata_file_output, append=False, start_rounds=0, end_rounds=1000):
        """Generates frames representing different game-states. The saved meta-data contains the:
                1. id of the image
                2. x and y coordinates of the human
                3. x and y coordinates of the ai
                4. x and y coordinates of the ball
            **WIPES THE ENTIRE DATA DIRECTORY IF NOT APPENDING**.
            Estimate roughly 50 seconds for 1000 images"""

        def build_data_dir_tree():
            os.mkdir(data_dir_output)
            os.mkdir(data_dir_output + "/" + img_dir_name)
            os.mkdir(data_dir_output + "/sorted_y")
            os.mkdir(data_dir_output + "/sorted_x")
            os.mkdir(data_dir_output + "/sorted_pytorch")

        img_dir_name = 'generated'
        if not os.path.exists(data_dir_output):
            build_data_dir_tree()
        if not append:
            shutil.rmtree(data_dir_output)
            build_data_dir_tree()
        meta_data = list()

        for i in range(start_rounds, end_rounds):
            # set player and ai position
            self.opponent.rect.centery = random.randrange(1, SCREEN_HEIGHT - 1)
            self.ai.rect.centery = random.randrange(1, SCREEN_HEIGHT - 1)
            # set ball position
            self.ball.rect.center = (random.randrange(1, SCREEN_WIDTH - 1),
                                     random.randrange(0, SCREEN_HEIGHT))  # 0, SCREEN_HEIGHT

            self.draw()
            img_name = str(i) + '.png'
            img_path = os.path.join(data_dir_output, img_dir_name, img_name)
            pygame.image.save(self.screen, img_path)
            meta_data.append({"img_id": img_name,
                              "humanx": self.opponent.rect.x,
                              "humany": self.opponent.rect.centery,
                              "aix": self.ai.rect.x,
                              "aiy": self.ai.rect.centery,
                              "ballx": self.ball.rect.x,
                              "bally": self.ball.rect.y})
            if i % 10000 == 0:
                print(f"iteration {i}")

        with open(metadata_file_output, 'w' if not append else 'a', newline='') as f:
            headers = list(meta_data[0].keys())
            writer = csv.DictWriter(f, fieldnames=headers)
            if not append:
                writer.writeheader()
            for row in meta_data:
                writer.writerow(row)

    def train_deepproblog(self, epochs, train_query_file, test_query_file, model_name, model_version):
        train_queries = load(train_query_file)
        test_queries = load(test_query_file)
        logger = train_model(self.ai.model, train_queries,
                             nr_epochs=epochs,
                             optimizer=self.ai.optimizer,
                             test_iter=500, test=lambda x: x.accuracy(test_queries, test=True),
                             log_iter=50,
                             model_name=model_name,
                             model_version=model_version)
        logger.write_to_file(f'deepproblog_training_loss_and_accuracy_{model_name}_v{model_version}')

    def train_pytorch(self, epochs):
        logger = train_pytorch(epochs=epochs,
                               test_iter=500,
                               log_iter=50)
        logger.write_to_file('pytorch_training_loss_and_accuracy')


if __name__ == "__main__":
    game = Game()

    # game.generate_data(data_dir_output="../data", metadata_file_output="../data/meta_data.csv", start_rounds=0, append=False, end_rounds=500)
    # game.train_pytorch(epochs=2)
    # game.train_deepproblog(epochs=1,
    #                        train_query_file='../data/deepproblog_train_data_x.txt',
    #                        test_query_file='../data/deepproblog_test_data_x.txt',
    #                        model_name='pong_net_x',
    #                        model_version=0)
    game.run(rounds=100)
