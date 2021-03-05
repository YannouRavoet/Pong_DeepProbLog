import csv
import os
import random
import shutil

import pygame
from dataclasses import dataclass
from GameObjects import Ball, AIPlayer, HumanPlayer, Opponent, RandomPlayer
from Global import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from baseline import train_pytorch
from data_loader import load
from train import train_model

PLAYER_SPEED = 1
BALL_SPEED = 1
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

        # self.opponent = Opponent(x=0, y=SCREEN_HEIGHT / 2, speed=GAME_SPEED, precision=0.9)
        self.opponent = RandomPlayer(x=0, y=SCREEN_HEIGHT/2, speed=PLAYER_SPEED)
        self.ai = AIPlayer(SCREEN_WIDTH - 1, SCREEN_HEIGHT / 2, speed=PLAYER_SPEED)
        self.ai.load_model_snapshot('./models/model_iter_80000.mdl')
        ball_collider_group = pygame.sprite.Group()
        ball_collider_group.add(self.opponent, self.ai)
        self.ball = Ball(speed=BALL_SPEED, collider_group=ball_collider_group)
        self.draw_group = pygame.sprite.Group()
        self.draw_group.add(self.opponent, self.ai, self.ball)

        self.round = 1
        self.scores = {'Opponent': 0, 'AI': 0}

    def update(self):
        self.draw_group.update(self.get_screen_pixels(), self.ball.rect.y)
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
        self.screen.fill(Colors.dark_gray)
        self.draw_group.draw(self.screen)

        pygame.display.flip()
        self.clock.tick(FPS)

    def get_screen_pixels(self):
        pixels = pygame.surfarray.pixels3d(self.screen)
        return pixels

    def run(self, rounds=1):
        while self.round <= rounds:
            self.draw()
            self.update()
        print(f"Results after {rounds} rounds:")
        print(f"Opponent: {self.scores['Opponent']} vs AI: {self.scores['AI']}")

    def generate_raw_data(self, data_dir, metadata_file, append=False, start_rounds=0, end_rounds=1000):
        """Generates frames representing different game-states. The saved meta-data contains the:
                1. id of the image
                2. x and y coordinates of the human
                3. x and y coordinates of the ai
                4. x and y coordinates of the ball
            **WIPES THE ENTIRE DATA DIRECTORY IF NOT APPENDING**.
            Estimate roughly 50 seconds for 1000 images"""

        def build_data_dir_tree():
            os.mkdir(data_dir)
            os.mkdir(data_dir + "/" + img_dir_name)

        img_dir_name = 'generated'
        if not os.path.exists(data_dir):
            build_data_dir_tree()
        if not append:
            shutil.rmtree(data_dir)
            build_data_dir_tree()
        data = list()
        for i in range(start_rounds, end_rounds):
            # set player and ai position
            self.opponent.rect.centery = random.randrange(1, SCREEN_HEIGHT - 1)
            self.ai.rect.centery = random.randrange(1, SCREEN_HEIGHT - 1)
            # set ball position
            self.ball.rect.center = (SCREEN_WIDTH - 2,
                                     random.randrange(max(0, self.ai.rect.centery - 2),  # 0
                                                      min(self.ai.rect.centery + 3, SCREEN_HEIGHT)))  # SCREEN_HEIGHT

            self.draw()
            img_name = str(i) + '.png'
            img_path = os.path.join(data_dir, img_dir_name, img_name)
            pygame.image.save(self.screen, img_path)
            data.append({"img_id": img_name,
                         "humanx": self.opponent.rect.centerx,
                         "humany": self.opponent.rect.centery,
                         "aix": self.ai.rect.centerx,
                         "aiy": self.ai.rect.centery,
                         "ballx": self.ball.rect.centerx,
                         "bally": self.ball.rect.centery})

        with open(metadata_file, 'w' if not append else 'a', newline='') as f:
            headers = list(data[0].keys())
            writer = csv.DictWriter(f, fieldnames=headers)
            if not append:
                writer.writeheader()
            for row in data:
                writer.writerow(row)

    def train_deepproblog(self, epochs=1):
        train_queries = load('../data/deepproblog_train_data.txt')
        test_queries = load('../data/deepproblog_test_data.txt')
        logger = train_model(self.ai.model, train_queries,
                             nr_epochs=epochs,
                             optimizer=self.ai.optimizer,
                             test_iter=500, test=lambda x: x.accuracy(test_queries, test=True),
                             snapshot_iter=5000,
                             log_iter=50)
        logger.write_to_file('deepproblog_training_loss_and_accuracy')

    def train_pytorch(self, epochs=1):
        logger = train_pytorch(epochs=epochs,
                               test_iter=500,
                               log_iter=50)
        logger.write_to_file('pytorch_training_loss_and_accuracy')


if __name__ == "__main__":
    game = Game()
    # game.generate_raw_data("../data", "../data/data.csv", start_rounds=100000, append=True, end_rounds=101000)
    # game.train_deepproblog(epochs=1)
    # game.train_pytorch(epochs=2)
    game.run(rounds=100)
