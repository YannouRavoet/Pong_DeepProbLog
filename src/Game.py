import csv
import os
import random
import shutil

import pygame
from dataclasses import dataclass
from GameObjects import Ball, AIPlayer, HumanPlayer
from Global import screen_width, screen_height


@dataclass
class Colors:
    light_gray = (200, 200, 200)
    dark_gray = pygame.Color('grey12')


class Game:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.SCALED)
        pygame.display.set_caption('Pong')

        self.player = HumanPlayer(x=0, y=screen_height / 2, speed=1)
        self.ai = AIPlayer(screen_width - 1, screen_height / 2, speed=1)
        ball_collider_group = pygame.sprite.Group()
        ball_collider_group.add(self.player, self.ai)
        self.ball = Ball(x_speed=-1, y_speed=1, collider_group=ball_collider_group)
        self.draw_group = pygame.sprite.Group()
        self.draw_group.add(self.player, self.ai, self.ball)

    def update(self):
        self.draw_group.update(self.get_screen_pixels())
        if self.ball.rect.right > screen_width:
            self.player.score += 1
            self.ball.reset()
            print(f"Player: {self.player.score} - AI: {self.ai.score}")
        if self.ball.rect.left < 0:
            self.ai.score += 1
            self.ball.reset()
            print(f"Player: {self.player.score} - AI: {self.ai.score}")

    def draw(self):
        self.screen.fill(Colors.dark_gray)
        self.draw_group.draw(self.screen)

        pygame.display.flip()
        self.clock.tick(10)

    def get_screen_pixels(self):
        pixels = pygame.surfarray.pixels3d(self.screen)
        return pixels

    def run(self):
        while True:
            self.update()
            self.draw()

    """Generates frames representing different game-states. The saved meta-data contains the:
            1. id of the image
            2. x and y coordinates of the player
            3. x and y coordinates of the opponent
            4. x and y coordinates of the ball
    Deletes all previously generated data if not appending."""

    def generate_raw_data(self, img_folder, csv_path, append=False, start_rounds=0, end_rounds=1000):
        if not append:
            shutil.rmtree(img_folder)
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
        data = list()
        for i in range(start_rounds, end_rounds):
            # set player and ai position
            self.player.rect.centery = random.randrange(1, screen_height - 1)
            self.ai.rect.centery = random.randrange(1, screen_height - 1)
            # set ball position
            self.ball.rect.center = (random.randrange(0, screen_width), random.randrange(0, screen_height))
            if self.ball.rect.centerx == 0:  # if ball could be colliding with player paddle
                while self.ball.rect.centery in range(self.player.rect.centery - 1, self.player.rect.centery + 2):
                    self.ball.rect.centery = random.randrange(0, screen_height)
            elif self.ball.rect.centerx == screen_width - 1:  # if ball could be colliding with opponent paddle
                while self.ball.rect.centery in range(self.ai.rect.centery - 1, self.ai.rect.centery + 2):
                    self.ball.rect.centery = random.randrange(0, screen_height)
            # set desired action
            if self.ai.rect.centery > self.ball.rect.centery:
                action = -1
            elif self.ai.rect.centery < self.ball.rect.centery:
                action = 1
            else:
                action = 0

            self.draw()
            img_name = str(i) + '.png'
            img_path = os.path.join(img_folder, img_name)
            pygame.image.save(self.screen, img_path)
            data.append({"img_id": img_name,
                         "playerx": self.player.rect.centerx,
                         "playery": self.player.rect.centery,
                         "opponentx": self.ai.rect.centerx,
                         "opponenty": self.ai.rect.centery,
                         "ballx": self.ball.rect.centerx,
                         "bally": self.ball.rect.centery,
                         "action": action})
        with open(csv_path, 'w' if not append else 'a', newline='') as f:
            headers = list(data[0].keys())
            writer = csv.DictWriter(f, fieldnames=headers)
            if not append:
                writer.writeheader()
            for row in data:
                writer.writerow(row)

    def train_ai(self, train_queries, test_queries):
        self.ai.train(train_queries, test_queries)


if __name__ == "__main__":
    game = Game()
    game.generate_raw_data("../data/imgs", "../data/data.csv", start_rounds=300000, append=True, end_rounds=400000)
    # game.train_ai()
    # game.run()
