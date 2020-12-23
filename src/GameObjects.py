import pygame
import random
from abc import abstractmethod

import torch

from Global import screen_height, screen_width

from deepproblog.network import Network
from model import Model
from optimizer import Optimizer
from pong_model import PONG_Net, neural_predicate
from train import train_model


class GameObject(pygame.sprite.Sprite):
    def __init__(self, img_path, x, y):
        super().__init__()
        self.image = pygame.image.load(img_path)
        self.rect = self.image.get_rect(center=(x, y))

    @abstractmethod
    def update(self, screen):
        raise NotImplementedError


class Player(GameObject):
    def __init__(self, x, y, speed):
        super().__init__('../resources/paddle.png', x, y)
        self.speed = speed
        self.movement = 0  # -1, 0, +1
        self.score = 0

    @abstractmethod
    def update_movement(self, screen):
        raise NotImplementedError

    def update(self, screen):
        self.update_movement(screen)
        self.rect.y += self.movement
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= screen_height:
            self.rect.bottom = screen_height


class HumanPlayer(Player):
    def update_movement(self, screen):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.movement -= self.speed
                if event.key == pygame.K_DOWN:
                    self.movement += self.speed
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    self.movement += self.speed
                if event.key == pygame.K_DOWN:
                    self.movement -= self.speed


class AIPlayer(Player):
    def __init__(self, x, y, speed):
        super().__init__(x, y, speed)
        with open('ai.pl') as f:
            self.problog_string = f.read()

        self.network = PONG_Net()
        self.net = Network(self.network, 'pong_net', neural_predicate)
        self.net.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.model = Model(self.problog_string, [self.net], caching=False)
        self.optimizer = Optimizer(self.model, 2)

    def train(self, train_queries, test_queries):
        train_model(self.model, train_queries,
                    nr_epochs=1,
                    optimizer=self.optimizer,
                    test_iter=1000, test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)

    """Calls the logic program passing the screen pixels as input"""
    def update_movement(self, screen):
        self.movement = random.choice([-1, 0, +1])


class Ball(GameObject):
    def __init__(self, x_speed, y_speed, collider_group):
        super().__init__('../resources/ball.png', screen_width / 2, screen_height / 2)
        self.x_speed = x_speed
        self.y_speed = y_speed
        self.collider_group = collider_group

    def update(self, screen):
        self.rect.x += self.x_speed
        self.rect.y += self.y_speed

        if self.rect.top <= 0 or self.rect.bottom >= screen_height:  # if you hit screen top or bottom edge
            self.y_speed *= -1

        if pygame.sprite.spritecollide(self, self.collider_group, False):
            collision_object = pygame.sprite.spritecollide(self, self.collider_group, False)[0].rect
            self.x_speed *= -1
            self.rect.x = collision_object.x + self.x_speed / abs(self.x_speed)

    def reset(self):
        self.rect.center = (screen_width / 2, screen_height / 2)
