import numpy as np
import pygame
import random
from abc import abstractmethod

import torch
from problog.logic import list2term, Term, Constant

from Global import SCREEN_HEIGHT, SCREEN_WIDTH
from network import Network
from model import Model
from optimizer import Optimizer
from models_deepproblog import PONG_Net_Y, neural_predicate_y, PONG_Net_X, neural_predicate_x


class GameObject(pygame.sprite.Sprite):
    def __init__(self, img_path, x, y):
        super().__init__()
        self.image = pygame.image.load(img_path)
        self.rect = self.image.get_rect(center=(x, y))
        self.x_init = x
        self.y_init = y

    @abstractmethod
    def update(self, screen, ball_y):
        raise NotImplementedError


class Player(GameObject):
    def __init__(self, x, y, speed):
        super().__init__('../resources/paddle.png', x, y)
        self.speed = speed
        self.movement = 0  # up, noop, +down

    @abstractmethod
    def update_movement(self, screen, ball_y):
        raise NotImplementedError

    def update(self, screen, meta):
        self.update_movement(screen, meta)
        self.rect.y += self.movement * self.speed
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT

    def reset(self):
        self.rect.centerx = self.x_init
        self.rect.centery = self.y_init + random.choice([-1, 0, 1])


class HumanPlayer(Player):
    def update_movement(self, screen, _):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.movement = -self.speed
                if event.key == pygame.K_DOWN:
                    self.movement = self.speed
            elif event.type == pygame.KEYUP:
                self.movement = 0


class RandomPlayer(Player):
    def __init__(self, x, y, speed):
        super().__init__(x, y, speed)

    def update_movement(self, screen, _):
        self.movement = random.choice([-1, 0, +1])


class Opponent(Player):
    def __init__(self, x, y, speed, precision=1):
        super().__init__(x, y, speed)
        self.precision = precision

    def update_movement(self, screen, meta):
        p = random.uniform(0, 1)
        if meta['bally'] > self.rect.centery:
            self.movement = 1 if p <= self.precision else random.choice([-1, 0])
        elif meta['bally'] < self.rect.centery:
            self.movement = -1 if p <= self.precision else random.choice([0, 1])
        else:
            self.movement = 0 if p <= self.precision else random.choice([-1, 1])


class AIPlayer_v1(Player):
    def __init__(self, x, y, speed):
        super().__init__(x, y, speed)
        with open('ai_v1.pl') as f:
            self.problog_string = f.read()

        self.network_y = PONG_Net_Y()
        self.net_y = Network(self.network_y, 'pong_net_y', neural_predicate_y)
        self.net_y.optimizer = torch.optim.Adam(self.network_y.parameters(), lr=0.001)
        # caching the queries doesn't really help since the img is passed in the query
        # => making the state-space of the cache enormous
        self.model = Model(self.problog_string, [self.net_y], caching=False, saving=False)
        self.optimizer = Optimizer(self.model, accumulation=1)

        self.action_mapping = {'down': +1, 'noop': 0, 'up': -1}  # maps the actions to the correct movement

    def load_model_snapshot(self, snapshot_file, net_name):
        self.model.load_state(snapshot_file, net_name)

    def update_movement(self, screen, _):
        """Calls the logic program passing the screen pixels as input"""
        screen = np.transpose(screen, (2, 1, 0))  # 30x16x3 -> 3x16x30
        screen = screen.flatten()
        screen = list(map(lambda x: int(x), screen))
        img_term = list2term(screen)
        results = self.model.solve(Term('choose_action', *[img_term, Constant(self.rect.centery), None]),
                                   test=True)
        best_action = max(results.keys(), key=lambda k: results[k][0]).args[-1]
        self.movement = self.action_mapping[str(best_action)]


class AIPlayer_v2(Player):
    def __init__(self, x, y, speed):
        super().__init__(x, y, speed)
        with open('ai_v2.pl') as f:
            self.problog_string = f.read()

        self.network_y = PONG_Net_Y()
        self.net_y = Network(self.network_y, 'pong_net_y', neural_predicate_y)
        self.net_y.optimizer = torch.optim.Adam(self.network_y.parameters(), lr=0.001)
        self.network_x = PONG_Net_X()
        self.net_x = Network(self.network_x, 'pong_net_x', neural_predicate_x)
        self.net_x.optimizer = torch.optim.Adam(self.network_x.parameters(), lr=0.001)

        self.model = Model(self.problog_string, [self.net_y, self.net_x], caching=False, saving=False)
        self.optimizer = Optimizer(self.model, accumulation=16)

        self.action_mapping = {'down': +1, 'noop': 0, 'up': -1}  # maps the actions to the correct movement

        self.previous_screen = None

    def load_model_snapshot(self, snapshot_file, net_name):
        self.model.load_state(snapshot_file, net_name)

    def update_movement(self, screen, _):
        """Calls the logic program passing the screen pixels as input"""
        screen = np.transpose(screen, (2, 1, 0))  # 30x16x3 -> 3x16x30
        screen = screen.flatten()
        screen = list(map(lambda x: int(x), screen))
        img_term = list2term(screen)
        if self.previous_screen is None:
            self.previous_screen = img_term
        results = self.model.solve(Term('choose_action', *[self.previous_screen,
                                                           img_term,
                                                           Constant(self.rect.centery),
                                                           None]), test=True)
        self.previous_screen = img_term
        best_action = max(results.keys(), key=lambda k: results[k][0]).args[-1]
        self.movement = self.action_mapping[str(best_action)]

    def reset(self):
        super().reset()
        self.previous_screen = None


class Ball(GameObject):
    def __init__(self, x, y, speed, collider_group):
        super().__init__('../resources/ball.png', x, y)
        self.speed = speed
        self.x_dir = random.choice([-1, 1])
        self.y_dir = random.choice([-1, 1])
        self.collider_group = collider_group

    def update(self, screen, ball_y):
        for _ in range(0, self.speed):
            self.rect.x += self.x_dir
            self.rect.y += self.y_dir
            if pygame.sprite.spritecollide(self, self.collider_group, False):
                self.x_dir *= -1
                self.rect.x += 2 * self.x_dir  # reverse the horizontal movement
            if self.rect.y == 0 or self.rect.y == SCREEN_HEIGHT - 1:  # if you hit screen top or bottom edge
                self.y_dir *= -1

    def reset(self):
        self.rect.center = (self.x_init, self.y_init)
        self.x_dir = random.choice([-1, 1])
        self.y_dir = random.choice([-1, 1])
