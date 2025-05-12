import pygame
from pygame.locals import *
import random
from agent.dql_agent_noper import DQNAgent
from agent.individual import Individual  # <-- dùng GA + ANN
import numpy as np
import math
import os
import csv
import torch

pygame.init()
clock = pygame.time.Clock()
fps = 120
screen_width = 400
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Flappy Bird')

font = pygame.font.SysFont('Bauhaus 93', 60)
white = (255, 255, 255)

ground_scroll = 0
time_alive = 0
scroll_speed = 4
flying = False
game_over = False
pipe_gap = random.randint(200, 230)
pipe_height = random.randint(-100, 100)
pipe_frequency = int(random.randint(1800, 2000) * (60 / fps))
last_pipe = -pipe_frequency

episode = 0
total_reward = 0
game_id = 1
ga_win = 0
dql_win = 0


bg = pygame.image.load('assets/img/bg.png')
ground_img = pygame.image.load('assets/img/ground.png')

# Load model DQL
MODEL_PATH = "saved_models_flappy/flappy_dqn_episode_1000.pth"
agent = DQNAgent(5, 2)
agent.main_network.load_state_dict(torch.load(MODEL_PATH, map_location=agent.device))
agent.main_network.eval()
agent.epsilon = 0.0

# Load genome GA + ANN
with open("models/saved_models_gaann/gen_98_fitness_14641.txt", "r") as f:
    genome = [float(w.strip()) for w in f.readlines()]
ga_individual = Individual(genome=genome)

def log_score(game_id, ga_score, dql_score):
    filepath = "test_scores_log.csv"
    write_header = not os.path.exists(filepath) or os.stat(filepath).st_size == 0
    with open(filepath, "a") as f:
        if write_header:
            f.write("Game,GA Score,DQL Score\n")
        f.write(f"{game_id},{ga_score},{dql_score}\n")
        
def draw_text(text, font, text_col, x, y):
	img = font.render(text, True, text_col)
	screen.blit(img, (x, y))


def get_state(pipe_group, bird):
    pipes_top = [p for p in pipe_group if p.position == 1]
    pipes_bottom = [p for p in pipe_group if p.position == -1]
    if not pipes_top or not pipes_bottom:
        return [bird.rect.centery, 600, 600, 400, bird.vel]
    next_top = next((p for p in pipes_top if p.rect.right > bird.rect.left), pipes_top[0])
    next_bottom = next((p for p in pipes_bottom if p.rect.right > bird.rect.left), pipes_bottom[0])
    bird_y = bird.rect.centery
    dy_bottom = next_bottom.rect.top - bird.rect.centery
    dy_top = bird.rect.centery - next_top.rect.bottom
    dx_pipe = next_top.rect.centerx - bird.rect.centerx
    velocity = bird.vel
    return [bird_y, dy_bottom, dy_top, dx_pipe, velocity]

class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y, color_index, ga=False):
        super().__init__()
        if ga:
            self.images = [pygame.image.load(f"assets/img/bird{num}ga.png") for num in range(1, 4)]
        else:
            self.images = [pygame.image.load(f"assets/img/bird{num}.png") for num in range(1, 4)]
        self.index = 0
        self.counter = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect(center=[x, y])
        self.vel = 0
        self.clicked = False
        self.score = 0
        self.jump = False
        self.mask = pygame.mask.from_surface(self.image)
        self.passed_pipe = False
        self.color_index = color_index

    def update(self):
        if flying:
            self.vel = min(self.vel + 0.6, 8)
            if self.rect.bottom < screen_height:
                self.rect.y += int(self.vel)
        if not game_over:
            if self.jump and not self.clicked:
                self.clicked = True
                self.vel = -10
                self.clicked = False
            if not self.jump:
                self.clicked = False
            self.counter += 1
            if self.counter > 5:
                self.counter = 0
                self.index = (self.index + 1) % len(self.images)
            self.image = pygame.transform.rotate(self.images[self.index], self.vel * -2)
        else:
            self.image = pygame.transform.rotate(self.images[self.index], -90)

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, position):
        super().__init__()
        self.image = pygame.image.load("assets/img/pipe.png")
        self.rect = self.image.get_rect()
        self.base_y = y
        self.oscillate_phase = random.uniform(0, 2 * np.pi)
        self.position = position
        self.speed = scroll_speed
        if position == 1:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.bottomleft = [x, y - int(pipe_gap / 2)]
        elif position == -1:
            self.rect.topleft = [x, y + int(pipe_gap / 2)]

    def update(self):
        self.rect.x -= self.speed
        if self.rect.right < 0:
            self.kill()

pipe_group = pygame.sprite.Group()
bird_group = pygame.sprite.Group()

# Bird 1: GA+ANN | Bird 2: DQL
bird_ga = Bird(100, screen_height // 2 - 20, color_index=1, ga=True)
bird_dql = Bird(100, screen_height // 2 + 20, color_index=2)
bird_group.add(bird_ga)
bird_group.add(bird_dql)

run = True
while run:
    clock.tick(fps)
    screen.blit(bg, (0,0))
    pipe_group.draw(screen)
    bird_group.draw(screen)
    screen.blit(ground_img, (ground_scroll, 600))

    for bird, controller in zip([bird_ga, bird_dql], [ga_individual, agent]):
        state = np.array(get_state(pipe_group, bird), dtype=np.float32)

        if isinstance(controller, Individual):
            # Thêm bias = 1.0 vào cuối state nếu là GA
            state_with_bias = np.append(state, 1.0)  # ← thêm bias
            prediction = controller.neural_network.predict(state_with_bias)
            bird.jump = bool(prediction)
        else:
            action = controller.make_decision(state)
            bird.jump = (action == 1)

    if flying and not game_over:
        time_now = pygame.time.get_ticks()
        if time_now - last_pipe > pipe_frequency:
            pipe_height = random.randint(-100, 100)
            btm_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, -1)
            top_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, 1)
            pipe_group.add(btm_pipe)
            pipe_group.add(top_pipe)
            last_pipe = time_now
        pipe_group.update()
        bird_group.update()
        ground_scroll -= scroll_speed
        if abs(ground_scroll) > 35:
            ground_scroll = 0

    for bird in bird_group.sprites():
        if len(pipe_group) > 0:
            pipe = pipe_group.sprites()[0]
            if bird.rect.left > pipe.rect.left and bird.rect.right < pipe.rect.right and not bird.passed_pipe:
                bird.passed_pipe = True
            if bird.passed_pipe and bird.rect.left > pipe.rect.right:
                bird.score += 1
                bird.passed_pipe = False
        if pygame.sprite.spritecollide(bird, pipe_group, False, pygame.sprite.collide_mask):
            bird_group.remove(bird)
            # flying = False
        if bird.rect.bottom >= screen_height or bird.rect.top <= 0:
            bird_group.remove(bird)
            # flying = False
            
    if len(bird_group) == 0:
        game_over = True
    # draw_text(f"GA: {bird_ga.score}  |  DQL: {bird_dql.score}", font, white, 40, 20)

    draw_text(f"GA Score: {bird_ga.score} | DQL Score: {bird_dql.score}", pygame.font.SysFont('Arial', 24), white, 10, 50)
    
    
    if game_over:
        print(f"Game Over → GA Score: {bird_ga.score} | DQL Score: {bird_dql.score}")
        log_score(ga_score=bird_ga.score, dql_score=bird_dql.score, game_id=game_id)
        game_id += 1
        if bird_ga.score > bird_dql.score:
            ga_win += 1
        else:
            dql_win += 1
        # Reset chim:
        bird_ga.rect.center = [100, screen_height // 2 - 20]
        bird_dql.rect.center = [100, screen_height // 2 + 20]
        bird_ga.vel = 0
        bird_dql.vel = 0
        bird_ga.score = 0
        bird_dql.score = 0

        # Thêm lại chim đã bị remove
        bird_group.empty()
        bird_group.add(bird_ga)
        bird_group.add(bird_dql)

        # Reset ống
        pipe_group.empty()
        game_over = False
        flying = True

    draw_text(f"Tỉ số: GA {ga_win} - {dql_win} DQL", pygame.font.SysFont('Arial', 24), white, 10, 110)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if not flying and not game_over:
            flying = True

    pygame.display.flip()

pygame.quit()