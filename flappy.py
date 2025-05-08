import pygame
from pygame.locals import *
import random
from population import Population
from individual import Individual
import numpy as np
from math import sqrt

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
pipe_gap = 190
pipe_frequency = int(1500 * (60 / fps))
last_pipe = pygame.time.get_ticks() - pipe_frequency
max_score = 0
column_count = 0
generation = 0
BIRD_X = 100

bg = pygame.image.load('img/bg.png')
ground_img = pygame.image.load('img/ground.png')

def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))

def reset_game(population):
    global time_alive, max_score, last_pipe, generation, column_count
    generation += 1
    time_alive = 0
    column_count = 0
    pipe_group.empty()
    population.evolve()

    bird_group.empty()
    for i in range(size_population):
        individual = population.individuals[i]
        bird = Bird(100, random.randint(150, 450))
        bird.individual = individual
        bird_group.add(bird)

    max_score = 0

    # Tạo pipe ngay từ đầu
    pipe_height = random.randint(-100, 100)
    btm_pipe = Pipe(screen_width + 100, int(screen_height / 2) + pipe_height, -1)
    top_pipe = Pipe(screen_width + 100, int(screen_height / 2) + pipe_height, 1)
    pipe_group.add(btm_pipe)
    pipe_group.add(top_pipe)
    last_pipe = pygame.time.get_ticks()

class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.images = [pygame.image.load(f"img/bird{n}.png") for n in range(1, 4)]
        self.index = 0
        self.counter = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect(center=(x, y))
        self.vel = -2
        self.score = 0
        self.jump = False
        self.individual = None
        self.mask = pygame.mask.from_surface(self.image)
        self.passed_pipe = False

    def update(self):
        if flying:
            self.vel += 0.4
            self.vel = min(self.vel, 8)
            if self.rect.bottom < screen_height:
                self.rect.y += int(self.vel)

            # Kiểm tra va chạm với mép trên và mép dưới
            if self.rect.top <= 0 or self.rect.bottom >= screen_height:
                self.individual.score = self.score
                self.individual.survival_time = time_alive
                self.individual.calculate_fitness()
                self.kill()
                return

        if not game_over:
            if self.jump:
                self.vel = -8
                self.jump = False

            self.counter += 1
            if self.counter > 5:
                self.counter = 0
                self.index = (self.index + 1) % len(self.images)
                self.image = self.images[self.index]

            angle = max(-30, min(30, self.vel * -3))
            self.image = pygame.transform.rotate(self.images[self.index], angle)
        else:
            self.image = pygame.transform.rotate(self.images[self.index], -90)

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, position):
        super().__init__()
        self.image = pygame.image.load("img/pipe.png")
        self.rect = self.image.get_rect()
        self.position = position
        self.passed = False

        if position == 1:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.bottomleft = [x, y - int(pipe_gap / 2)]
        else:
            self.rect.topleft = [x, y + int(pipe_gap / 2)]

    def update(self):
        self.rect.x -= scroll_speed
        if self.rect.right < 0:
            self.kill()

def get_state(pipe_group, bird):
    pipes_top = [p for p in pipe_group if p.position == 1]
    pipes_bottom = [p for p in pipe_group if p.position == -1]

    if not pipes_top or not pipes_bottom:
        return [bird.rect.centery, 100, 100, 150, bird.vel]

    next_top = next((p for p in pipes_top if p.rect.right > bird.rect.left), pipes_top[0])
    next_bottom = next((p for p in pipes_bottom if p.rect.right > bird.rect.left), pipes_bottom[0])

    bird_y = bird.rect.centery
    dy_bottom = next_bottom.rect.top - bird_y
    dy_top = bird_y - next_top.rect.bottom
    dx_pipe = next_top.rect.centerx - bird.rect.centerx

    return [bird_y, dy_bottom, dy_top, dx_pipe, bird.vel]

# === Khởi tạo ===
size_population = 10
input_size = 5
population = Population(size_population, input_size)
pipe_group = pygame.sprite.Group()
bird_group = pygame.sprite.Group()

for i in range(size_population):
    individual = Individual(input_size)
    population.individuals[i] = individual
    bird = Bird(100, random.randint(150, 450))
    bird.individual = individual
    bird_group.add(bird)

run = True
while run:
    clock.tick(fps)
    screen.blit(bg, (0, 0))
    pipe_group.draw(screen)
    bird_group.draw(screen)
    screen.blit(ground_img, (ground_scroll, 600))

    if flying and not game_over:
        time_alive += 1
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

        # Đếm số cột đã vượt qua
        for pipe in pipe_group:
            if pipe.position == 1 and not pipe.passed and pipe.rect.centerx < BIRD_X:
                column_count += 1
                pipe.passed = True

        ground_scroll -= scroll_speed
        if abs(ground_scroll) > 35:
            ground_scroll = 0

    # Hiển thị số cột đã vượt qua
    draw_text(str(column_count), font, white, int(screen_width / 2), 20)

    for bird in bird_group:
        inputs = get_state(pipe_group, bird)
        output = bird.individual.brain.predict(np.array(inputs).reshape(1, -1))
        bird.jump = output > 0.3

    collided = pygame.sprite.groupcollide(
        bird_group, pipe_group, False, False, collided=pygame.sprite.collide_mask
    )
    for bird in collided:
        bird.individual.score = bird.score
        bird.individual.survival_time = time_alive
        bird.individual.calculate_fitness()
        bird.kill()

    for bird in bird_group:
        if bird.rect.bottom >= screen_height or bird.rect.top <= 0:
            bird.individual.score = bird.score
            bird.individual.survival_time = time_alive
            bird.individual.calculate_fitness()
            bird.kill()

    if len(bird_group) == 0:
        game_over = True
        flying = False

    if game_over:
        game_over = False
        reset_game(population)
        flying = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if not flying:
            flying = True

    pygame.display.flip()

pygame.quit()
