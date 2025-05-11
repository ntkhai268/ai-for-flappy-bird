
import pygame
from pygame.locals import *
import random
from dql_agent_noper import DQNAgent
import numpy as np
import math
import os
import csv

pygame.init()

clock = pygame.time.Clock()
fps = 120

screen_width = 400
screen_height = 600

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Flappy Bird')

#define font
font = pygame.font.SysFont('Bauhaus 93', 60)

#define colours
white = (255, 255, 255)

#define game variables
ground_scroll = 0
time_alive = 0
scroll_speed = 4
flying = False
game_over = False
pipe_gap = random.randint(180, 200)
pipe_height = random.randint(-100, 100)
pipe_frequency = int(random.randint(1500, 2000) * (60 / fps))
SAVE_MODEL_EVERY_N_EPISODES = 1000
TARGET_EPISODE_TO_SAVE_FINAL = 10000
MODELS_DIRECTORY = "saved_models_flappy"

last_pipe = -pipe_frequency
pass_pipe = False
episode = 0 
total_reward = 0
tracking_data = []

#load images
bg = pygame.image.load('img/bg.png')
ground_img = pygame.image.load('img/ground.png')
button_img = pygame.image.load('img/restart.png')

#function for outputting text onto the screen
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))

def save_tracking_data(tracking_data, filename="training_report.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Episode", "Score", "Total Reward", "Time Alive", "Epsilon", "Replay Buffer Size", "Train Steps"])
        for data in tracking_data:
            writer.writerow(data)

# def get_state(pipe_group, bird):
#     pipes_top = [p for p in pipe_group if p.position == 1]
#     pipes_bottom = [p for p in pipe_group if p.position == -1]

#     if not pipes_top or not pipes_bottom:
#         return [bird.rect.centery / screen_height, 1.0, 1.0, 1.0, bird.vel / 8.0]

#     next_top = next((p for p in pipes_top if p.rect.right > bird.rect.left), pipes_top[0])
#     next_bottom = next((p for p in pipes_bottom if p.rect.right > bird.rect.left), pipes_bottom[0])

#     bird_y = 2* (bird.rect.centery / screen_height)  - 1 # [-1, 1]
#     dy_bottom = (next_bottom.rect.top - bird.rect.centery) / screen_height  # [-1, 1]
#     dy_top = (bird.rect.centery - next_top.rect.bottom) / screen_height  # [-1, 1]
#     dx_pipe = 2 * ((next_top.rect.centerx - bird.rect.centerx) / screen_width)  - 1 # [1, 1]
#     velocity = bird.vel / 8.0  # [-1.25, 1]

#     return [bird_y, dy_bottom, dy_top, dx_pipe, velocity]

def get_state(pipe_group, bird):
    pipes_top = [p for p in pipe_group if p.position == 1]
    pipes_bottom = [p for p in pipe_group if p.position == -1]

	# Nếu chưa có đủ pipe → trả về mặc định
    if not pipes_top or not pipes_bottom:
        return [bird.rect.centery, 600, 600, 400, bird.vel] # trường hợp 6 đầu vào

    # Ống gần nhất
    next_top = next((p for p in pipes_top if p.rect.right > bird.rect.left), pipes_top[0])
    next_bottom = next((p for p in pipes_bottom if p.rect.right > bird.rect.left), pipes_bottom[0])

    bird_y = bird.rect.centery
    dy_bottom = next_bottom.rect.top - bird.rect.centery
    dy_top = bird.rect.centery - next_top.rect.bottom
    dx_pipe = next_top.rect.centerx - bird.rect.centerx

    velocity = bird.vel

    return [bird_y, dy_bottom, dy_top, dx_pipe, velocity]

class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        self.index = 0
        self.counter = 0
        for num in range(1, 4):
            img = pygame.image.load(f"img/bird{num}.png")
            self.images.append(img)
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.vel = 0
        self.clicked = False
        self.score = 0
        self.jump = False
        self.individual = None
        self.mask = pygame.mask.from_surface(self.image)
        self.passed_pipe = False
        self.time_alive = 0

    def update(self):   
        if flying == True:
            self.vel += 0.6
            if self.vel > 8:
                self.vel = 8
            if self.rect.bottom < 600:
                self.rect.y += int(self.vel)

        if game_over == False:
            if self.jump and not self.clicked:
                self.clicked = True
                self.vel = -10
                self.clicked = False
            if not self.jump:
                self.clicked = False
                
            flap_cooldown = 5
            self.counter += 1
            if self.counter > flap_cooldown:
                self.counter = 0
                self.index += 1
                if self.index >= len(self.images):
                    self.index = 0
                self.image = self.images[self.index]

            self.image = pygame.transform.rotate(self.images[self.index], self.vel * -2)
        else:
            self.image = pygame.transform.rotate(self.images[self.index], -90)

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("img/pipe.png")
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
        offset = 0
        if self.position == 1:
            self.rect.bottom = self.base_y - int(pipe_gap / 2) + offset
        else:
            self.rect.top = self.base_y + int(pipe_gap / 2) + offset
        if self.rect.right < 0:
            self.kill()

pipe_group = pygame.sprite.Group()
bird_group = pygame.sprite.Group()

bird = Bird(100, screen_height // 2)
bird_group.add(bird)

state_size = 5
action_size = 2     
agent = DQNAgent(state_size, action_size)
learn_counter = 0
reward = 0
loss = 0

run = True
while run:
    clock.tick(fps)
    screen.blit(bg, (0,0))
    pipe_group.draw(screen)
    bird_group.draw(screen)
    screen.blit(ground_img, (ground_scroll, 600))

    state = np.array(get_state(pipe_group, bird), dtype=np.float32)
    action = agent.make_decision(state)
    bird.jump = (action == 1)
    reward = 0.1  # Phần thưởng liên tục
    total_reward += reward

    if len(pipe_group) > 0:
        # for p in pipe_group.sprites():
        #     if p.rect.left > bird.rect.left:
        #         pipe_next = p
        #         break
        # #sua lai logic phan nay
        # mid_gap = (pipe_next.rect.midtop[0], pipe_next.rect.midtop[1] - pipe_gap / 2)
        # distance = math.dist(bird.rect.center, mid_gap)
        # # distance = distance if bird.rect.center[0] >= mid_gap[0] else - distance
        # pygame.draw.line(screen, (255, 0, 0), bird.rect.center, mid_gap, 2)
        # reward += 10 / (distance + 1)
        # total_reward += reward
        pipe = pipe_group.sprites()[0]
        if bird.rect.left > pipe.rect.left and bird.rect.right < pipe.rect.right and not bird.passed_pipe:
            bird.passed_pipe = True
        if bird.passed_pipe and bird.rect.left > pipe.rect.right:
            bird.score += 1
            reward = 100
            total_reward += reward
            bird.passed_pipe = False

    draw_text(str(bird.score), font, white, int(screen_width / 2), 20)

    if flying == True and game_over == False:
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

        ground_scroll -= scroll_speed
        if abs(ground_scroll) > 35:
            ground_scroll = 0
    
    next_state = np.array(get_state(pipe_group, bird), dtype=np.float32)
    if pygame.sprite.spritecollide(bird, pipe_group, False, pygame.sprite.collide_mask):
        reward = -100
        total_reward += reward
        game_over = True
        flying = False

    if bird.rect.bottom >= 600 or bird.rect.top <= 0:
        reward = -100
        total_reward += reward
        game_over = True
        flying = False
        
    agent.save_experience(state, action, reward, next_state, game_over)
    
    if learn_counter % 10 == 0:  # Tăng tần suất huấn luyện
        loss = agent.train_main_network(256)  # Tăng batch size
        # loss_value = 0
        # if loss:
        #     loss_value = loss.item()
    learn_counter += 1
    
    if game_over:
        episode += 1
        tracking_data.append([
            episode,
            bird.score,
            total_reward,
            time_alive,
            loss,
            agent.epsilon,
            len(agent.replay_buffer),
            agent.train_steps
        ])
        if episode % 10 == 0:
            save_tracking_data(tracking_data)
            tracking_data = []
            print(f"Episode {episode}: Score={bird.score}, Total Reward={total_reward}, Time Alive={time_alive}, Loss={loss:.4f}, Epsilon={agent.epsilon:.3f}")
        
        if not os.path.exists(MODELS_DIRECTORY):
            os.makedirs(MODELS_DIRECTORY)

        if episode % SAVE_MODEL_EVERY_N_EPISODES == 0 and episode > 0:
            checkpoint_filename = os.path.join(MODELS_DIRECTORY, f"flappy_dqn_episode_{episode}.pth")
            agent.save_model(checkpoint_filename)

        if episode == TARGET_EPISODE_TO_SAVE_FINAL:
            final_filename = os.path.join(MODELS_DIRECTORY, f"flappy_dqn_episode_{TARGET_EPISODE_TO_SAVE_FINAL}_final.pth")
            agent.save_model(final_filename)
            print(f"ĐÃ ĐẠT {TARGET_EPISODE_TO_SAVE_FINAL} EPISODES. MODEL CUỐI CÙNG ĐÃ ĐƯỢC LƯU.")
        bird.rect.center = [100, screen_height // 2]
        bird.vel = 0
        total_reward = 0
        bird.score = 0
        time_alive = 0
        pipe_group.empty()
        game_over = False
        flying = True
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if flying == False and game_over == False:
            flying = True

    pygame.display.flip()

pygame.quit()
