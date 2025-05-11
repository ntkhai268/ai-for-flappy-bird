import pygame
from pygame.locals import *
import random
from population import *
from individual import *
from math import sqrt

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
scroll_speed = 4 + int(time_alive/1)  # mỗi 800 frame tăng 1 đơn vị tốc độ
flying = False
game_over = False
pipe_gap = random.randint(180, 200)
pipe_height = random.randint(-100, 100)
pipe_frequency = int(random.randint(800, 1200) * (60 / fps))  # điều chỉnh theo tốc độ khung hình gốc

last_pipe = pygame.time.get_ticks() - pipe_frequency
max_score = 0
pass_pipe = False

generation = 0
max_score_list = []


#load images
bg = pygame.image.load('img/bg.png')
ground_img = pygame.image.load('img/ground.png')
button_img = pygame.image.load('img/restart.png')


#function for outputting text onto the screen
def draw_text(text, font, text_col, x, y):
	img = font.render(text, True, text_col)
	screen.blit(img, (x, y))

def reset_game(population): 
	global time_alive, max_score, last_pipe, generation
	generation += 1
	max_score_list.append(max_score)
	print(max_score_list)
	time_alive = 0
	pipe_group.empty()
	population.evolve()
	for i in range(size_population):
		individual = population.individuals[i]
		bird = Bird(100, int(screen_height / random.randint(1, 10)))
		bird.individual = individual
		bird_group.add(bird)
		bird_group.sprites()[i].score = 0

	max_score = 0
	return max_score


class Bird(pygame.sprite.Sprite):
	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		self.images = []
		self.index = 0
		self.counter = 0
		for num in range (1, 4):
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
			#apply gravity
			self.vel += 0.6
			if self.vel > 8:
				self.vel = 8
			if self.rect.bottom < 600:
				self.rect.y += int(self.vel)

		if game_over == False:
			#jump
			if self.jump and not self.clicked:
				self.clicked = True
				self.vel = -10
				self.clicked = False

			if not self.jump:
				self.clicked = False  # cho phép nhảy lần sau nếu điều kiện đúng
				
			#handle the animation
			flap_cooldown = 5
			self.counter += 1
			
			if self.counter > flap_cooldown:
				self.counter = 0
				self.index += 1
				if self.index >= len(self.images):
					self.index = 0
				self.image = self.images[self.index]


			#rotate the bird
			self.image = pygame.transform.rotate(self.images[self.index], self.vel * -2)
		else:
			#point the bird at the ground
			self.image = pygame.transform.rotate(self.images[self.index], -90)

class Pipe(pygame.sprite.Sprite):
	def __init__(self, x, y, position):
		pygame.sprite.Sprite.__init__(self)
		self.image = pygame.image.load("img/pipe.png")
		self.rect = self.image.get_rect()
		self.base_y = y  # lưu vị trí gốc để dao động
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
		# Dao động theo sin
		# offset = 20 * np.sin(pygame.time.get_ticks() / 500 + self.oscillate_phase)
		offset = 0
		if self.position == 1:
			self.rect.bottom = self.base_y - int(pipe_gap / 2) + offset
		else:
			self.rect.top = self.base_y + int(pipe_gap / 2) + offset

		if self.rect.right < 0:
			self.kill()


size_population = 10
population = Population(size_population)
bird = [None] * size_population

pipe_group = pygame.sprite.Group()
bird_group = pygame.sprite.Group()

#create restart button instance
for i in range(size_population):
	individual = Individual()
	population.individuals.append(individual)
	bird[i] = Bird(100, int(screen_height / random.randint(1, 10)))
	bird[i].individual = individual
	bird_group.add(bird[i])

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
    d = sqrt(dx_pipe ** 2 + (bird.rect.centery - (next_top.rect.bottom + next_bottom.rect.top) / 2) ** 2)
    d = d if (bird.rect.centery - (next_top.rect.bottom + next_bottom.rect.top) / 2) > 0 else -d
    velocity = bird.vel

    return [bird_y, dy_bottom, dy_top, dx_pipe, velocity]

run = True
while run:

	clock.tick(fps)
	#draw background
	screen.blit(bg, (0,0))

	pipe_group.draw(screen)
	bird_group.draw(screen)

	#draw and scroll the ground
	screen.blit(ground_img, (ground_scroll, 600))

	#check the score
	for i in range(len(bird_group)):
		if len(pipe_group) > 0:
			if bird_group.sprites()[i].rect.left > pipe_group.sprites()[0].rect.left\
				and bird_group.sprites()[i].rect.right < pipe_group.sprites()[0].rect.right\
				and bird_group.sprites()[i].passed_pipe == False:
				bird_group.sprites()[i].passed_pipe = True
			if bird_group.sprites()[i].passed_pipe == True:
				if bird_group.sprites()[i].rect.left > pipe_group.sprites()[0].rect.right:
					bird_group.sprites()[i].score += 1
					bird_group.sprites()[i].passed_pipe = False

	for i in range(len(bird_group)):
		if bird_group.sprites()[i].score > max_score:
			max_score = bird_group.sprites()[i].score
	
	draw_text(str(max_score), font, white, int(screen_width / 2), 20)
	# draw_text(f"Gen {generation}", pygame.font.SysFont('Arial', 24), white, 430, 20)

	for i in range(len(bird_group)):
		inputs = get_state(pipe_group, bird_group.sprites()[i])
		# print(inputs)
		prediction = bird_group.sprites()[i].individual.neural_network.predict(inputs)
		bird_group.sprites()[i].jump = bool(prediction)

	#look for collision
	bird_pipe_collision = pygame.sprite.groupcollide(bird_group, pipe_group, False, False, collided=pygame.sprite.collide_mask)
	for flappy in bird_pipe_collision:
		flappy.individual.evaluate_fitness(time_alive + flappy.score * 100)
		flappy.kill()

	#once the  bird has hit the ground it's game over and no longer flying
	for flappy in bird_group:
		if flappy.rect.bottom >= 600:
			flappy.individual.evaluate_fitness(time_alive + flappy.score * 100)
			flappy.kill()
		if flappy.rect.top <= 0:
			flappy.individual.evaluate_fitness(time_alive + flappy.score * 100)
			flappy.kill()
			
	if len(bird_group) == 0:
		game_over = True
		flying = False


	if flying == True and game_over == False:
		time_alive += 1
		#generate new pipes
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

	#check for game over and reset
	if game_over == True:
		game_over = False
		score = reset_game(population)
		flying = True   # Tự động tiếp tục mà không cần di chuột


	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
		if flying == False and game_over == False:
			flying = True

	pygame.display.flip()

pygame.quit()
