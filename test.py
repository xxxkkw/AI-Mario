import pygame

keybinding = {
    'action': pygame.K_s,
    'jump': pygame.K_a,
    'left': pygame.K_LEFT,
    'right': pygame.K_RIGHT,
    'down': pygame.K_DOWN
}

# 初始化 Pygame
pygame.init()
print(pygame.K_s, pygame.K_a, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DOWN)
# 获取当前键盘状态
keys = pygame.key.get_pressed()
if not keys[pygame.K_s]:
    print("s")
if not keys[pygame.K_a]:
    print("a")
if not keys[pygame.K_LEFT]:
    print("l")
if not keys[pygame.K_RIGHT]:
    print("r")
if not keys[pygame.K_DOWN]:
    print("d")
# 将元组转换为列表，这样就可以修改它了
keys_list = list(keys)

# 假设你想将按键 'A'（键码 4）设置为按下状态
keys_list[4] = True

# 你可以继续修改其他按键的状态

# 现在可以使用修改后的 keys_list 了
print(keys_list)