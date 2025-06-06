import numpy as np
import pygame

from unicycle_env.envs.unicycle_basic import UniCycleBasicEnv

env = UniCycleBasicEnv(render_mode="human")
obs, info = env.reset()
env_count = env.get_environment_count()

v = 0.0
omega = 0.0

running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # WASD or arrow keys => control the agent
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        v = 250.0
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        v = -50.0
    else:
        v = 0.0

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        omega = 5.0
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        omega = -5.0
    else:
        omega = 0.0

    # Space => reset
    if keys[pygame.K_SPACE]:
        obs, info = env.reset()

    # Escape => quit
    if keys[pygame.K_ESCAPE]:
        running = False

    # o => select a random environment
    if keys[pygame.K_o]:
        env.select_environment(np.random.randint(0, env_count))
        env.reset()

    action = np.array([v, omega], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
