# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:35:32 2022

@author: mmbio
"""

import env, sensors, features, geohelp
import pygame
import math
import random as rd


def main():
    pygame.init()
    
    environment = env.BuildEnvironment((600, 1200))
    environment.originalMap = environment.map.copy()
    laser = sensors.LaserSensor(200, environment.originalMap, Uncertainty = (0.5, 0.01))
    environment.map.fill((0, 0, 0))
    environment.infomap = environment.map.copy()
    
    running = True
    
    while running:
        
        sensorON = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            sensorON = pygame.mouse.get_focused()
                
        if sensorON:
            position = pygame.mouse.get_pos()
            laser.position = position
            sensorData = laser.senseObstacle()
            
            environment.dataStorage(sensorData)
            environment.showSensorData()
            
        # Update and show drawing of map 
        environment.map.blit(environment.infomap, (0, 0))
        pygame.display.update()
    
    pygame.quit()



main()













