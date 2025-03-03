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
                
            if pygame.mouse.get_focused():
                sensorON = True
            elif not pygame.mouse.get_focused():
                sensorON = False
                
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




def randomColor():
    levels = range(32, 256, 32)
    
    return tuple(rd.choice(levels) for _ in range(3))


def main2():
    pygame.init()

    featureMap = features.featuresDetection()
    environment = env.BuildEnvironment((600, 1200))
    originalMap = environment.map.copy()
    laser = sensors.LaserSensor(200, originalMap, Uncertainty = (0.5, 0.01))
    environment.map.fill((255, 255, 255))
    environment.infomap = environment.map.copy()
    originalMap = environment.map.copy()
    running = True
    FEATURE_DETECTION = True
    BREAK_POINT_INDEX = 0
    
    while running:
        environment.infomap = originalMap.copy()
        FEATURE_DETECTION = True
        BREAK_POINT_INDEX = 0
        ENDPOINTS = [0, 0]
        sensorON = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        if pygame.mouse.get_focused():
            sensorON = True
        elif not pygame.mouse.get_focused():
            sensorON = False
    
        if sensorON:
            position = pygame.mouse.get_pos()
            laser.position = position
            sensorData = laser.senseObstacle()
            
            featureMap.laserPointsSet(sensorData)
            
            while BREAK_POINT_INDEX < (featureMap.NP - featureMap.PMIN):
                seedSeg = featureMap.seedSegmentDetection(laser.position, BREAK_POINT_INDEX)
                
                if seedSeg == False:
                    break
                else:
                    seedSegment = seedSeg[0]
                    
                    PREDICTED_POINTS_TO_DRAW = seedSeg[1]
                    
                    INDICES = seedSeg[2]
                    
                    results = featureMap.seedSegmentGrowing(INDICES, BREAK_POINT_INDEX)
                    
                    if results == False:
                        BREAK_POINT_INDEX = INDICES[1]
                        continue
                    else:
                        lineEq = results[1]
                        a, c = results[5]
                        lineSeg = results[0]
                        OUTERMOST = results[2]
                        BREAK_POINT_INDEX = results[3]
                        
                        ENDPOINTS[0] = geohelp.GeometricHelper().point2LineProjection(OUTERMOST[0], a, c)
                        ENDPOINTS[1] = geohelp.GeometricHelper().point2LineProjection(OUTERMOST[0], a, c)
                        
                        COLOR = randomColor()
                        
                        for point in lineSeg:
                            environment.infomap.set_at((int(point[0][0]), int(point[0][1])), (0, 255, 0))
                    
                            pygame.draw.circle(environment.infomap, COLOR, 
                               (int(point[0][0]), int(point[0][1])), 2, 0)
                            
                        pygame.draw.line(environment.infomap, (255, 0, 0), ENDPOINTS[0], ENDPOINTS[1], 2)
                        environment.dataStorage(sensorData)
                        
        environment.map.blit(environment.infomap, (0, 0))
        pygame.display.update()
        
    pygame.quit()


main()













