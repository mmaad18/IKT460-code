# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:22:21 2022

@author: mmbio
"""

import math
import pygame


class BuildEnvironment:
    
    def __init__(self, MapDimensions):
        pygame.init()
        
        self.pointCloud = []
        self.externalmap = pygame.image.load("SLAM_MAP_1H.png")
        self.mapH, self.mapW = MapDimensions
        
        self.MapWindowName = "LIDAR SIM"
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.mapW, self.mapH))
        self.map.blit(self.externalmap, (0, 0))
        
        # Colors 
        self.black = (0, 0, 0)
        self.grey = (70, 70, 70)
        self.blue = (0, 0, 255)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.white = (255, 255, 255)


    def AD2pos(self, distance, angle, position):
        x = distance * math.cos(angle) + position[0]
        y = - distance * math.sin(angle) + position[1]
        
        return (int(x), int(y))
    
    
    def dataStorage(self, data):
        print(len(self.pointCloud))
        
        if data != False:
            for element in data:
                point = self.AD2pos(element[0], element[1], element[2])
                
                if point not in self.pointCloud:
                    self.pointCloud.append(point)
    
    
    def showSensorData(self):
        self.infomap = self.map.copy()
        
        for point in self.pointCloud:
            self.infomap.set_at((int(point[0]), int(point[1])), (255, 0, 0))







