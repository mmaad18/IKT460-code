# -*- coding: utf-8 -*-
"""
Created on Sat May 21 00:16:24 2022

@author: mmbio
"""

import numpy as np
import math as mt
import fractions as fr
from scipy.odr import *

from geohelp import *




class featuresDetection:
    def __init__(self):
        # CONSTS 
        self.EPSILON = 10
        self.DELTA = 20
        self.SNUM = 6
        self.PMIN = 20 # Minimum number of points a seed segment should have
        self.GMAX = 20
        self.LMIN = 20 # Minimum length of a line segment 
        self.LR = 0 # Real length of a line segment 
        self.PR = 0 # Number of laser points contained in the line segment 
        
        self.SEED_SEGMENTS = []
        self.LINE_SEGMENTS = []
        self.LASER_POINTS = []
        self.LINE_PARAMS = None
        self.NP = len(self.LASER_POINTS) - 1
        
        
    def laserPointsSet(self, data):
        self.LASER_POINTS = []

        if data:
            for point in data:
                # distance, angle, position
                coordinates = GeometricHelper().AD2pos(point[0], point[1], point[2])
                
                self.LASER_POINTS.append([coordinates, point[1]])

        self.NP = len(self.LASER_POINTS) - 1
        

    def linearFunction(self, p, x):
        a, b = p
        
        return a*x + b
    
    
    def odrFit(self, laserPoints):
        x = np.array([i[0][0] for i in laserPoints])
        y = np.array([i[0][1] for i in laserPoints])
    
        # Create model for fitting 
        linearModel = Model(self.linearFunction)
        
        # Create a RealData object using our initiated data from above
        data = RealData(x, y)
        
        # Set up Orthogonal Distance Regression (ODR)
        odrModel = ODR(data, linearModel, beta0=[0., 0.])
        
        # Run regression 
        out = odrModel.run()
        a, b = out.beta
        
        return a, b
    
    
    def predictPoint(self, lineParams, sensedPoint, position):
        a, b = GeometricHelper().points2line(position, sensedPoint)
        
        params1 = GeometricHelper().line2GeneralForm(a, b)
        
        x_hat, y_hat = GeometricHelper().lineIntersectionGF(params1, lineParams)
        
        return x_hat, y_hat
    
    
    def seedSegmentDetection(self, position, breakPointIndex):
        flag = True
        
        self.NP = max(0, self.NP)
        self.SEED_SEGMENTS = []
        
        for i in range(breakPointIndex, (self.NP - self.PMIN)):
            predictedPointsToDraw = []
            
            j = i + self.SNUM
            
            a, c = self.odrFit(self.LASER_POINTS[i:j])
            
            params = GeometricHelper().line2GeneralForm(a, c)
            
            # Check if points belong in current seed segment 
            for k in range(i, j):
                predictedPoint = self.predictPoint(params, self.LASER_POINTS[k][0], position)
                
                predictedPointsToDraw.append(predictedPoint)
                
                d1 = GeometricHelper().euclidianDistance(predictedPoint, self.LASER_POINTS[k][0])
                
                if d1 > self.DELTA:
                    flag = False
                    break
                
            if flag:
                self.LINE_PARAMS = params
                
                return [self.LASER_POINTS[i:j], predictedPointsToDraw, (i, j)]
                
        return False
        
        
    def seedSegmentGrowing(self, indices, breakPoint):
        lineEq = self.LINE_PARAMS
        
        i, j = indices
        
        # Beginning and final points in the line segment 
        PB, PF = max(breakPoint, i-1), min(j+1, len(self.LASER_POINTS))
                    
        
        while GeometricHelper().euclidianDistance(lineEq, self.LASER_POINTS[PF][0]) < self.EPSILON:
            if PF > self.NP - 1:
                break
            else:
                a, b = self.odrFit(self.LASER_POINTS[PB:PF])
                
                lineEq = GeometricHelper().line2GeneralForm(a, b)
        
                point = self.LASER_POINTS[PF][0]
                
            PF = PF + 1
            nextPoint = self.LASER_POINTS[PF][0]
            
            if GeometricHelper().euclidianDistance(point, nextPoint) > self.GMAX:
                break
            
        PF = PF - 1
            
        while GeometricHelper().euclidianDistance(lineEq, self.LASER_POINTS[PB][0]) < self.EPSILON:
            if PB < breakPoint:
                break
            else:
                a, b = self.odrFit(self.LASER_POINTS[PB:PF])
                lineEq = GeometricHelper().line2GeneralForm(a, b)
                
                point = self.LASER_POINTS[PF][0]
                
                
            PB = PB - 1
            nextPoint = self.LASER_POINTS[PB][0]
        
        
            if GeometricHelper().euclidianDistance(point, nextPoint) > self.GMAX:
                break
            
        PB = PB + 1
        
        LR = GeometricHelper().euclidianDistance(self.LASER_POINTS[PB][0], self.LASER_POINTS[PF][0])
        PR = len(self.LASER_POINTS[PB:PF])
        
        if (LR >= self.LMIN) and (PR >= self.PMIN):
            self.LINE_PARAMS = lineEq
            
            a, b = GeometricHelper().lineGF2SI(lineEq[0], lineEq[1], lineEq[2])
            twoPoints = GeometricHelper().get2PointsFromLine(a, b)
            
            self.LINE_SEGMENTS.append((self.LASER_POINTS[PB + 1][0], self.LASER_POINTS[PF - 1][0]))
            
            return [self.LASER_POINTS[PB:PF], 
                    twoPoints,
                    (self.LASER_POINTS[PB + 1][0], self.LASER_POINTS[PF - 1][0]), 
                    PF,
                    lineEq,
                    (a, b)]
        
        else:
            return False
        
        














