# -*- coding: utf-8 -*-
"""
Created on Sat May 21 00:21:45 2022

@author: mmbio
"""

import math as mt
import fractions as fr
import numpy as np


class GeometricHelper():
    #def __init__(self):
        
        
    def euclidianDistance(self, p1, p2):
        dx2 = (p2[0] - p1[0])**2
        dy2 = (p2[1] - p1[1])**2
        return mt.sqrt(dx2 + dy2)
        
    
    def point2LineProjection(self, point, a, b):
        x, y = point
        
        a2 = -1/a
        b2 = y - a2*x
        
        x_intersection = -(b - b2)/(a - a2)
        y_intersection = a2 * x_intersection + b2
        
        return x_intersection, y_intersection
    
    
    def rotatePoint(self, c, angle, point):
        x, y = point
        
        cx, cy = c
        
        s = mt.sin(angle)
        c = mt.cos(angle)
        
        # Translate "c" to origin
        x = x - cx
        y = y - cy
        
        # Rotate 
        xNew = x*c - y*s
        yNew = x*s - y*c
        
        # Translate "c" back 
        x = xNew + cx
        y = yNew + cy
        
        return x, y
    
    
    def interpolation(self, p1, p2, alpha):
        x1, y1 = p1
        x2, y2 = p2
        
        x = x2*alpha + x1*(1 - alpha)
        y = y2*alpha + y1*(1 - alpha)
        
        return x, y
    
    
    def line2GeneralForm(self, a, b):
        A, B, C = -a, 1, -b
        
        if A < 0:
            A, B, C = -A, -B, -C 
            
        denA = fr.Fraction(A).limit_denominator(1000).as_integer_ratio()[1]
        denC = fr.Fraction(C).limit_denominator(1000).as_integer_ratio()[1]
        
        gcd = np.gcd(denA, denC)
        lcm = denA * denC / gcd
        
        A = A * lcm
        B = B * lcm
        C = C * lcm
        
        return A, B, C
    
    
    def point2LineDistanceGF(self, params, point):
        A, B, C = params
        return abs(A*point[0] + B*point[1] + C) / mt.sqrt(A**2 + B**2)
    
    
    def get2PointsFromLine(self, a, b, x = 5, x2 = 2000):
        y = a*x + b
        y2 = a*x2 + b
        
        return [(x, y), (x2, y2)]
    
    
    def lineGF2SI(self, A, B, C):
        a = -A/B
        b = -C/B
        
        return a, b
    
    
    # Will produce error if lines are paralell 
    def lineIntersectionGF(self, params1, params2):
        A1, B1, C1 = params1
        A2, B2, C2 = params2
    
        x = (C1*B2 - B1*C2) / (B1*A2 - A1*B2)
        y = (A1*C2 - A2*C1) / (B1*A2 - A1*B2)
        
        return x, y
    
    
    def points2line(self, point1, point2):
        a, b = 0, 0
        
        if point1[0] != point2[0]:
            a = (point2[1] - point1[1]) / (point2[0] - point1[0])
            b = point2[1] - a*point2[0]
        
        return a, b
    
    
    def AD2pos(self, distance, angle, position):
        x = distance * mt.cos(angle) + position[0]
        y = - distance * mt.sin(angle) + position[1]
        
        return (int(x), int(y))
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    