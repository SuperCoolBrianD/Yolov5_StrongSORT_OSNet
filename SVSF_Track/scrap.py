# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:11:44 2022

@author: salma
"""

import numpy as np
from CCS_Conversion_Functions import*
from scipy.interpolate import CubicSpline

x = np.arange(10)
y = np.sin(x)

cs = CubicSpline(x, y)

T = generateArcLengths(cs)