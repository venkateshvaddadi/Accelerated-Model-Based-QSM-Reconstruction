#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:31:57 2020

@author: cds
"""


class Config :
    
    def __init__(self):
        
        
        self.gpu       = True
        self.gpuid     = 0
        self.batchsize = 5
        self.epochs    = 25