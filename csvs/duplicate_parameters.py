#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:25:39 2023

@author: mariahboudreau
"""


#Make CSV have 1000 entries of the same parameters

import numpy as np
import pandas as pd


# Read in the file

df = pd.read_csv('violin_code_vacc_parameters.csv')


df = pd.concat([df]*1000)


# Rewrite the file with the copies

df.to_csv("violin_code_vacc_parameters.csv")
