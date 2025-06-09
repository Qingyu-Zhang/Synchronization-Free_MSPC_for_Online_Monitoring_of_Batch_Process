#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 11:27:54 2025

@author: zqy
"""

import pandas as pd 
from src.preprocessing import load_and_concatenate_batches

df_augmented = load_and_concatenate_batches("data/B41_NOC/")
print(df_augmented.shape)
df_augmented.to_csv("data/example_augmented.csv", index=False)


df_newbatch1=load_and_concatenate_batches("data/B64_Faulty/")
print(df_newbatch1.shape)
df_newbatch1.to_csv('data/example_newbatch_B64_Faulty.csv')

df_newbatch2=load_and_concatenate_batches("data/B89_Faulty/")
print(df_newbatch2.shape)
df_newbatch2.to_csv("data/example_newbatch_B89_Faulty.csv")
