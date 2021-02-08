# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 08:21:46 2021

@author: othmane.mounjid
"""


### import libraries
import pandas as pd
import numpy as np
import utils.plotting.plotting as pltg
import matplotlib.pyplot as plt

### columns display 
pd.options.display.max_columns = 15

option_save = "" # "save"


### lsgd: read result files

config_files = [
            [1e-2, 'result/lsgd_reduced_gan_50_runganlsgdsimplified_lr_0.01.csv'],
            [5e-3, 'result/lsgd_reduced_gan_50_runganlsgdsimplified_lr_0.005.csv'],
            [1e-3, 'result/lsgd_reduced_gan_50_runganlsgdsimplified_lr_0.001.csv'],
            [5e-4, 'result/lsgd_reduced_gan_50_runganlsgdsimplified_lr_0.0005.csv'],
            [1e-4, 'result/lsgd_reduced_gan_50_runganlsgdsimplified_lr_0.0001.csv']
            ]

res, res_g_loss, labels = [], [], []
for lr, file_res  in config_files:
    
    df_sgd = pd.read_csv(file_res)
    
    max_iter = len(df_sgd['val_accuracy_test'])
    res_sgd = [ np.arange(1,max_iter+1), df_sgd['val_accuracy_test'].values[:max_iter]]
    res_sgd_reduced_g_loss = [ np.arange(1,max_iter+1), df_sgd['val_g_loss_test'].values[:max_iter]]
    
    res.append(res_sgd)
    res_g_loss.append(res_sgd_reduced_g_loss)
    labels.append('LSGD lr = {:.0e}'.format(lr))     



### print all 
pltg.Plot_plot(res[:-1],labels = labels[:-1],option=option_save,path ="image/readf_res_gan_allrun_lrvary",ImageName="",xtitle="",
               xlabel ="epoch", ylabel ="accuracy",Nset_tick_x = False, bbox_to_anchor_0 = (0.6,.95),
               mark = ['o','v', 's', '*'])
plt.show()

pltg.Plot_plot(res_g_loss[:-1],labels = labels[:-1],option=option_save,path ="image/readf_res_gan_allrun_gloss_lrvary",ImageName="",xtitle="",
               xlabel ="epoch", ylabel ="loss generator",Nset_tick_x = False, bbox_to_anchor_0 = (0.6,.95),
               mark = ['o','v', 's', '*'])
plt.show()
