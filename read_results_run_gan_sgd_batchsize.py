# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:03:57 2021

@author: othmane.mounjid
"""


### import libraries
import pandas as pd
import numpy as np
import plotting.plotting as pltg
import matplotlib.pyplot as plt

### columns display 
pd.options.display.max_columns = 15

option_save = "" # "save"#"save"


### res sgd

config_files = [
            [16, 'result/sgd_reduced_gan_50_rungansgdsimplified_batch_16.csv'],
            [32, 'result/sgd_reduced_gan_50_rungansgdsimplified_batch_32.csv'],
            [64, 'result/sgd_reduced_gan_50_rungansgdsimplified_batch_64.csv'],
            [128, 'result/sgd_reduced_gan_50_rungansgdsimplified_batch_128.csv'],
            [256, 'result/sgd_reduced_gan_50_rungansgdsimplified_batch_256.csv']
            ]

res, res_g_loss, labels = [], [], []
for lr, file_res  in config_files:
    
    df_sgd = pd.read_csv(file_res)
    
    max_iter = len(df_sgd['val_accuracy_test'])
    res_sgd = [ np.arange(1,max_iter+1), df_sgd['val_accuracy_test'].values[:max_iter]]
    res_sgd_reduced_g_loss = [ np.arange(1,max_iter+1), df_sgd['val_g_loss_test'].values[:max_iter]]
    
    res.append(res_sgd)
    res_g_loss.append(res_sgd_reduced_g_loss)
    labels.append('SGD batch = {}'.format(lr))    


### print all 
pltg.Plot_plot(res,labels = labels,option=option_save,path ="image/readf_res_gan_allrun_bs",ImageName="",xtitle="",
               xlabel ="epoch", ylabel ="accuracy",Nset_tick_x = False, bbox_to_anchor_0 = (0.61,1.02),
               mark = ['o','v', 's', '*', 'P'])
plt.show()

#pltg.Plot_plot(res_g_loss,labels = labels,option=option_save,path ="image_server/readf_res_gan_allrun6_gloss_batch",ImageName="",xtitle="",
#               xlabel ="epoch", ylabel ="loss generator",Nset_tick_x = False, bbox_to_anchor_0 = (0.65,.95),
#               mark = ['o','v', 's', '*'])
#plt.show()

#### print first figure
#pltg.Plot_plot([res[0]], labels = [''], option=option_save,path ="image_server/readf_res_gan_SGDreducedrun_fluc",ImageName="",xtitle="",
#               xlabel ="epoch", ylabel ="accuracy",Nset_tick_x = False)
#plt.show()
#
#res_g_loss1 = [[res_g_loss[0][0][5:], res_g_loss[0][1][5:]]]
#pltg.Plot_plot(res_g_loss1, labels = [''], option=option_save,path ="image_server/readf_res_gan_SGDreducedrun_generatorloss_fluc",ImageName="",xtitle="",
#               xlabel ="epoch", ylabel ="loss generator",Nset_tick_x = False)
#plt.show()
