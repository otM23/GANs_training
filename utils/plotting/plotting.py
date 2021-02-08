# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 01:18:37 2020

@author: othmane.mounjid
"""
### Import libraries 
import matplotlib.pyplot as plt


def Plot_plot(df,labels,option=False,path ="",ImageName="",xtitle="", xlabel ="", ylabel ="", fig = False, a = 0, b = 0, subplot0 = 0, linewidth= 3.0, Nset_tick_x = True, xlim_val = None, ylim_val = None,mark = None, col = 'blue',marksize=12, bbox_to_anchor_0 = (0.6,.95)):
    if mark is None:
        mark = ['o']*(len(df))
    if not fig:
        ax = plt.axes()
    else:
        ax = fig.add_subplot(a,b,subplot0)
    
    count = 0
    for elt in df:
        ax.plot(elt[0],elt[1], label = labels[count], linewidth= linewidth, marker = mark[count],  markersize = marksize)
        count +=1
    ax.set_title(xtitle,fontsize = 18)
    ax.set_xlabel(xlabel,fontsize = 18)
    ax.set_ylabel(ylabel,fontsize = 18)
    ax.grid(b=True)
    ax.legend(loc = 2, bbox_to_anchor = bbox_to_anchor_0)
    if Nset_tick_x:
        ax.set_xticklabels([])
    if xlim_val :
        ax.set_xlim(xlim_val)
    if ylim_val :
        ax.set_ylim(ylim_val)
    if option == "save" :
        plt.savefig(path+ImageName+".pdf", bbox_inches='tight') 
