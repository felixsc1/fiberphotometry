# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 09:13:23 2018

@author: felix
"""

import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected=True)


def CreateLayout(stim_events,duration,inputframe,titleinput):
    shapes1 = list()
    
#    for i in stim_events:
#        shapes1.append({
#            'type': 'line',
#            'xref': 'x',
#            'yref': 'y',
#            'x0': i,
#            'y0': 0,
#            'x1': i,
#            'y1': inputframe.max().max(),
#            'opacity':0.4,
#            'layer': 'above',
#            'line':{
#                'color':'red',
#            },
#        })
#    
    
    for i in stim_events:
        shapes1.append({
            'type': 'rect',
            'xref': 'x',
            'yref': 'y',
            'x0': i,
            'y0': 0,
            'x1': i+duration,
            'y1': inputframe.max().max(),
            'opacity':0.3,
            'fillcolor':'red',
            'line':{
                'width':0,
            },
        })
    
        
    layout = go.Layout(
        title=titleinput,
        yaxis=dict(
        title='%'
        ),
        xaxis=dict(
        title='time [s]'
        ),
         shapes=shapes1
        )
    return layout

def plotlyplot(inputframe,stim_events,duration=0.5,plotmean=True,titleinput='',
               normalize=False):
    if normalize:
        inputframe=inputframe.divide(inputframe.max())
        
    names=inputframe.columns
    xaxis=inputframe.index.values
    traceA = []
    k=0
    for i in range(inputframe.columns.size):
        traceA.append(go.Scatter(x=xaxis, y=inputframe.iloc[:,i].values,
                                 mode = 'lines' ,line=dict(width=1.0),
                                 name=names[k], showlegend=True,
                                 connectgaps=True))
        k+=1
    
    if plotmean==True:
        traceA.append(go.Scatter(x=xaxis, y=inputframe.mean(axis=1).values,
                                 mode = 'lines' ,line=dict(width=1.5,color='black'),
                                 name='mean', showlegend=True,
                                 connectgaps=True))

    
    fig = go.Figure(data=traceA, layout=CreateLayout(stim_events,duration,inputframe,titleinput))
    py.offline.iplot(fig)   