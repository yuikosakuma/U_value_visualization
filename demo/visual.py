import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

import pandas as pd
import numpy as np
from thermal_performance import calc_u, output_text

from textwrap import dedent

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Graph(id='graph'),
    html.Div([
        dcc.Markdown(dedent('''
            ### House thermal performance visualization
            ''')),
        ]),
    html.Div([
        html.Label('window size'),
        dcc.Slider(id='window_slider', min=1, max=5, marks={i:'{}'.format(i) for i in range(1, 6)}, value=1),
        ], style={'columnCount':1}),
    html.Div([
        html.Label('Home'),
        dcc.Checklist(
            id='home_id',
            options=[
              {'label':'home_1 : occupancy changes', 'value': 1},
              {'label':'home_2 : ventilation changes', 'value': 2},
              {'label':'home_3 : baseline', 'value': 3},
              {'label':'home_4 : occupancy changes', 'value': 4},
              {'label':'home_5 : ventilation changes', 'value': 5},
              {'label':'home_6 : baseline', 'value': 6}
                ],
            #values=['home_1','home_2','home_3','home_4','home_5','home_6']
            values=[x for x in range(1, 7)]
            ),
        html.Hr(style={'borderTop':'none'}),
        html.Label('Model'),
        dcc.RadioItems(
            id='model_id',
            options=[
                {'label':'1R1C', 'value': 1},
                {'label':'2R1C', 'value': 2},
                {'label':'3R1C', 'value': 3},
                ],
            value = 1
            ),
        html.Hr(style={'borderTop':'none'}),
        html.Label('Description'),
        dcc.Markdown(dedent('''
        - Designed U-value
            - Home 1, 2, 3 : 0.45
            - Home 4, 5, 6 : 0.82
            ''')),

        ], style={'columnCount':3}),

    ])

@app.callback(
        dash.dependencies.Output('graph', 'figure'),
        [dash.dependencies.Input('window_slider', 'value'),
            dash.dependencies.Input('home_id', 'values'),
            dash.dependencies.Input('model_id', 'value')]
        )

def update_figure(window, home_list, model):
    traces = []
    for home_num in home_list:
        idx, u_list, residual_list = calc_u(window, home_num, model, 0.02)
        text_list = output_text(home_num)
        traces.append(
                go.Scattergl(
                    x=idx[:-1],
                    y=u_list[:-1],
                    mode='markers',
                    text=text_list[:-1],
                    marker={
                        'size':10
                        },
                    name = 'home_{}'.format(home_num)
                    )
                )
    return {
            'data':traces,
            'layout':go.Layout(
                xaxis={'title':'Time', 'showgrid':True, 'mirror':'ticks', 'showline':True, 'linewidth':1},
                #yaxis={'title':'$d, r \\text{ (solar radius)}$', 'range':[0, 1.1], 'showgrid':True,'showline':True, 'mirror':'ticks', 'linewidth':1}
                yaxis={'title':'Heat transfer coefficient, U [W/m^2K]', 'range':[0, 1.1], 'showgrid':True,'showline':True, 'mirror':'ticks', 'linewidth':1}
                #legend={'x' : 0, 'y' : 1}
                )
            } 

if __name__ == '__main__':
        app.run_server(host='0.0.0.0',debug=True)
