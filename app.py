import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
import EntityNormalization as ne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import networkx as nx
import matplotlib.pyplot as plt
from Visualization import *


# def plot_fig(node_list, edge_list, save_name):
#     # G = nx.random_geometric_graph(40, 0.125)
#     G = nx.Graph()
#     node_len = len(node_list)
#     node_label = [i.replace('the ', '').replace('The ', '') for i in node_list]
#     edge_pair = edge_list
#     # edge_weight = np.sqrt(np.array([i[1] for i in edge_pair]))
#
#     person_node = list(set([i[0] for i in edge_pair]))
#     node = np.linspace(1, node_len, node_len, dtype=int) - 1
#     node_color = ['LightSkyBlue' if i in person_node else 'DarkSlateGrey' for i in node]
#
#     G.add_nodes_from(node)
#     G.add_weighted_edges_from(edge_pair)
#
#     # pos = nx.kamada_kawai_layout(G)
#     pos = nx.spring_layout(G, k=5/np.sqrt(node_len))
#     # pos = nx.spiral_layout(G)
#     # pos = nx.multipartite_layout(G)
#     # pos = nx.spectral_layout(G)
#     # pos = nx.random_layout(G)
#     # pos = nx.bipartite_layout(G, person_node)
#
#
#     edge_x = []
#     edge_y = []
#     node_size = []
#     for edge in G.edges():
#         # x0, y0 = G.nodes[edge[0]]['pos']
#         # x1, y1 = G.nodes[edge[1]]['pos']
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x.append(x0)
#         edge_x.append(x1)
#         edge_x.append(None)
#         edge_y.append(y0)
#         edge_y.append(y1)
#         edge_y.append(None)
#
#
#     edge_trace = go.Scatter(
#         x=edge_x, y=edge_y,
#         line=dict(width=0.5, color='#888'),
#         hoverinfo='none',
#         mode='lines'
#
#     )
#
#     node_x = []
#     node_y = []
#     for node in G.nodes():
#         # x, y = G.nodes[node]['pos']
#         x, y = pos[node]
#         node_x.append(x)
#         node_y.append(y)
#
#     node_trace = go.Scatter(
#         x=node_x, y=node_y,
#         mode='markers+text',
#         text=node_label,
#         textposition='top center',
#         hoverinfo='text',
#         marker=dict(
#             showscale=False,
#             # colorscale options
#             #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
#             #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
#             #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
#             # colorscale='YlGnBu',
#             reversescale=True,
#             opacity = 0.9,
#             color=node_color,
#             size=10,
#             # colorbar=dict(
#             #     thickness=15,
#             #     title='Node Connections',
#             #     xanchor='left',
#             #     titleside='right'
#             # ),
#             line_width=2))
#
#
#     node_adjacencies = np.zeros(node_len)
#     node_text = node_label
#     for i in edge_pair:
#         node_adjacencies[i[0]] += i[2]
#         node_adjacencies[i[1]] += i[2]
#     node_adjacencies = node_adjacencies
#     # for node, adjacencies in enumerate(G.adjacency()):
#     #     node_adjacencies.append(len(adjacencies[1]))
#     #     node_text.append('# of connections: '+str(len(adjacencies[1])))
#
#     # node_trace.marker.color = node_adjacencies
#     node_trace.marker.size = node_adjacencies
#     node_trace.text = node_text
#
#     year_num = save_name[:4]
#     q_num = int((int(save_name[5:7]) - 1) / 3 + 1)
#     fig = go.Figure(data=[edge_trace, node_trace],
#                  layout=go.Layout(
#                     title='IRTM NBA News Relation Visulization: {0} Q{1} Human-Team Relations'.format(year_num, q_num),
#                     titlefont_size=35,
#                     showlegend=False,
#                     hovermode='closest',
#                     margin=dict(b=20,l=5,r=5,t=40),
#                     annotations=[ dict(
#                         text="{0} Q{1} Human-Team Relations based on NBA News Report".format(year_num, q_num),
#                         showarrow=False,
#                         xref="paper", yref="paper",
#                         x=0.005, y=-0.002 ) ],
#                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#                     )
#
#     fig.update_layout(
#         autosize=False,
#         width=1500,
#         height=1500,
#         margin=dict(
#             l=50,
#             r=50,
#             b=100,
#             t=100,
#             pad=4
#         ),
#     )
#
#     if not os.path.exists("images"):
#         os.mkdir("images")
#     fig.write_image("images/{}_spring_layout.png".format(save_name))
#     return fig


for i in range(len(ne.node_list)):
    fig = plot_fig(ne.node_list[i], ne.edge_list[i], ne.date_point[i])


# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
#
# app.layout = html.Div([
#     dcc.Graph(
#         id='life-exp-vs-gdp',
#         figure=fig
#     )
# ])
#
# if __name__ == '__main__':
#     app.run_server(debug=True)