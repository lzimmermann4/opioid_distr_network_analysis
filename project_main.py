# Description: This script includes the network analysis for the project for EPID 637.
# Author: Lauren Zimmermann
# Date: April 27, 2022
# Data: Uses publicly available Washington Post data on transactions of opiates within the US, 
#        which was collected from the Drug Enforcement Agency. 
#        The analysis focuses on sales of opiates in Michigan from 2006-2014.

## Import modules
import random as pr
import csv
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as pl
import community as comm

## Outline
# start with 1 county as test example or subset in some way (e.g. 5% random sample from one county from full data)
# note that quite dense and so network graph not informative, examine other formats
# degree distribution histogram to examine structural properties of network (e.g., connectivity)
# find top 5 distributor companies by degree centrality for a given node 
# remove node with highest degree centrality and examine how this affects the market structure
# perform Louvain partitioning to examine community structure/clustering

## Data Preparation 
# read in csv files
wp_data = pd.read_csv('~/arcos-mi-statewide-itemized.csv')
wp_data.head(5)
print('Dimensions of WP dataframe for Michigan: ', wp_data.shape)

# limit to one county for testing 
wp_data_test = wp_data[(wp_data['BUYER_COUNTY']=='WAYNE')]
print('Dimensions of WP dataframe for Wayne: ', wp_data_test.shape)

# remove duplicate rows and add weight for both test and full dataset
df2 = wp_data_test.groupby(['REPORTER_DEA_NO','BUYER_DEA_NO','BUYER_COUNTY']).size().reset_index(name='count')
df2_full = wp_data.groupby(['REPORTER_DEA_NO','BUYER_DEA_NO','BUYER_COUNTY']).size().reset_index(name='count')
df2.head(5)
df3 = wp_data_test.groupby(['REPORTER_DEA_NO','BUYER_DEA_NO','BUYER_COUNTY']).agg('sum').reset_index()
df3_full = wp_data.groupby(['REPORTER_DEA_NO','BUYER_DEA_NO','BUYER_COUNTY']).agg('sum').reset_index()
df3.head(5)
df_af = pd.merge(df3,df2,how="left",left_on=['REPORTER_DEA_NO','BUYER_DEA_NO','BUYER_COUNTY'],right_on=['REPORTER_DEA_NO','BUYER_DEA_NO','BUYER_COUNTY'])
df_af_full = pd.merge(df3_full,df2_full,how="left",left_on=['REPORTER_DEA_NO','BUYER_DEA_NO','BUYER_COUNTY'],right_on=['REPORTER_DEA_NO','BUYER_DEA_NO','BUYER_COUNTY'])
print('Dimensions of deduplicated WP dataframe for Wayne: ', df_af.shape)
print('Dimensions of deduplicated WP dataframe for Michigan: ', df_af_full.shape)
df_af2=df_af.sample(frac=0.05, replace=False, random_state=1)
print('Dimensions of 5pct random sample of deduplicated WP dataframe for Wayne: ', df_af2.shape)

# create test network directed graph (for 5pct random sample of Wayne County) and for full Michigan
G = nx.from_pandas_edgelist(df_af2,source='REPORTER_DEA_NO',target='BUYER_DEA_NO', edge_attr='QUANTITY', create_using=nx.DiGraph())
G_full = nx.from_pandas_edgelist(df_af_full,source='REPORTER_DEA_NO',target='BUYER_DEA_NO', edge_attr='QUANTITY', create_using=nx.DiGraph())
#G=nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
# return list of tuples
list_degree=list(G.degree()) 
# build node list and corresponding degree list
nodes , degree = map(list, zip(*list_degree)) 
pl.figure(figsize=(12,8))
nx.draw(G, nodelist=nodes, node_size=[(v * 5)+1 for v in degree])
pl.show()
#nx.draw(G,pos=nx.spring_layout(G,k=0.15),node_size = 50)
#pl.figure(figsize=(12, 8))  
#nx.draw(G, with_labels=False, font_weight='bold')
pl.savefig('project_fig_1a.png')

# alternative visualizations (to untangle hairball plot) for michigan
import nxviz as nv 
from nxviz import annotate
#mtrxpl=nv.MatrixPlot(G_full,group_by="BUYER_COUNTY", node_color_by="BUYER_COUNTY")
#annotate.matrix_group(G, group_by="BUYER_COUNTY")
mtrxpl=nv.MatrixPlot(G)
mtrxpl.draw()
pl.show()
a = nv.ArcPlot(G)
#pl.savefig('project_fig_1b.png')
a.draw()
a = nv.ArcPlot(G_full)
a.draw()
#pl.title("Arc Plot for Michigan")
#pl.xlabel("Nodes")
#pl.savefig('project_fig_1c.png')
#c=nv.CircosPlot
#c.draw()
#pl.show()


# create degree histogram for test of Wayne sample
import collections
degree_sequence = sorted([d for n, d in G.degree()], reverse=True) 
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
fig, ax = pl.subplots()
pl.figure(figsize=(12, 8)) 
pl.bar(deg, cnt, width=0.80, color='c')
pl.title("Degree Histogram for Wayne County (5 percent random sample)")
pl.ylabel("Count")
pl.xlabel("Degree")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)
# overlay hairball plot
pl.axes([0.4, 0.4, 0.5, 0.5])
Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
pos = nx.spring_layout(G)
pl.axis('off')
nx.draw_networkx_nodes(G, pos, node_size=20)
nx.draw_networkx_edges(G, pos, alpha=0.4)
pl.savefig('project_fig_2a.png')

# create degree histogram for full Michigan
degree_sequence = sorted([d for n, d in G_full.degree()], reverse=True) 
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
fig, ax = pl.subplots()
pl.figure(figsize=(12, 8)) 
pl.bar(deg, cnt, color='r')
pl.title("Degree Histogram for Michigan")
pl.ylabel("Count")
pl.xlabel("Degree")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)
print(max(deg)) # highest degree value is 1703
pl.savefig('project_fig_2b.png')

# compute measures of centrality for test of Wayne sample
dcent = nx.degree_centrality(G)
df_m1 = pd.DataFrame.from_dict({
    'Degree Node': list(dcent.keys()),
    'Degree Centrality': list(dcent.values())
})
df_m1=df_m1.sort_values('Degree Centrality', ascending=False)
df_m1=df_m1.reset_index(drop=True)

ccent = nx.closeness_centrality(G)
df_m2 = pd.DataFrame.from_dict({
    'Closeness Node': list(ccent.keys()),
    'Closeness Centrality': list(ccent.values())
})
df_m2=df_m2.sort_values('Closeness Centrality', ascending=False)
df_m2=df_m2.reset_index(drop=True)

bcent = nx.betweenness_centrality(G)
df_m3 = pd.DataFrame.from_dict({
    'Betweenness Node': list(bcent.keys()),
    'Betweenness Centrality': list(bcent.values())
})
df_m3=df_m3.sort_values('Betweenness Centrality', ascending=False)
df_m3=df_m3.reset_index(drop=True)
df_m3.head()

# compute measures of centrality for full Michigan
dcent = nx.degree_centrality(G_full)
df_m1 = pd.DataFrame.from_dict({
    'Degree Node': list(dcent.keys()),
    'Degree Centrality': list(dcent.values())
})
df_m1=df_m1.sort_values('Degree Centrality', ascending=False)
df_m1=df_m1.reset_index(drop=True)

ccent = nx.closeness_centrality(G_full)
df_m2 = pd.DataFrame.from_dict({
    'Closeness Node': list(ccent.keys()),
    'Closeness Centrality': list(ccent.values())
})
df_m2=df_m2.sort_values('Closeness Centrality', ascending=False)
df_m2=df_m2.reset_index(drop=True)

bcent = nx.betweenness_centrality(G_full)
df_m3 = pd.DataFrame.from_dict({
    'Betweenness Node': list(bcent.keys()),
    'Betweenness Centrality': list(bcent.values())
})
df_m3=df_m3.sort_values('Betweenness Centrality', ascending=False)
df_m3=df_m3.reset_index(drop=True)
df_m3.head()

dfs=pd.concat([df_m1,df_m2,df_m3],axis=1,sort = False)
dfs.head(5)
top5_centrality=dfs.head()
# write to csv file
top5_centrality.to_csv('project_centrality_michigan.csv',index=False)

# look at most central nodes
wp_data_mostcentral = wp_data[(wp_data['REPORTER_DEA_NO']=='PM0030849')]
wp_data_mostcentral.head()
df_af_full_mostcentral = df_af_full[(df_af_full['REPORTER_DEA_NO']=='PM0030849')]
df_af_full_mostcentral.head()
df3_full_mostcentral = df3_full[(df3_full['REPORTER_DEA_NO']=='PM0030849')]
df3_full_mostcentral['QUANTITY'].head()

# remove nodes with degree centrality higher than a specified value 
remove = [node for node,degree in dict(G.degree()).items() if degree > 2]
remove
print(len(remove))
G.remove_nodes_from(remove)

# recreate degree histogram for test of Wayne sample with removed nodes
import collections
degree_sequence = sorted([d for n, d in G.degree()], reverse=True) 
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
fig, ax = pl.subplots()
pl.figure(figsize=(12, 8)) 
pl.bar(deg, cnt, width=0.80, color='c')
pl.title("Degree Histogram for Wayne County (5 percent random sample)")
pl.ylabel("Count")
pl.xlabel("Degree")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)
# overlay hairball plot
pl.axes([0.4, 0.4, 0.5, 0.5])
Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
pos = nx.spring_layout(G)
pl.axis('off')
nx.draw_networkx_nodes(G, pos, node_size=20)
nx.draw_networkx_edges(G, pos, alpha=0.4)
pl.savefig('project_fig_3c.png')

# partition into Louvain communities to better learn about the structure of the network. 
# Note that partitioning is performed on undirected graph since Louvain partitioning does not exist on directed graph types, at the time of this study
# from netgraph import Graph
# obtain best partition into communities using Louvain method
G_Undirected = nx.Graph(G)
node_to_community = comm.best_partition(G_Undirected, weight='QUANTITY', resolution=1)
print(node_to_community) # there are 41 communities for Wayne sample

G_full_Undirected = nx.Graph(G_full)
node_to_community_full = comm.best_partition(G_full_Undirected, weight='QUANTITY', resolution=1)
print(node_to_community_full) # there are 41 communities for Wayne sample

# note that the below graph code is commented out because there are too many communities to color-code in the graph
#community_to_color = {
#    0 : 'tab:blue',
#    1 : 'tab:orange',
#    2 : 'tab:green',
#    3 : 'tab:red',
#    4 : 'tab:yellow',
#    ... #note that there are 41 communities so would need that many distinct colors
#}
#node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}

#Graph(G_Undirected,
#      node_color=node_color, node_edge_width=0, edge_alpha=0.1,
#      node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
#      edge_layout='bundled', edge_layout_kwargs=dict(k=2000),
#)
#pl.show()

## End of script