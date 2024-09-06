# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:41:01 2023

@author: mym852
"""
# import required pacakges
import pandas as pd
import geopandas as gpd
from shapely import wkt

# File name of the scenario input file and output file of aggregated

path = ".\\"

Network_file =  path +'Results_Pathway\\CSV\\Pathway-Network-EnInd-DH-2023-08-10.csv'
NG_nodes = path + 'Input\\IGGIELGN_Network0_nodesgdf.csv'

Output_file = path + 'Results_Pathway\\CSV\\Pathway-03-Nodes-2023-09-26.csv'

# Function to read topology
def read_topo(file_name):
    file = pd.read_csv(file_name, index_col = 0)
    file['geometry'] = file['geometry'].apply(wkt.loads)
    file_gpd = gpd.GeoDataFrame(file, crs='EPSG:4326')
    return (file_gpd)

network = read_topo(Network_file)
network.drop(columns = {'source_flow', 'target_flow', 'net_flow', 'capacity_utilization', 'CAPEX', 'OPEX'}, inplace = True)

# create a graph with bi directional edges  for that a dataframe for each direction is created
Graph = pd.DataFrame()
Graph['source']=network['Source_ID'].astype(str)
Graph['target']=network['Target_ID'].astype(str)
Graph['combined_capacity']= round(network.combined_capacity_MWh_h.astype(int))
Graph['weight'] = round(network.Cost/1000).astype(int)

Graph1 = pd.DataFrame()
Graph1['source']=network['Target_ID'].astype(str)
Graph1['target']=network['Source_ID'].astype(str)
Graph1['combined_capacity']= round(network.combined_capacity_MWh_h.astype(int))
Graph1['weight'] = round(network.Cost/1000).astype(int)

# combine edges for one directed together to have a bidirectional graph 
Graph_double = pd.concat([Graph, Graph1])

# Create a list of all nodes
Source = [int(i) for i in Graph_double['source']]
Target = [int(i) for i in Graph_double['target']]

source_nodes = list(Source)
target_nodes = list(Target)

# create one list with unique identifiers of the nodes
network_nodes = list(set(source_nodes + target_nodes))


# read in all nodes of the NG pipeline netwprk 
nodes = pd.read_csv(NG_nodes, index_col=0)
nodes['geometry'] = nodes['geometry'].apply(wkt.loads)
nodes_gdf = gpd.GeoDataFrame(nodes, crs='EPSG:3857')

# select only the requried nodes
nodes_in_network = nodes_gdf.query('index in @network_nodes')

nodes_in_network.to_csv(Output_file)