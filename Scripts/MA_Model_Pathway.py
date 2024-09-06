#!/usr/bin/env python
# coding: utf-8

# In[1303]:


import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from shapely import wkt
from datetime import date

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from shapely.geometry import Point

from matplotlib import cm
from math import sqrt
from pulp import*
import re

# Variables and input path and files
Date = str(date.today())

save_geojson= False
save_csv = False
# overall network
path = ".\\" 

# Target pathway file 
Scenario_file = path + 'Results_Pathway\\Inputs\\04_Scenario-EnInd-Trucks-DH-2023-08-02.csv'
#Previous pathway file 
Network_file =  path + 'Results\\CSV\\Base-Network-2023-08-02.csv'
# base network
Base_network = path + 'Results\\CSV\\Base-Network-2023-08-02.csv'

# Output file
to_file_geojson =  path + 'Results_Pathway\\GeoJSON\\Pathway-Network-NonEnInd-'+Date+'.geojson'
to_file_csv =  path + 'Results_Pathway\\CSV\\Pathway-Network-NonEnInd-'+Date+'.csv'

''' Read Topology of base and pathway file '''

def read_topo(file_name):
    file = pd.read_csv(file_name, index_col = 0)
    file['geometry'] = file['geometry'].apply(wkt.loads)
    file_gpd = gpd.GeoDataFrame(file, crs='EPSG:4326')
    return (file_gpd)


network = read_topo(Network_file)
base_network = read_topo(Base_network)
network.drop(columns = {'source_flow', 'target_flow', 'net_flow', 'capacity_utilization'}, inplace = True)


''' Physical properties of natural gas and hydrogen '''
# Physical Properties of Hydrogen and Natural Gas to calculate capacity and diameter in later steps

pi = 3.14 

LHV_NG = 13.9/1000 # MWh/kg https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html
LHV_NG_m3 = 35.8 # MJ/m^3 https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html
v_NG = 25 #m/s https://onlinelibrary.wiley.com/doi/full/10.1002/ese3.932
rho_NG = 52.548 #kg/m^3 https://www.unitrove.com/engineering/tools/gas/natural-gas-density7

LHV_H2 = 33.3/1000 # MWh/kg
v_H2 = 15 # m/s
rho_H2 = 5.246 #kg/m^3 https://cmb.tech/hydrogen-tools


''' Adjust Capacities for functioning model '''
# Industry  & Trucks & DH
network.at[73,'combined_capacity_MWh_h'] = 2600
network.at[94,'combined_capacity_MWh_h'] = 7100
network.at[188,'combined_capacity_MWh_h'] = 13000
network.at[556,'combined_capacity_MWh_h'] = 7100
network.at[702,'combined_capacity_MWh_h'] = 4100
network.at[714,'combined_capacity_MWh_h'] = 4100
network.at[897,'combined_capacity_MWh_h'] = 1500

# Industry & DH
# network.at[26,'combined_capacity_MWh_h'] = 1700
# network.at[73,'combined_capacity_MWh_h'] = 3500
# network.at[107,'combined_capacity_MWh_h'] = 10200
# network.at[136,'combined_capacity_MWh_h'] = 10200
# network.at[156,'combined_capacity_MWh_h'] = 10200
# network.at[158,'combined_capacity_MWh_h'] = 10200
# network.at[702,'combined_capacity_MWh_h'] = 4200
# network.at[714,'combined_capacity_MWh_h'] = 4200



''' Read in the file from Scenario development '''
aggregated = pd.read_csv(Scenario_file, index_col=0)
aggregated['geometry'] = aggregated['geometry'].apply(wkt.loads)
aggregated = gpd.GeoDataFrame(aggregated, crs='EPSG:4326')

'''Read geodata on Europe and Germany '''
Europe = gpd.read_file(path +'Input\\NUTS_RG_60M_2021_4326.shp', crs = 'EPSG:4326')
Europe = Europe.loc[(Europe['LEVL_CODE'] == 3)]
Europe.drop(list(range(778,787)), inplace=True) # Spanische Kolonien Nuts 3
Europe.drop(list(range(905,910)), inplace=True) # Französische Kolonien Nuts 3 
Europe.drop([1138,1139], inplace=True) #Portugiesische Kolonien Nuts 3
Europe.drop([1950, 1951], inplace=True) #Svalbard
Europe.drop([537, 960], inplace=True) #Island
DE = Europe.loc[(Europe['CNTR_CODE'] == 'DE')]


# Balance the supply in Berlin to distribute in Berlin

Berlin = DE[DE['NUTS_ID']== 'DE300']
Berlin.to_crs('EPSG:4326', inplace = True)
points_within_Berlin = gpd.sjoin(aggregated, Berlin, predicate='within')
demand_at_nodes = int((-points_within_Berlin.Demand.sum()+99)/(len(points_within_Berlin.index)-1))
high_demand = points_within_Berlin.index[points_within_Berlin.Demand !=0]
aggregated.at[high_demand[0],'demand'] = -99
points_within_Berlin.drop([high_demand[0]], inplace = True)

# Needs to be manually adjusted to work in specific scenarios
for i in points_within_Berlin.index: 
    aggregated.demand[i] = demand_at_nodes

''' Bring input data into right format for algorithm'''

aggregated.demand = aggregated.demand.astype(int)


'''Create Graph with bi directional edges'''

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

''' Create all inputs for optimization'''
Graph_double = pd.concat([Graph, Graph1])

Supply = pd.Series(aggregated.demand.values,index=aggregated.ID.astype(str)).to_dict()

Source = [str(i) for i in Graph_double['source']]
Target = [str(i) for i in Graph_double['target']]
source_nodes = list(Source)
target_nodes = list(Target)

# Nodes
Nodes = list(set(source_nodes + target_nodes))
Nodes.sort(key = int)

# Edges
E = [(Source[i], Target[i]) for i in range(len(Source))]

# Assign cost and capacity for each edge
Cost = defaultdict(dict)
Capacity = defaultdict(dict)
for (i,j) in E:
    Cost[i][j] = int(Graph_double[((Graph_double['source'] == i) & (Graph_double['target'] == j))]['weight'])
    Capacity[i][j] = int(Graph_double[((Graph_double['source'] == i) & (Graph_double['target'] == j))]['combined_capacity'])

''' OPTIMIZATION MODEL '''
#Create a LP Model
model1 = LpProblem('model', LpMinimize)

# Create Flow variables
flow_var = LpVariable.dicts("Flow_Var",E, 0,None, LpInteger)  # d_r in the example
flow_used = LpVariable.dicts("Flow_Used", E, 0, 1,  LpInteger)  # r in the example
a_very_large_number = 1000000000000000 # replace or calculate

#Objective function
model1 += lpSum([(flow_used[(i,j)], 1)*Cost[i][j] for (i,j) in E])-lpSum([Cost[i][j] for (i,j) in E])


# Flow Constraints
for i in Nodes: 
    model1 += (lpSum([flow_var[i,j] for j in Nodes if (i,j) in E]) - lpSum([flow_var[k,i] for k in Nodes if (k,i) in E])) == Supply[i] 

# Capacity Constraints
for (i,j) in E: 
    model1 += flow_var[i,j] <= Capacity[i][j] 

# Binary Variable Constraint
for (i, j) in E:
    model1 += a_very_large_number * flow_used[(i,j)]>= flow_var[(i,j)] #,"Route used"

# Solve with Solver
status = model1.solve(GUROBI(msg = True))

''' RESULT PROCESSING'''

# Save Results in lists
Edges = []
Flow = []
for v in model1.variables():
    Edges.append(re.findall(r'\d+', v.name))
    Flow.append(v.varValue)


# Process Results in Edges and Flow 
Flow_df = pd.DataFrame()
Source = []
Target = []
Flow1 = []
for c in range(int(len(Edges)/2),len(Edges)):
    Source.append(Edges[c][0])
    Target.append(Edges[c][1])
    Flow1.append(Flow[c])

# Save Results
Flow_df = pd.DataFrame(list(zip(Source, Target, Flow1)), columns = ['source', 'target', 'flow'])

# Filter the original dataframe with the results from the optimitaion and append the results 

network['source_flow'] = [Flow_df[(Flow_df['source']==source) & (Flow_df['target']==target)]['flow'].values[0] for source,target in network[['Source_ID','Target_ID']].astype(str).values]
network['target_flow'] = [-Flow_df[(Flow_df['target']==source) & (Flow_df['source']==target)]['flow'].values[0] for source,target in network[['Source_ID','Target_ID']].astype(str).values]
network['net_flow'] = abs(network.source_flow + network.target_flow)
network_flow = network[(network.source_flow > 0) | (network.target_flow < 0)]

'''Capacity Utilizatiion Calculation'''
network_flow['capacity_utilization'] = ((network_flow['net_flow']/network_flow['combined_capacity_MWh_h'])*100)


'''Cost Calculation '''
network_flow.Cost.sum()
network_flow.CAPEX.sum()
network_flow.OPEX.sum()

'''Length Calculation'''
network_flow.length_m.astype(int).sum()/1000


'''Save Output'''

# bring results in right format
network_flow.net_flow = network_flow.net_flow.astype(int) 

if save_geojson: 
    network_arcgis = network_flow[['geometry', 'capacity_MWh_h', 'diameter_mm', 'length_m', 'net_flow', 'Cost', 'OPEX', 'CAPEX', 'capacity_utilization']]
    network_arcgis.to_file(to_file_geojson, driver='GeoJSON') 
if save_csv: 
    network_flow.to_csv(to_file_csv)
''


''' Visualisations '''


network_flow = network_flow.to_crs(crs= 4326)
network = network.to_crs(crs=4326)


# Dataframes for visualizing input data
nodes_supply= aggregated[aggregated['Supply']>0]
nodes_supply.set_crs('EPSG:4326')
nodes_demand= aggregated[aggregated['Demand']>0]
nodes_demand.set_crs('EPSG:4326')
demand_nodes = aggregated[aggregated['demand']!=0]


''' Visualisation of network flow by color'''


fig,(ax) = plt.subplots(1,1 , figsize =(15,15))
DE.plot(ax=ax, edgecolor= 'lightgrey', facecolor= 'none')

base_network.plot(ax=ax, 
            color='grey', 
            linewidth =2 , 
            label = 'Hydrogen Network 2045')
network_flow.plot(ax=ax, 
                  color= 'blue',
                  linewidth = 2,
                  label = 'Hydrogen Network HtA_ITH')

aggregated[aggregated.demand!=0].plot(ax=ax, 
                color= 'orange', 
                markersize = 30,
                label = 'Supply & Demand Nodes')

leg = plt.legend(loc = 'lower center', fontsize = 16, ncol = 2, markerscale = 2)
for legobj in leg.legend_handles:
    legobj.set_linewidth(5.0)
plt.axis('off')



''' Visulaisation of Flow by thinkness'''

max_flow = network_flow[network_flow. net_flow == max(network_flow. net_flow)]
max_supply = nodes_supply[nodes_supply.demand== max(nodes_supply.demand)]
max_demand = nodes_demand[nodes_demand.demand== max(nodes_demand.demand)]


fig,(ax) = plt.subplots(1,1 , figsize =(15,15))
DE.plot(ax=ax, edgecolor= 'lightgrey', facecolor = 'none')

max_supply.plot(ax=ax, 
                color= 'green', 
                markersize = 30,
                label = 'Supply')
max_demand.plot(ax=ax, 
                color= 'red', 
                markersize = 30,
                label = 'Demand')
max_flow.plot(ax=ax, 
                  color = 'blue', 
                  linewidth = 2, 
                  label = 'Hydrogen Flow [MWh/h]')

network_flow.plot(ax=ax, 
                  color = 'blue', 
                  linewidth = network_flow.net_flow.values/2000)



nodes_supply.plot(ax=ax, 
                color= 'green', 
                markersize = nodes_supply.Supply.values/50)

nodes_demand.plot(ax=ax, 
                color= 'red', 
                markersize = nodes_demand.Demand.values/50)



leg = plt.legend(loc = 'lower center', fontsize = 16, ncol = 2, markerscale = 2)
for legobj in leg.legend_handles:
    legobj.set_linewidth(5.0)
plt.axis('off')



''' Capacity Utilization Visualisation'''

# Manipulated in Ind & CHP 
# Manipulated = network_flow[(network_flow.key == '(47, 616)')   |(network_flow.key == '(212, 468)')  \
#                            | (network_flow.key == '(315, 469)') |(network_flow.key == '(382, 497)')  | (network_flow.key == '(250, 315)')\
#                            | (network_flow.key == '(250, 497)') | (network_flow.key == '(61, 467)') | (network_flow.key == '(61, 246)')]
# # Manipulated in Ind & Trucks & CHP 
Manipulated = network_flow[(network_flow.key == '(212, 468)')   |(network_flow.key == '(1, 468)')   | (network_flow.key == '(83, 239)') \
                           | (network_flow.key == '(514, 650)') |(network_flow.key == '(61, 467)')  | (network_flow.key == '(61, 246)')\
                           | (network_flow.key == '(250, 497)') | (network_flow.key == '(183, 605)')| (network_flow.key == '(108, 183)')\
                           | (network_flow.key == '(101, 211)')]
Capacity_75 = network_flow[network_flow.capacity_utilization>75]
Capacity_50 = network_flow[network_flow.capacity_utilization>50]
Capacity_25 = network_flow[network_flow.capacity_utilization>25]
Capacity_0 = network_flow[network_flow.capacity_utilization>0]



fig,(ax) = plt.subplots(1,1 , figsize =(15,15))
DE.plot(ax=ax, edgecolor= 'lightgrey', facecolor = 'none')


Capacity_0.plot(ax=ax,
                  color= 'grey', #my_cmap(network_flow.net_flow.values), 
                  linewidth = 2,
                  label = 'Capacity utilization up to 25%')
Capacity_25.plot(ax=ax, 
                  color= 'cornflowerblue', #my_cmap(network_flow.net_flow.values), 
                  linewidth = 2,
                  label = 'Capacity utilization 25%-50%')
Capacity_50.plot(ax=ax, 
                  color= 'blue', #my_cmap(network_flow.net_flow.values), 
                  linewidth = 2.1,
                  label = 'Capacity utilization 50%-75%')
Capacity_75.plot(ax=ax, 
                  color= 'purple', #my_cmap(network_flow.net_flow.values), 
                  linewidth = 2,
                  label = 'Capacity utilization 75%-100%')
Manipulated.plot(ax=ax, 
                  color= 'red', #my_cmap(network_flow.net_flow.values), 
                  linewidth = 2,
                  label = 'Exceeded Capacity')


leg = plt.legend(loc = 'lower left', ncol = 2, fontsize = 16, bbox_to_anchor = (-0,-0.07), markerscale = 2)
for legobj in leg.legend_handles:
    legobj.set_linewidth(5.0)
plt.axis('off')




