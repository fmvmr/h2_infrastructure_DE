# Import necessary libraries

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

from shapely.geometry import Point
from collections import defaultdict

from matplotlib import cm
from math import sqrt
from pulp import *
import re


# Variables and input path and files
Date = str(date.today())

save_geojson= False
save_csv = False

path = ".\\"
Scenario_file = path + 'Input\\00_Scenario-Base-2023-08-02.csv'

to_file_geojson =  path +'Results\\GeoJSON\\Scenario-Base-'+Date+'.geojson'
to_file_csv =  path + 'Results\\CSV\\Scenario-EnInd-DH-'+Date+'.csv'



'''Read Topology and Nodes of natural gas network'''
def read_topo(file_name):
    file = pd.read_csv(file_name,sep = ';')
    file['geometry'] = file['geometry'].apply(wkt.loads)
    file_gpd = gpd.GeoDataFrame(file, crs='EPSG:3857')
    return (file_gpd)


# Network
network = read_topo(path + "Input\\IGGIELGN_Network0.csv")
network['source'] = network['source'].apply(lambda x: Point(eval(x)))
network['target'] = network['target'].apply(lambda x: Point(eval(x)))
network.drop(columns = ['Country', 'tags', 'index', 'id','name', 'method', 'uncertainty'], inplace = True)

# Nodes
nodes = pd.read_csv(path + "Input\\IGGIELGN_Network0_nodesgdf.csv", index_col=0)
nodes['geometry'] = nodes['geometry'].apply(wkt.loads)
nodes_gdf = gpd.GeoDataFrame(nodes, crs='EPSG:3857')

#Map the corresponding ID to edge dataframe
network = pd.merge(network, nodes_gdf, how='left', left_on='source', right_on='geometry', suffixes=('', '_1'))
network.rename(columns = {'ID':'Source_ID'},inplace=True)
network = pd.merge(network, nodes_gdf, how='left', left_on='target', right_on='geometry', suffixes=('', '_2'))
network.rename(columns = {'ID':'Target_ID'},inplace=True)
network.drop(columns = ['geometry_1', 'geometry_2'], inplace = True)
network = network.to_crs(crs=4326)

'''Read Peak Load Data from Scenario File '''

# Read input file with peak load per node
aggregated = pd.read_csv(Scenario_file, index_col=0)
aggregated['geometry'] = aggregated['geometry'].apply(wkt.loads)
aggregated = gpd.GeoDataFrame(aggregated, crs='EPSG:4326')



'''Capacity Calculation in Peak Hour'''
# Physical Properties of Hydrogen and Natural Gas to calculate capacity and diameter 

pi = 3.14 

LHV_NG = 13.9/1000 # MWh/kg https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html
LHV_NG_m3 = 35.8 # MJ/m^3 https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html
v_NG = 25 #m/s https://onlinelibrary.wiley.com/doi/full/10.1002/ese3.932
rho_NG = 52.548 #kg/m^3 https://www.unitrove.com/engineering/tools/gas/natural-gas-density7

LHV_H2 = 33.3/1000 # MWh/kg
v_H2 = 15 # m/s
rho_H2 = 5.246 #kg/m^3 https://cmb.tech/hydrogen-tools

# Calculate Capacity in MWh/h
network['capacity_MWh_h']=round(network['capacity_Mm^3/d']*LHV_NG_m3*1000000/3600/24*0.85).astype(int) 


''' Parallel infrastructure Calculation '''

# Find parallel pipelines in the dataset
network['key'] = network.apply(lambda row: tuple(sorted((row['Source_ID'], row['Target_ID']))), axis=1)

network1 = network.groupby('key')['capacity_MWh_h'].sum().reset_index()
network1.rename(columns = {'capacity_MWh_h' : 'combined_capacity_MWh_h'}, inplace = True)

network2 = network.groupby('key')['diameter_mm'].sum().reset_index()
network2.rename(columns = {'diameter_mm' : 'combined_diameter'}, inplace = True)

network3 = network.merge(network1, how='left', on='key')
network4 = network3.merge(network2, how='left', on='key')


# network without duplicates and combined capacity
network_noduplicates = network4.sort_values('capacity_MWh_h', ascending=False).drop_duplicates('key').sort_index()
network_noduplicates['Parallel']  = np.where((network_noduplicates['diameter_mm'] != network_noduplicates['combined_diameter']), 0, 1)

# network of only the duplicates
pipeline_duplicates = network4.sort_values('capacity_MWh_h', ascending=False).sort_index()
pipeline_duplicates['Parallel']  = np.where((pipeline_duplicates['diameter_mm'] != pipeline_duplicates['combined_diameter']), 0, 1)
pipeline_duplicates = pipeline_duplicates[pipeline_duplicates.diameter_mm!=pipeline_duplicates.combined_diameter]

# Function to calculate the cost of retrofitting of parallel pipeline by considering the average over all strings
def parallel_pipeline_cost(key):
    cum = 0
    lengths = []
    diams = []
    costs = []
    for cap, diam, length in zip(pipeline_duplicates.query("key == @key").capacity_MWh_h, pipeline_duplicates.query("key == @key").diameter_mm, pipeline_duplicates.query("key == @key").length_m):
        cum += cap
        diams.append(diam)
        lengths.append(length)
        costs.append((length*(1.67*10**(-4)*(diam**2)-2*10**(-13)*diam-7.8*10**(-10))\
                                 +(length*(1.1*10**(-4)*(diam**2)-1.6*10**(-2)*diam+2))\
                                 +(length*(1*10**(-4)*(diam**2)-1.5*10**(-12)*diam-2.9*10**(-10)))))
        return sum(costs) / len(costs)


# use function to calculate average cost of retrotofitting over all strings as a weight
parallel_pipe_cost_dict = {idx: [key, parallel_pipeline_cost(key)] for key, idx in zip(pipeline_duplicates.key, pipeline_duplicates.index)}
parallel_pipe_cost = pd.DataFrame.from_dict(parallel_pipe_cost_dict,orient='index', columns = ['key', 'cost'])


'''Weight Calculations - Cost'''
# Calculate the cost of retroffiting by Cerniauskas 2020 for all singular strings
network_noduplicates['Cost']= network_noduplicates.length_m*(1.67*10**(-4)*(network_noduplicates.combined_diameter**2)-2*10**(-13)*network_noduplicates.combined_diameter-7.8*10**(-10))\
                             +(network_noduplicates.length_m*(1.1*10**(-4)*(network_noduplicates.combined_diameter**2)-1.6*10**(-2)*network_noduplicates.combined_diameter+2))\
                             +(network_noduplicates.length_m*(1*10**(-4)*(network_noduplicates.combined_diameter**2)-1.5*10**(-12)*network_noduplicates.combined_diameter-2.9*10**(-10)))

# assign average cost for retrofitting multiple strings
for key, cost in zip(parallel_pipe_cost.key, parallel_pipe_cost.cost):
    network_noduplicates.loc[network_noduplicates['key'] == key, 'Cost'] =  cost


# Delete Interconnectoren, that are too far and not relevant to the calculation for better pictur
network_noduplicates = network_noduplicates[(network_noduplicates.country_code != "['XX', 'DE']") & (network_noduplicates.country_code != "['NO', 'DE']") & (network_noduplicates.country_code != "['DE', 'FI']") & (network_noduplicates.country_code != "['PL', 'DE']")]

'''Adaption in capacity per edge'''
# Adaptation of import capacities in case of Base Case in Bremen and Stade

network_noduplicates.at[545,'combined_capacity_MWh_h'] = 2000
network_noduplicates.at[544,'combined_capacity_MWh_h'] = 1720

# Read Europe and Germany Shapefile
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

# Needs to be manually adjusted to work in specific scenarios
aggregated.at[high_demand[0],'demand'] = -102
points_within_Berlin.drop([high_demand[0]], inplace = True)

for i in points_within_Berlin.index: 
    aggregated.demand[i] = demand_at_nodes


''' Bring input data into right format for algorithm'''
aggregated.demand = aggregated.demand.astype(int)
aggregated.demand = aggregated.demand*(-1)


'''Create Graph with bi directional edges'''
Graph = pd.DataFrame()
Graph['source']=network_noduplicates['Source_ID'].astype(str)
Graph['target']=network_noduplicates['Target_ID'].astype(str)
Graph['combined_capacity']= round(network_noduplicates.combined_capacity_MWh_h.astype(int))
Graph['weight'] = round(network_noduplicates.Cost/1000).astype(int)
#Graph with prioritzizing parallel infrasturcture through penalty: Graph['weight'] = round(network_noduplicates.Cost/1000).astype(int)*network_noduplicates.Parallel.astype(int)

Graph1 = pd.DataFrame()
Graph1['source']=network_noduplicates['Target_ID'].astype(str)
Graph1['target']=network_noduplicates['Source_ID'].astype(str)
Graph1['combined_capacity']= round(network_noduplicates.combined_capacity_MWh_h.astype(int))
Graph1['weight'] = round(network_noduplicates.Cost/1000).astype(int)
#Graph with prioritzizing parallel infrasturcture through penalty: Graph1['weight'] = round(network_noduplicates.Cost/1000).astype(int)*network_noduplicates.Parallel.astype(int)

''' Create all inputs for optimization'''
Graph_double = pd.concat([Graph, Graph1])

Supply = pd.Series(aggregated.demand.values,index=aggregated.ID.astype(str)).to_dict()

Nodes = list(Supply.keys())

# Edges
Source = [str(i) for i in Graph_double['source']]
Target = [str(i) for i in Graph_double['target']]
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
flow_var = LpVariable.dicts("Flow_Var",E, 0,None, LpInteger)  
flow_used = LpVariable.dicts("Flow_Used", E, 0, 1,  LpInteger)  
a_very_large_number = 100000000000000000000000000000

# Objective function
model1 += lpSum([(flow_used[(i,j)], 1)*Cost[i][j] for (i,j) in E])-lpSum([Cost[i][j] for (i,j) in E])


# Flow Constraints
for i in Nodes: 
    model1 += (lpSum([flow_var[i,j] for j in Nodes if (i,j) in E]) - lpSum([flow_var[k,i] for k in Nodes if (k,i) in E])) == Supply[i] 

# Capacity Constraint
for (i,j) in E: 
    model1 += flow_var[i,j] <= Capacity[i][j] 

# Binary Constraint
for (i, j) in E:
    model1 += a_very_large_number * flow_used[(i,j)]>= flow_var[(i,j)]#,"Route used"

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
for c in range(int(len(Edges)/2),len(Edges)): #int(len(Edges)/2)
    Source.append(Edges[c][0])
    Target.append(Edges[c][1])
    Flow1.append(Flow[c])

# Save Results
Flow_df = pd.DataFrame(list(zip(Source, Target, Flow1)), columns = ['source', 'target', 'flow'])
Flow_df[Flow_df.flow!=0]

# Filter the original dataframe with the results from the optimitaion and append the results 
network_noduplicates['source_flow'] = [Flow_df[(Flow_df['source']==source) & (Flow_df['target']==target)]['flow'].values[0] for source,target in network_noduplicates[['Source_ID','Target_ID']].astype(str).values]
network_noduplicates['target_flow'] = [-Flow_df[(Flow_df['target']==source) & (Flow_df['source']==target)]['flow'].values[0] for source,target in network_noduplicates[['Source_ID','Target_ID']].astype(str).values]
network_noduplicates['net_flow'] = abs(network_noduplicates.source_flow + network_noduplicates.target_flow)

# Save the results handling in one dataframe 
network_flow = network_noduplicates[(network_noduplicates.source_flow > 0) | (network_noduplicates.target_flow < 0)]

''' Calculate the actual cost of retrotitting the parallel pipeline depending on how many strings are needed to transport the volumes'''

# Save parallel pipelines in network
parallel_in_network = network_flow[network_flow.Parallel == 0]
# Save capacity of the parallel pipelines
capacity_in_network = parallel_in_network[['key', 'net_flow']]

# merge with network where all edges are included
parallel_in_network1 = network.merge(capacity_in_network, how='right', on='key')
parallel_in_network2 = parallel_in_network1[~parallel_in_network1.isnull().any(axis=1)]
parallel_in_network3 = parallel_in_network2.sort_values(by=['key', 'capacity_MWh_h'])

# Function for cost calculation of parallel pipeline
def parallel_pipeline_capacity(key):
    cum = 0
    lengths = []
    diams = []
    costs = []
    capex = []
    opex = []
    
    for cap, diam, length, netflow in zip(parallel_in_network3.query("key == @key").capacity_MWh_h, parallel_in_network3.query("key == @key").diameter_mm, parallel_in_network3.query("key == @key").length_m, parallel_in_network3.query("key == @key").net_flow):
        cum += cap
        diams.append(diam)
        lengths.append(length)
        costs.append((round(length*(1.67*10**(-4)*(diam**2)-2*10**(-13)*diam-7.8*10**(-10))\
                                 +(length*(1.1*10**(-4)*(diam**2)-1.6*10**(-2)*diam+2))\
                                 +(length*(1*10**(-4)*(diam**2)-1.5*10**(-12)*diam-2.9*10**(-10))))))
        capex.append((round(length*(1.67*10**(-4)*(diam**2)-2*10**(-13)*diam-7.8*10**(-10)))))
        opex.append((round((length*(1.1*10**(-4)*(diam**2)-1.6*10**(-2)*diam+2))\
                                 +(length*(1*10**(-4)*(diam**2)-1.5*10**(-12)*diam-2.9*10**(-10))))))
        if cum < netflow:
            continue
        else:
                return sum(costs), sum(capex), sum(opex)

# Apply function to each parallel pipeline in the result networks
parallel_pipe_cost_dict = {idx: [key, parallel_pipeline_capacity(key)[0], parallel_pipeline_capacity(key)[1], parallel_pipeline_capacity(key)[2]] for key, idx in zip(parallel_in_network3.key, parallel_in_network3.index)}
parallel_pipe_cost = pd.DataFrame.from_dict(parallel_pipe_cost_dict,orient='index', columns = ['key', 'cost', 'capex', 'opex'])

'''Function to calculate Cost '''
def capex_cost(x):

    capex = x.length_m*(1.67*10**(-4)*(x.diameter_mm**2)-2*10**(-13)*x.diameter_mm-7.8*10**(-10))

    return capex


def opex_cost(x):
    
    
    opex = (x.length_m*(1.1*10**(-4)*(x.combined_diameter**2)-1.6*10**(-13)*x.combined_diameter+2))\
                +(x.length_m*(1*10**(-4)*(x.combined_diameter**2)-1.5*10**(-12)*x.combined_diameter-2.9*10**(-10)))
    return  opex

def capex_opex(x):
    cost = x.length_m*(1.67*10**(-4)*(x.combined_diameter**2)-2*10**(-13)*x.combined_diameter-7.8*10**(-10))\
                             +(x.length_m*(1.1*10**(-4)*(x.combined_diameter**2)-1.6*10**(-13)*x.combined_diameter+2))\
                             +(x.length_m*(1*10**(-4)*(x.combined_diameter**2)-1.5*10**(-12)*x.combined_diameter-2.9*10**(-10)))
    
    return cost

''' Calculate Cost by applying functions for singular strings and multiple strings'''

# Singular strings
network_flow['CAPEX'] = network_flow.apply(capex_cost, axis=1)
network_flow['OPEX'] = network_flow.apply(opex_cost, axis=1)
network_flow['Cost'] = network_flow.apply(capex_opex, axis=1)

# Parallel strings
for key, cost, capex, opex in zip(parallel_pipe_cost.key, parallel_pipe_cost.cost, parallel_pipe_cost.capex, parallel_pipe_cost.opex):
    network_flow.loc[network_flow['key'] == key, 'Cost'] =  cost
    network_flow.loc[network_flow['key'] == key, 'CAPEX'] =  capex
    network_flow.loc[network_flow['key'] == key, 'OPEX'] =  opex
    
# Overall Cost

network_flow.CAPEX.sum()
network_flow.OPEX.sum()
network_flow.Cost.sum()/1000

'''Capacity Utilizatiion Calculation'''
network_flow['capacity_utilization'] = ((network_flow['net_flow']/network_flow['combined_capacity_MWh_h'])*100)


'''Length of results network and original network '''

network_flow.length_m.astype(int).sum()/1000

network.length_m.astype(int).sum()/1000

''' Change Data Type for Visualisation'''

network_flow.net_flow = network_flow.net_flow.astype(int) 
max_flow = network_flow[network_flow. net_flow == max(network_flow. net_flow)]


''' Save Results'''
if save_geojson: 
    network_arcgis = network_flow[['geometry', 'capacity_MWh_h', 'diameter_mm', 'length_m', 'net_flow', 'Cost', 'OPEX', 'CAPEX', 'capacity_utilization']]
    network_arcgis.to_file(to_file_geojson, driver='GeoJSON') 
if save_csv: 
    network_flow.to_csv(to_file_csv)


''' Visualisations '''



network_flow = network_flow.to_crs(crs= 4326)
network_noduplicates = network_noduplicates.to_crs(crs=4326)


'''Colors for Visualisation of Flow'''


min_val, max_val = 0.3,0.9
n = 500
orig_cmap = cm.Blues
colors = orig_cmap(np.linspace(min_val, max_val, n))
my_cmap = cm.colors.LinearSegmentedColormap.from_list("mycmap", colors)
# gradient = np.linspace(0, 1 ,int(max(network_flow.net_flow)) )

''' Input Data Visualisation'''
nodes_supply= aggregated[aggregated['Supply']>0]
nodes_supply.set_crs('EPSG:4326')
nodes_demand= aggregated[aggregated['Demand']>0]
nodes_demand.set_crs('EPSG:4326')
demand_nodes = aggregated[aggregated['demand']!=0]


''' Visualisation of network flow by color'''

fig,(ax) = plt.subplots(1,1 , figsize =(15,15))
DE.plot(ax=ax, edgecolor= 'lightgrey', facecolor= 'none')

network_flow.plot(ax=ax, 
                  color= my_cmap(network_flow.net_flow.values), 
                  linewidth = 2,
                  label = 'Hydrogen Network 2045')

demand_nodes.plot(ax=ax, 
                color= 'green', 
                markersize = 10,
                label = 'Supply & Demand Nodes')

sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=int(min(network_flow.net_flow)), vmax=int(max(network_flow.net_flow))))

plt.colorbar(sm, ax=ax, label ='Energy Flow [MWh/h]',fraction=0.04, pad=0.03)
plt.legend(loc = 'lower left')
plt.axis('off')
plt.show()




''' Visulaisation of Flow by thinkness'''

fig,(ax) = plt.subplots(1,1 , figsize =(15,15))
DE.plot(ax=ax, edgecolor= 'lightgrey', facecolor ='none')

max_flow.plot(ax=ax, 
                  color = 'blue', 
                  linewidth = 2, 
                  label = 'Hydrogen Flow [MWh/h]')

network_flow.plot(ax=ax, 
                  color = 'blue', 
                  linewidth =  network_flow.net_flow.values/3000)

nodes_supply.plot(ax=ax, 
                color= 'green', 
                markersize = nodes_supply.Supply.values/50,
                label = 'Supply')

nodes_demand.plot(ax=ax, 
                color= 'red', 
                markersize = nodes_demand.Demand.values/50,
                label = 'Demand')

plt.legend(loc = 'lower left')
plt.axis('off')
plt.show()


''' Capacity Utilization Visualisation'''

# Save classifications in specific dataframes
Manipulated = network_flow[(network_flow.key == (143, 622)) | (network_flow.key == (143, 187))]
Capacity_75 = network_flow[network_flow.capacity_utilization>75]
Capacity_50 = network_flow[network_flow.capacity_utilization>50]
Capacity_25 = network_flow[network_flow.capacity_utilization>25]
Capacity_0 = network_flow[network_flow.capacity_utilization>0]


fig,(ax) = plt.subplots(1,1 , figsize =(15,15))
DE.plot(ax=ax, edgecolor= 'lightgrey', facecolor = 'none')


Capacity_0.plot(ax=ax,
                  color= 'gray', #my_cmap(network_flow.net_flow.values), 
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

plt.legend(loc = 'lower left', ncol = 3)
plt.axis('off')
plt.show()


'''OUTLOOK: Visualisation of Trucks Usage'''


Trucks = 420 #kg
MWh_per_Trucks = Trucks*LHV_H2

Trucks = network_flow[network_flow.net_flow <=MWh_per_Trucks]
Pipeline = network_flow[network_flow.net_flow >MWh_per_Trucks]


fig,(ax) = plt.subplots(1,1 , figsize =(10,10))
DE.plot(ax=ax, edgecolor= 'lightgrey', facecolor = 'none')

Pipeline.plot(ax=ax, 
                  color = 'blue', 
                  linewidth = 2, 
                  label = 'Hydrogen Network 2045')

Trucks.plot(ax=ax, 
                 color= 'orange', 
                 linewidth = 2, 
                 label = 'Trucks')


plt.legend(loc = 'lower left')
plt.axis('off')
plt.show()

