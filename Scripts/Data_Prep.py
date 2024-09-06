# Import necessary libraries
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
from shapely import wkt
import geopy.distance
from shapely.geometry import Point
from tqdm import tqdm
from datetime import date
import matplotlib.pyplot as plt


# Define the scenario demand settings
# File name of the scenario input file and output file of aggregated
Date = str(date.today())

#Path to the MA Input folder
path = "C:\\Users\\mym852\\OneDrive - AFRY\\Documents\\MA_Files\\"
file_name = path +"Scenarios\\00_Scenario_BaseCase_Final.xlsx"
to_file = path +'Scenarios\\Inputs\\00_Scenario-Base'+Date+'.csv'

Save_Output = False
Visualisation = False

#Binary Scenario Settinfs
#Industry_overall
Industry = False

#Demand sectors
Refineries = False
Chemicals = True
Steel = True 
Paper = True
Mineral_Processing = True 
Metal_Processing = True
Non_Metallic_Minerals = True

Other_Industry = True

Residential_Heat = True
District_Heat = True

Passenger = True  
Public = True

Air = True
Train = True
Trucks = True

# Functions

def read_topo(file_name):
    file = pd.read_csv(file_name,sep = ';')
    file['geometry'] = file['geometry'].apply(wkt.loads)
    file_gpd = gpd.GeoDataFrame(file, crs='EPSG:3857')
    return (file_gpd)

def read_data(file_name, sheet_name):
    file = pd.read_excel(file_name, sheet_name=sheet_name)
    file_gpd = gpd.GeoDataFrame(file, geometry=gpd.points_from_xy(file.Longitude, file.Latitude),
                                crs='EPSG:4326').to_crs('EPSG:3857')
    return (file_gpd)


def AggregatedDemand(df1, df2):
    for index, row1 in tqdm(df1.iterrows(), total=df1.shape[0]):
        for _, row2 in df2.iterrows():
            if row1['ID'] == row2['Closest_node']:
                df1.loc[index, 'Demand'] += row2['Peak Load [MWh/h]']
    return df1

def AggregatedSupply(df1, df2):
    for index, row1 in tqdm(df1.iterrows(), total=df1.shape[0]):
        for _, row2 in df2.iterrows():
            if row1['ID'] == row2['Closest_node']:
                df1.loc[index, 'Supply'] += row2['Peak Load [MWh/h]']
    return df1

def AddStorage(df1, df2, aggregated):
    if sum(df2['Peak Load [MWh/h]']) <= 0:
        df2['Peak Load [MWh/h]'] = df2['Peak Load [MWh/h]'] * (-1)
        aggregated = AggregatedSupply(df1, df2)
    else:
        aggregated = AggregatedDemand(df1, df2)

    return aggregated


def find_closest_location(df1, df2):
    closest_node = []
    closest_distance = []

    for _, row1 in tqdm(df1.iterrows(), total=df1.shape[0]):
        max_distance = float('inf')

        for _, row2 in df2.iterrows():
            coords_1 = (row1['Latitude'], row1['Longitude'])
            coords_2 = (row2['Latitude'], row2['Longitude'])
            distance = geopy.distance.distance(coords_1, coords_2).km
            if distance < max_distance:
                node = row2['ID']
                max_distance = distance

        closest_node.append(node)
        closest_distance.append(max_distance)
    return closest_node, closest_distance

def get_NUT3_centroid(df1, df2):
    shape= pd.DataFrame()
    shape['NUTS3'] = [df2.NUTS_ID.values[i] for i in range(len(df2.index))]
    shape['geometry']= [df2.geometry.values[i] for i in range(len(df2.index))]

    merged_df = pd.merge(df1, shape[['NUTS3', 'geometry']], on = 'NUTS3', how='left')
    merged_gdf = gpd.GeoDataFrame(merged_df, geometry= 'geometry',crs = 'EPSG:4326')
    merged_gdf.to_crs('EPSG:3857', inplace = True)
    merged_gdf.dropna(inplace=True)
    merged_gdf['geometry']= [merged_gdf.geometry.centroid.values[i] for i in range(len(merged_gdf.index))]
    merged_gdf.to_crs('EPSG:4326', inplace = True)

    merged_gdf['Longitude']=merged_gdf.geometry.x
    merged_gdf['Latitude']=merged_gdf.geometry.y
    return(merged_gdf)


# Nodes Dataframe with cosistent nodes -> when pathways are calculated use nodes from previous time step
nodes = pd.read_csv(path + "Input\\IGGIELGN_Network0_nodesgdf.csv", index_col=0)

nodes['geometry'] = nodes['geometry'].apply(wkt.loads)
nodes_gdf = gpd.GeoDataFrame(nodes, crs='EPSG:3857')
nodes_gdf.to_crs('EPSG:4326', inplace = True)
nodes_gdf['Longitude'] = nodes_gdf.geometry.apply(lambda p: p.x)
nodes_gdf['Latitude'] = nodes_gdf.geometry.apply(lambda p: p.y)
nodes_gdf['Supply'] = 0
nodes_gdf['Demand'] = 0 


# Create a new Dataframe where the input data with peak laod at each node can be saved
aggregated = pd.DataFrame()

# Read Data from each Excel sheet in this section with exact locations
sheet_name = "LH2_Input"
lh2 = read_data(file_name, sheet_name)
lh2.to_crs(4326, inplace = True)
lh2['Closest_node'], lh2['distance'] = find_closest_location(lh2, nodes_gdf)
aggregated =AggregatedSupply(nodes_gdf, lh2)

sheet_name = "IC_Input"
ic = read_data(file_name, sheet_name)
ic.to_crs(4326, inplace = True)
ic['Closest_node'], ic['distance'] = find_closest_location(ic, nodes_gdf)
aggregated = AggregatedSupply(nodes_gdf, ic)

sheet_name = "Ind_Elec_Input"
elec_ind_projects = read_data(file_name, sheet_name)
elec_ind_projects= elec_ind_projects[elec_ind_projects['Peak Load [MWh/h]'] != 0]
elec_ind_projects.reset_index()
elec_ind_projects['Closest_node'], elec_ind_projects['distance'] = find_closest_location(elec_ind_projects, nodes_gdf)
elec_ind_projects.to_crs('EPSG:4326', inplace = True)
aggregated = AggregatedSupply(nodes_gdf, elec_ind_projects)

if Industry:
    sheet_name = "Ind_Input"
    ind = read_data(file_name, sheet_name)
    ind= ind[ind['Peak Load [MWh/h]'] != 0]
    ind.reset_index()
    elec_ind_projects
    ind['Closest_node'], ind['distance'] = find_closest_location(ind, nodes_gdf)
    ind.to_crs('EPSG:4326', inplace = True)


if Refineries: 
    sheet_name = "Refineries_Input"
    refineries = read_data(file_name, sheet_name)
    refineries['Closest_node'], refineries['distance'] = find_closest_location(refineries, nodes_gdf)
    refineries.to_crs('EPSG:4326', inplace = True)
    aggregated = AggregatedDemand(nodes_gdf, refineries)

if Chemicals: 
    sheet_name = "Chemicals_Input"
    chemicals = read_data(file_name, sheet_name)
    chemicals['Closest_node'], chemicals['distance'] = find_closest_location(chemicals, nodes_gdf)
    chemicals.to_crs('EPSG:4326', inplace = True)
    aggregated = AggregatedDemand(nodes_gdf, chemicals)

if Steel: 
    sheet_name = "Steel_Input"
    steel = read_data(file_name, sheet_name)
    steel['Closest_node'], steel['distance'] = find_closest_location(steel, nodes_gdf)
    steel.to_crs('EPSG:4326', inplace = True)
    aggregated = AggregatedDemand(nodes_gdf, steel)


if Mineral_Processing:
    sheet_name = "Mineral_Processing_Input"
    mineral_processing = read_data(file_name, sheet_name)
    mineral_processing['Closest_node'], mineral_processing['distance'] = find_closest_location(mineral_processing, nodes_gdf)
    mineral_processing.to_crs('EPSG:4326', inplace = True)
    aggregated = AggregatedDemand(nodes_gdf, mineral_processing)


if Paper:
    sheet_name = "Paper_Input"
    paper = read_data(file_name, sheet_name)
    paper['Closest_node'], paper['distance'] = find_closest_location(paper, nodes_gdf)
    paper.to_crs('EPSG:4326', inplace = True)
    aggregated = AggregatedDemand(nodes_gdf, paper)


if Metal_Processing: 
    sheet_name = "Metal_Processing_Input"
    metal_processing = read_data(file_name, sheet_name)
    metal_processing['Closest_node'], metal_processing['distance'] = find_closest_location(metal_processing, nodes_gdf)
    metal_processing.to_crs('EPSG:4326', inplace = True)
    aggregated = AggregatedDemand(nodes_gdf, metal_processing)


if Non_Metallic_Minerals: 
    sheet_name = "Non_Metallic_Minerals_Input"
    non_metallic_minerals = read_data(file_name, sheet_name)
    non_metallic_minerals['Closest_node'], non_metallic_minerals['distance'] = find_closest_location(non_metallic_minerals, nodes_gdf)
    non_metallic_minerals.to_crs('EPSG:4326', inplace = True)
    aggregated = AggregatedDemand(nodes_gdf, non_metallic_minerals)


if Air:
    sheet_name = "Air_Input"
    air = read_data(file_name, sheet_name)
    air['Closest_node'], air['distance'] = find_closest_location(air, nodes_gdf)
    air.to_crs('EPSG:4326', inplace = True)
    aggregated = AggregatedDemand(nodes_gdf, air)


if Trucks:
    sheet_name = "Trucks_Input"
    trucks = read_data(file_name, sheet_name)
    trucks['Closest_node'], trucks['distance'] = find_closest_location(trucks, nodes_gdf)
    trucks.to_crs('EPSG:4326', inplace = True)
    aggregated = AggregatedDemand(nodes_gdf, trucks)


sheet_name = "Storage_Input"
storage = read_data(file_name, sheet_name)
storage['Closest_node'], storage['distance'] = find_closest_location(storage, nodes_gdf)
storage.to_crs('EPSG:4326', inplace = True)
aggregated = AddStorage(nodes_gdf, storage, aggregated)


# Read Europe data set for Nuts3 classifications if only NUTS3 level available
Europe = gpd.read_file(path + 'Input\\NUTS_RG_60M_2021_4326.shp')
Europe = Europe.loc[(Europe['LEVL_CODE'] == 3)]
Europe.drop(list(range(778,787)), inplace=True) # Spanische Kolonien Nuts 3
Europe.drop(list(range(905,910)), inplace=True) # Französische Kolonien Nuts 3 
Europe.drop([1138,1139], inplace=True) #Portugiesische Kolonien Nuts 3
Europe.drop([1950, 1951], inplace=True) #Svalbard
Europe.drop([537, 960], inplace=True) #Island
DE = Europe.loc[(Europe['CNTR_CODE'] == 'DE')]

# Map postal code to nuts ID
plz_file = path + "Input\\plz_nuts3.csv"
plz = pd.read_csv(plz_file, delimiter = ';')
plz['NUTS3']=[plz['NUTS3'][i].strip().replace("'","") for i in plz.index]
plz['Standort-PLZ']=[plz['CODE'][i].strip().replace("'","") for i in plz.index]
plz.drop(columns = ['CODE'], inplace = True)
plz['Standort-PLZ']= plz['Standort-PLZ'].astype(int)

# Read data where NUT§ region or Postal code available
sheet_name = "Elec_Input"
File_elec = pd.read_excel(file_name, sheet_name = sheet_name)
File_elec= File_elec[File_elec['Peak Load [MWh/h]'] != 0]
File_elec.reset_index()
elec_projects = get_NUT3_centroid(File_elec, DE)
elec_projects['Closest_node'], elec_projects['distance'] = find_closest_location(elec_projects, nodes_gdf)
elec_projects.set_crs('EPSG:4326', inplace = True)
aggregated = AggregatedSupply(nodes_gdf, elec_projects)


if Other_Industry:
    sheet_name = "Other_Industry_Input"
    other_industry = pd.read_excel(file_name, sheet_name = sheet_name)
    other_industry= other_industry[other_industry['Peak Load [MWh/h]'] != 0]
    other_industry.reset_index()
    other_industry = get_NUT3_centroid(other_industry, DE)
    other_industry['Closest_node'], other_industry['distance'] = find_closest_location(other_industry, nodes_gdf)
    other_industry.set_crs('EPSG:4326', inplace = True)
    aggregated = AggregatedDemand(nodes_gdf, other_industry)

if District_Heat:
    sheet_name = "District_Input"
    district_file = pd.read_excel(file_name, sheet_name)
    district_file = district_file.rename(columns={'Postcode': 'Standort-PLZ'})
    district_df = pd.merge(district_file, plz, on = 'Standort-PLZ', how='left')
    district_df.drop(columns=['Standort-PLZ'], inplace = True)
    #adjust postal codes manually, which are not available in data set
    district_df.loc[25,'NUTS3'] = 'DE300'
    district_df.loc[90,'NUTS3'] = 'DEB24'
    district_df.loc[93,'NUTS3'] = 'DEA34'
    district_df.loc[94,'NUTS3'] = 'DEA34'
    district_df.loc[95,'NUTS3'] = 'DEA34'
    district_df.loc[112,'NUTS3'] = 'DE269'
    district_df.loc[126,'NUTS3'] = 'DE112'
    district_gdf = get_NUT3_centroid(district_df, DE)
    district_gdf['Closest_node'], district_gdf['distance'] = find_closest_location(district_gdf, nodes_gdf)
    aggregated = AggregatedDemand(nodes_gdf, district_gdf)


if Residential_Heat:
    sheet_name = "Residential_Input"
    residential = pd.read_excel(file_name, sheet_name = sheet_name)
    residential= residential[residential['Peak Load [MWh/h]'] != 0]
    residential.reset_index()
    residential = get_NUT3_centroid(residential, DE)
    residential['Closest_node'], residential['distance'] = find_closest_location(residential, nodes_gdf)
    residential.set_crs('EPSG:4326', inplace = True)
    aggregated = AggregatedDemand(nodes_gdf, residential)


if Passenger:
    sheet_name = "Passenger_Input"
    passenger = pd.read_excel(file_name, sheet_name = sheet_name)
    passenger= passenger[passenger['Peak Load [MWh/h]'] != 0]
    passenger.reset_index()
    passenger = get_NUT3_centroid(passenger, DE)
    passenger['Closest_node'], passenger['distance'] = find_closest_location(passenger, nodes_gdf)
    passenger.set_crs('EPSG:4326', inplace = True)
    aggregated = AggregatedDemand(nodes_gdf, passenger)



if Public: 
    sheet_name = "Public_Input"
    public = pd.read_excel(file_name, sheet_name = sheet_name)
    public= public[public['Peak Load [MWh/h]'] != 0]
    public.reset_index()
    public = get_NUT3_centroid(public, DE)
    public['Closest_node'], public['distance'] = find_closest_location(public, nodes_gdf)
    public.set_crs('EPSG:4326', inplace = True)
    aggregated = AggregatedDemand(nodes_gdf, public)



if Train: 
    sheet_name = "Train_Input"
    train_file = pd.read_excel(file_name, sheet_name)
    train_df = pd.merge(train_file, plz, on = 'Standort-PLZ', how='left')
    train_df.loc[23,'NUTS3'] = 'DE501'
    train_df.loc[146,'NUTS3'] = 'DE40B'
    train_df.loc[150,'NUTS3'] = 'DE71E'
    train_df.loc[171,'NUTS3'] = 'DEE08'
    train_df.drop(columns=['Standort-PLZ'], inplace = True)
    train_gdf = get_NUT3_centroid(train_df, DE)
    train_gdf['Closest_node'], train_gdf['distance'] = find_closest_location(train_gdf, nodes_gdf)
    aggregated = AggregatedDemand(nodes_gdf, train_gdf)


#create a column where demand and supply are balanced at each node
aggregated['demand']=aggregated['Supply']-aggregated['Demand']

if Save_Output: 
    aggregated.to_csv(to_file)

if Visualisation: 
    
    ''' INDUSTRY VISUALISATION'''
    # Create a point dataframe to ensure the same legend size for each category
    max_chemicals = chemicals[chemicals['Peak Load [MWh/h]']== max(chemicals['Peak Load [MWh/h]'])]
    max_steel = steel[steel['Peak Load [MWh/h]']== max(steel['Peak Load [MWh/h]'])]
    max_paper = paper[paper['Peak Load [MWh/h]']== max(paper['Peak Load [MWh/h]'])]
    max_mineral_processing = mineral_processing[mineral_processing['Peak Load [MWh/h]']== max(mineral_processing['Peak Load [MWh/h]'])]
    max_metal_processing = metal_processing[metal_processing['Peak Load [MWh/h]']== max(metal_processing['Peak Load [MWh/h]'])]
    max_non_metallic_minerals = chemicals[chemicals['Peak Load [MWh/h]']== max(chemicals['Peak Load [MWh/h]'])]
    max_other_industry = other_industry[other_industry['Peak Load [MWh/h]']== max(other_industry['Peak Load [MWh/h]'])]
    
    # Plotting of Industry
    fig,ax = plt.subplots(figsize =(15,15))
    
    DE.plot(ax=ax, edgecolor= 'grey', facecolor= 'none')
    
    chemicals.plot(ax=ax, color='tan', markersize = chemicals['Peak Load [MWh/h]']/10 )
    steel.plot(ax=ax, color='royalblue', markersize = steel['Peak Load [MWh/h]']/10 )
    paper.plot(ax=ax, color='navy', markersize = paper['Peak Load [MWh/h]']/10 )
    mineral_processing.plot(ax=ax, color='green', markersize = mineral_processing['Peak Load [MWh/h]']/10 )
    metal_processing.plot(ax=ax, color='olive', markersize = metal_processing['Peak Load [MWh/h]']/10 )
    non_metallic_minerals.plot(ax=ax, color='orange', markersize = non_metallic_minerals['Peak Load [MWh/h]']/10 )
    other_industry.plot(ax=ax, color='darkred', markersize = other_industry['Peak Load [MWh/h]']/10 )
    
    # Plot for legend
    max_chemicals.plot(ax=ax, color='tan', markersize = 5 , label = 'Chemical Industry')
    max_steel.plot(ax=ax, color='royalblue', markersize = 5 , label = 'Steel and Iron')
    max_paper.plot(ax=ax, color='navy', markersize = 5, label = 'Paper and Printing ')
    max_mineral_processing.plot(ax=ax, color='green', markersize = 5 , label = 'Mineral Processing')
    max_metal_processing.plot(ax=ax, color='olive', markersize = 5 , label = 'Metal Processing')
    max_non_metallic_minerals.plot(ax=ax, color='orange', markersize = 5 , label = 'Non Metallic Minerals')
    max_other_industry.plot(ax=ax, color='darkred', markersize = 5 , label = 'Other Industry')
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
              fancybox=True, shadow=False, ncol=3 , fontsize = 16, markerscale =5)
    #plt.legend(loc = 'lower right')
    plt.axis('off')
    
    ''' Transport VISUALISATION'''
    # Create a point dataframe to ensure the same legend size for each category
    max_passenger = passenger[passenger['Peak Load [MWh/h]']== max(passenger['Peak Load [MWh/h]'])]
    max_public = public[public['Peak Load [MWh/h]']== max(public['Peak Load [MWh/h]'])]
    max_air = air[air['Peak Load [MWh/h]']== max(air['Peak Load [MWh/h]'])]
    max_train_gdf = train_gdf[train_gdf['Peak Load [MWh/h]']== max(train_gdf['Peak Load [MWh/h]'])]
    max_trucks = trucks[trucks['Peak Load [MWh/h]']== max(trucks['Peak Load [MWh/h]'])]
    
    # Plotting of Transport
    fig,ax = plt.subplots(figsize =(15,15))
    
    DE.plot(ax=ax, edgecolor= 'grey', facecolor= 'none')
    
    passenger.plot(ax=ax, color='tan', markersize = passenger['Peak Load [MWh/h]']/5 )
    public.plot(ax=ax, color='royalblue', markersize = public['Peak Load [MWh/h]']/5 )
    air.plot(ax=ax, color='navy', markersize = air['Peak Load [MWh/h]']/5 )
    train_gdf.plot(ax=ax, color='green', markersize = train_gdf['Peak Load [MWh/h]']/5)
    trucks.plot(ax=ax, color='olive', markersize = trucks['Peak Load [MWh/h]']/5 )
    
    # Plot for legend
    max_passenger.plot(ax=ax, color='tan', markersize = 5 , label = 'Passenger Cars')
    max_public.plot(ax=ax, color='royalblue', markersize = 5 , label = 'Public Transport')
    max_air.plot(ax=ax, color='navy', markersize = 5, label = 'Aviation')
    max_train_gdf.plot(ax=ax, color='green', markersize = 5 , label = 'Rail Transport')
    max_trucks.plot(ax=ax, color='olive', markersize = 5 , label = 'Long Distance Trucking')
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
              fancybox=True, shadow=False, ncol=3, fontsize = 16, markerscale =5)
    #plt.legend(loc = 'lower right')
    plt.axis('off')
    
    ''' Transformation VISUALISATION'''
    max_residential = residential[residential['Peak Load [MWh/h]']== max(residential['Peak Load [MWh/h]'])]
    max_district_gdf = district_gdf[district_gdf['Peak Load [MWh/h]']== max(district_gdf['Peak Load [MWh/h]'])]
    
    # Plotting of Transportmation

    fig,ax = plt.subplots(figsize =(15,15))
    
    DE.plot(ax=ax, edgecolor= 'grey', facecolor= 'none')
    
    residential.plot(ax=ax, color='tan', markersize = residential['Peak Load [MWh/h]']/5 )
    district_gdf.plot(ax=ax, color='royalblue', markersize = district_gdf['Peak Load [MWh/h]']/5 )
    
    # Plot for legend
    max_residential.plot(ax=ax, color='tan', markersize = 5 , label = 'Residential Heat')
    max_district_gdf.plot(ax=ax, color='royalblue', markersize = 5 , label = 'Power and District Heat')
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
              fancybox=True, shadow=False, ncol=3, fontsize = 16, markerscale =5)
    #plt.legend(loc = 'lower right')
    plt.axis('off')
    
    ''' Supply VISUALISATION'''
    max_ic = ic[ic['Peak Load [MWh/h]']== max(ic['Peak Load [MWh/h]'])]
    max_lng = lh2[lh2['Peak Load [MWh/h]']== max(lh2['Peak Load [MWh/h]'])]
    max_elec_projects = elec_projects[elec_projects['Peak Load [MWh/h]']== max(elec_projects['Peak Load [MWh/h]'])]
    max_elec_ind_projects = elec_ind_projects[elec_ind_projects['Peak Load [MWh/h]']== max(elec_ind_projects['Peak Load [MWh/h]'])]
    max_storage = storage[storage['Peak Load [MWh/h]']== max(storage['Peak Load [MWh/h]'])]
    
    # Plotting of Supply
    fig,ax = plt.subplots(figsize =(15,15))
    DE.plot(ax=ax, edgecolor= 'grey', facecolor= 'none')
    
    storage.plot(ax=ax, color='olive', markersize = storage['Peak Load [MWh/h]']/5)
    ic.plot(ax=ax, color='tan', markersize = ic['Peak Load [MWh/h]']/5 )
    lh2.plot(ax=ax, color='royalblue', markersize = lh2['Peak Load [MWh/h]']/5 )
    elec_projects.plot(ax=ax, color='navy', markersize = elec_projects['Peak Load [MWh/h]']/5 )
    elec_ind_projects.plot(ax=ax, color='green', markersize = elec_ind_projects['Peak Load [MWh/h]']/5)

    
    # Plot for legend
    max_ic.plot(ax=ax, color='tan', markersize = 5 , label = 'Interconnector')
    max_public.plot(ax=ax, color='royalblue', markersize = 5 , label = 'LH2')
    max_elec_projects.plot(ax=ax, color='navy', markersize = 5, label = 'Annouced Electrolyser Projects')
    max_elec_ind_projects.plot(ax=ax, color='green', markersize = 5 , label = 'Electrolyser at Industry Locations')
    max_storage.plot(ax=ax, color='olive', markersize = 5 , label = 'Storage')
      
    legend1 =plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
              fancybox=True, shadow=False, ncol=3, fontsize = 16, markerscale =5)
    
    plt.axis('off')