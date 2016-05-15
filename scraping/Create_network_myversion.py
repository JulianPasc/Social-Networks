####################################################################
# Julien Pascal
#
# Code to draw a graph based on the information collected on Twitter
####################################################################

import tweepy
import time
import os
import sys
import json
import argparse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from tabulate import tabulate
import csv
import statsmodels.formula.api as sm

#################################
#ADJUST THE PATHS TO YOUR SETTING
#locate in the correct directory:
path = '/home/julien/Social-Networks/scraping'
path_data = '/home/julien/Social-Networks/data'
path_figures = '/home/julien/Social-Networks/figures'
path_tables = '/home/julien/Social-Networks/tables'

os.chdir(path) 
LINKS_DIR = 'Links'
#################################

#################################
# Define useful functions

############################################"
# Calculate the reciprocity of the network
#
#    Copyright (C) 2015 by 
#    Haochen Wu <wuhaochen42@gmail.com>
#    All rights reserved.
#    BSD license.
#    source: https://networkx.readthedocs.io/en/latest/_modules/networkx/algorithms/reciprocity.html#reciprocity
def reciprocity(G, nodes=None):
    """Compute the reciprocity in a directed graph.

    The reciprocity of a directed graph is defined as the ratio
    of the number of edges pointing in both directions to the total
    number of edges in the graph.
    Formally, :math:`r = |{(u,v) \in G|(v,u) \in G}| / |{(u,v) \in G}|`.

    The reciprocity of a single node u is defined similarly,
    it is the ratio of the number of edges in both directions to
    the total number of edges attached to node u.

    Parameters
    ----------
    G : graph
       A networkx directed graph
    nodes : container of nodes, optional (default=whole graph)
       Compute reciprocity for nodes in this container.

    Returns
    -------
    out : dictionary
       Reciprocity keyed by node label.

    Notes
    -----
    The reciprocity is not defined for isolated nodes.
    In such cases this function will return None.
    
    """
    # If `nodes` is not specified, calculate the reciprocity of the graph.
    if nodes is None:
        return overall_reciprocity(G)

    # If `nodes` represents a single node in the graph, return only its
    # reciprocity.
    if nodes in G:
        reciprocity = next(_reciprocity_iter(G,nodes))[1]
        if reciprocity is None:
            raise NetworkXError('Not defined for isolated nodes.')
        else:
            return reciprocity

    # Otherwise, `nodes` represents an iterable of nodes, so return a
    # dictionary mapping node to its reciprocity.
    return dict(_reciprocity_iter(G,nodes))


def _reciprocity_iter(G,nodes):
    """ Return an iterator of (node, reciprocity).  

    """
    n = G.nbunch_iter(nodes)
    for node in n:
        pred = set(G.predecessors(node))
        succ = set(G.successors(node))
        overlap = pred & succ
        n_total = len(pred) + len(succ)

        # Reciprocity is not defined for isolated nodes.
        # Return None.
        if n_total == 0:
            yield (node,None)
        else:
            reciprocity = 2.0*float(len(overlap))/float(n_total)
            yield (node,reciprocity)
        
def overall_reciprocity(G):
    """Compute the reciprocity for the whole graph.

    See the doc of reciprocity for the definition.

    Parameters
    ----------
    G : graph
       A networkx graph
    
    """
    n_all_edge = G.number_of_edges()
    n_overlap_edge = (n_all_edge - G.to_undirected().number_of_edges()) *2

    if n_all_edge == 0:
        raise NetworkXError("Not defined for empty graphs")

    return float(n_overlap_edge)/float(n_all_edge)

 
########################################
# LOOK AT THE FULL NETWORK
# i.e. including people who
# do not participated directly
# into the negotiation (the 'observers')
########################################

#Retrieve the name of the entities
list_user = [] 
#Load the list of the participants in the thread #COP21MIX between May27th and May 30th:
file_name = path_data + '/' + 'List_COP21MIW.csv' #load the csv file
df = pd.read_csv(file_name)

##########################
#LOOK AT THE FULL NETWORK
Twitter_accounts = [] #Put the names into a list
for i in range(0,len(df)):
  Twitter_accounts.append(df.ix[i,'@name'])

#initialization:
user_json = []
number_nodes = 0

#Load the data in json format:
print('Loading data on Twitter accounts')
print('Entities:')
for i in range(0,len(Twitter_accounts)):
	userfname =  path + '/' + LINKS_DIR + '/' + Twitter_accounts[i] + '.json'
	if os.path.exists(userfname):
		with open(userfname) as data_file:    
			user_json.append(json.load(data_file))
			number_nodes = number_nodes + 1
		print(Twitter_accounts[i]) #print the name of the loaded elemnts of the network


#Look at the links: 
graph = [] #initialization
labels = {}
node_list_States = []
node_list_non_States = []
node_list_observers = []

G=nx.Graph()
for i in range(0, len(user_json)): #move along the "rows" of the link matrix
    for j in range(0, len(user_json)): #move along the "columns" of the link matrix
        if i != j: # a node cannot be linked to itself
            for l in range(0, len(user_json[j]['followers_ids'])):
                if user_json[j]['followers_ids'][l] == user_json[i]['id']:
                    #print('individual %d is following indivudual %d', (user_json[i]['screen_name'],user_json[j]['screen_name']) )
                    graph.append((i,j)) #individuals i is following individual j
                    G.add_node(i)
                    #labels[i] = user_json[i]['name']
                    screen_name = user_json[i]['screen_name']
                    G.add_edge(i,j) 
                    #check the type of the node:
                    if df.loc[df['@name'] == screen_name, 'States'].values == 1:
                      node_list_States.append(i)
                    elif df.loc[df['@name'] == screen_name, 'Non State'].values == 1:
                      node_list_non_States.append(i)
                    else:
                      labels[i] = user_json[i]['name'] #just put label for observers, otherwise impossible to see anything
                      node_list_observers.append(i)


#I have duplicates in the vectors node_list_States and node_list_non_States
#Keep unique elements only:
node_list_States = np.unique(node_list_States).tolist() #convert the np array object into a list object
node_list_non_States = np.unique(node_list_non_States).tolist()
node_list_observers = np.unique(node_list_observers).tolist()

pos = nx.spring_layout(G)

#nx.draw(G, pos, nodecolor='b', edge_color = 'b')

# Draw edges:
nx.draw_networkx_edges(G,pos, edge_color='b', alpha=0.5)

# Draw labels
nx.draw_networkx_labels(G, pos, labels, font_size=10)

# Draw nodes
#Give specific color to groups:
#A. States
nx.draw_networkx_nodes(G,pos,
                       nodelist=node_list_States,
                       node_color='#ff6633',
                       node_size=500,
                       alpha=0.5,
                        label='State')
#B. Non States
nx.draw_networkx_nodes(G,pos,
                       nodelist=node_list_non_States,
                       node_color='#99ff33',
                       node_size=500,
                       alpha=0.5,
                       label='Non State')
# C. Observers:
nx.draw_networkx_nodes(G,pos,
                       nodelist=node_list_observers,
                       node_color='#3399ff',
                       node_size=500,
                       alpha=0.5,
                       label='Observer')

plt.legend(loc='best')
plt.axis('off')
#plt.savefig(path_figures + '' + '/plot_entire_network', orientation='landscape')
plt.show()

nx.number_of_nodes(G)

#########################################
# LOOK AT THE NETWORK RESTRICTED
# TO DIRECT PARTICIPANTS
# i.e THE ONE ENGAGED IN THE NEGOTIATION
#########################################

#Retrieve the name of the entities
list_user = [] 
#Load the list of the participants in the thread #COP21MIX between May27th and May 30th:
file_name = path_data + '/' + 'List_COP21MIW.csv' #load the csv file
df = pd.read_csv(file_name)

##########################
# Keep only Sparticipants
Twitter_accounts = [] #Put the names into a list
for i in range(0,len(df)):
  if df.ix[i,'direct participant'] == 1: # Keep only direct
    Twitter_accounts.append(df.ix[i,'@name'])

#initialization:
user_json = []
number_nodes = 0

#Load the data in json format:
print('Loading data on Twitter accounts')
print('Entities:')
for i in range(0,len(Twitter_accounts)):
  userfname =  path + '/' + LINKS_DIR + '/' + Twitter_accounts[i] + '.json'
  if os.path.exists(userfname):
    with open(userfname) as data_file:    
      user_json.append(json.load(data_file))
      number_nodes = number_nodes + 1
    print(Twitter_accounts[i]) #print the name of the loaded elemnts of the network


#Look at the links: 
graph = [] #initialization
labels = {}
node_list_States = []
node_list_non_States = []
node_other = []

G=nx.Graph()
for i in range(0, len(user_json)): #move along the "rows" of the link matrix
    for j in range(0, len(user_json)): #move along the "columns" of the link matrix
        if i != j: # a node cannot be linked to itself
            for l in range(0, len(user_json[j]['followers_ids'])):
                if user_json[j]['followers_ids'][l] == user_json[i]['id']:
                    #print('individual %d is following indivudual %d', (user_json[i]['screen_name'],user_json[j]['screen_name']) )
                    graph.append((i,j)) #individuals i is following individual j
                    G.add_node(i)
                    labels[i] = user_json[i]['name']
                    screen_name = user_json[i]['screen_name']
                    G.add_edge(i,j) 
                    #check the type of the node:
                    if df.loc[df['@name'] == screen_name, 'States'].values == 1:
                      node_list_States.append(i)
                    elif df.loc[df['@name'] == screen_name, 'Non State'].values == 1:
                      node_list_non_States.append(i)
                    else:
                      node_other.append(i)


#I have duplicates in the vectors node_list_States and node_list_non_States
#Keep unique elements only:
node_list_States = np.unique(node_list_States).tolist() #convert the np array object into a list object
node_list_non_States = np.unique(node_list_non_States).tolist()
node_other = np.unique(node_other).tolist()

pos = nx.spring_layout(G)

#nx.draw(G, pos, nodecolor='b', edge_color = 'b')

# Draw edges:
nx.draw_networkx_edges(G,pos, edge_color='b', alpha=0.5)

# Draw labels
nx.draw_networkx_labels(G, pos, labels, font_size=10)

# Draw nodes
#Give specific color to groups:
#A. States
nx.draw_networkx_nodes(G,pos,
                       nodelist=node_list_States,
                       node_color='#ff6633',
                       node_size=500,
                       alpha=0.5,
                        label='State')
#B. Non States
nx.draw_networkx_nodes(G,pos,
                       nodelist=node_list_non_States,
                       node_color='#99ff33',
                       node_size=500,
                       alpha=0.5,
                       label='Non State')
# C. Observers:
nx.draw_networkx_nodes(G,pos,
                       nodelist=node_other,
                       node_color='#3399ff',
                       node_size=500,
                       alpha=0.5,
                       label='Other')

plt.legend(loc='best')
plt.axis('off')
#plt.savefig(path_figures + '' + '/plot_network_direct_participants', orientation='landscape')
plt.show()

################################
# DIRECTED GRAPH OF PARTICIPANTS
################################
# Get rid of one observation ('Eva Sahores')
# otherwise the plot of the graph is not readable.
# She follows to entities, but she is 
# not followed by anyone. Hence she is very far away
# from the rest of the network. 
list_user = [] 
#Load the list of the participants in the thread #COP21MIX between May27th and May 30th:
file_name = path_data + '/' + 'List_COP21MIW.csv' #load the csv file
df = pd.read_csv(file_name)

##########################
# Keep only Sparticipants
Twitter_accounts = [] #Put the names into a list
for i in range(0,len(df)):
  if df.ix[i,'direct participant'] == 1: # Keep only direct
    Twitter_accounts.append(df.ix[i,'@name'])

#initialization:
user_json = []
number_nodes = 0

#Load the data in json format:
print('Loading data on Twitter accounts')
print('Entities:')
for i in range(0,len(Twitter_accounts)):
  userfname =  path + '/' + LINKS_DIR + '/' + Twitter_accounts[i] + '.json'
  if os.path.exists(userfname):
    with open(userfname) as data_file:    
      user_json.append(json.load(data_file))
      number_nodes = number_nodes + 1
    print(Twitter_accounts[i]) #print the name of the loaded elemnts of the network


#Look at the links: 
graph = [] #initialization
labels = {}
node_list_States = []
node_list_non_States = []
node_other = []

G=nx.DiGraph() #create a directed graph
for i in range(0, len(user_json)): #move along the "rows" of the link matrix
    for j in range(0, len(user_json)): #move along the "columns" of the link matrix
        if i != j: # a node cannot be linked to itself
          if user_json[i]['name'] != 'Eva Sahores': #otherwise the graph is not readable
            for l in range(0, len(user_json[j]['followers_ids'])):
                if user_json[j]['followers_ids'][l] == user_json[i]['id']:
                    #print('individual %d is following indivudual %d', (user_json[i]['screen_name'],user_json[j]['screen_name']) )
                    graph.append((i,j)) #individuals i is following individual j
                    G.add_nodes_from([i,j])
                    labels[i] = user_json[i]['name']
                    screen_name = user_json[i]['screen_name']
                    G.add_edge(i,j)  #i follows j
                    #check the type of the node:
                    if df.loc[df['@name'] == screen_name, 'States'].values == 1:
                      node_list_States.append(i)
                    elif df.loc[df['@name'] == screen_name, 'Non State'].values == 1:
                      node_list_non_States.append(i)
                    else:
                      node_other.append(i)


#I have duplicates in the vectors node_list_States and node_list_non_States
#Keep unique elements only:
node_list_States = np.unique(node_list_States).tolist() #convert the np array object into a list object
node_list_non_States = np.unique(node_list_non_States).tolist()
node_other = np.unique(node_other).tolist()

pos = nx.spring_layout(G)

#nx.draw(G, pos, nodecolor='b', edge_color = 'b')

# Draw edges:
nx.draw_networkx_edges(G,pos, edge_color='b', alpha=0.5, arrows=True)

# Draw labels
nx.draw_networkx_labels(G, pos, labels, font_size=10)

# Draw nodes
#Give specific color to groups:
#A. States
nx.draw_networkx_nodes(G,pos,
                       nodelist=node_list_States,
                       node_color='#ff6633',
                       node_size=500,
                       alpha=0.5,
                        label='State')
#B. Non States
nx.draw_networkx_nodes(G,pos,
                       nodelist=node_list_non_States,
                       node_color='#99ff33',
                       node_size=500,
                       alpha=0.5,
                       label='Non State')
# C. Observers:
nx.draw_networkx_nodes(G,pos,
                       nodelist=node_other,
                       node_color='#3399ff',
                       node_size=500,
                       alpha=0.5,
                       label='Other')

plt.legend(loc='best')
plt.axis('off')
#plt.savefig(path_figures + '' + '/plot_directed_network_direct_participants', orientation='landscape')
plt.show()

################################################
# Compute the "in" and "out" degree distribution
# for the directed graph of direct participants
################################################
# 'in' = someone is following me
# 'out' = I am following someone

# A. 'In' degrees ditribution: 
in_degrees  = G.in_degree() #dictionary type
in_values = sorted(set(in_degrees.values())) # get the possible values for "in_degrees" in the population

in_hist = [] #initialisation

for i in in_values: #move along the possible "in" values (number of followers)
  number = 0 #initialisation. going to store the number of times this "in" value appeas in the distribution.
  for j in set(in_degrees.keys()): #move along the distibution of "in" degrees:
    if in_degrees[j] == i:
      number = number + 1 #incrementation
  in_hist.append(number) #store the number of times the "in" degree "i" appears in the population

#B. 'Out' degrees distribution
out_degrees  = G.out_degree() #dictionary type
out_values = sorted(set(out_degrees.values())) # get the possible values for "in_degrees" in the population

out_hist = [] #initialisation

for i in out_values: #move along the possible "in" values (number of followers)
  number = 0 #initialisation. going to store the number of times this "in" value appeas in the distribution.
  for j in set(out_degrees.keys()): #move along the distibution of "in" degrees:
    if out_degrees[j] == i:
      number = number + 1 #incrementation
  out_hist.append(number) #store the number of times the "in" degree "i" appears in the population


##########################
# Plot degree distribution
plt.figure()
# in-degrees : 
plt.plot(in_values,in_hist,'ro-') 
# out-degrees:
plt.plot(out_values,out_hist,'bv-') 

plt.legend(['In-degree','Out-degree'], loc='best')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
#plt.savefig(path_figures + '' +'/degrees_distribution.png')
plt.show()

##############################
# Summary Stats on the network: 
av_reciprocity = reciprocity(G)

# to calculate average clustering coefficient,
# I have to convert the directed graph 'G' into
# an indirected one:
G_undirected = G.to_undirected()

# average clustering coeff:
av_clustering = nx.average_clustering(G_undirected)

# Diamater of the graph:
# Defined as the maximum eccentricity. The eccentricity of a node v is the maximum distance from v to all other nodes in G.
diameter_graph = nx.diameter(G_undirected)

# Average distance in the graph:
average_distance = nx.average_shortest_path_length(G_undirected)

# Number of nodes:
nb_nodes = nx.number_of_nodes(G_undirected)
nb_nodes

# Number of edges:
nb_edges = nx.number_of_edges(G_undirected)
nb_edges 

# Create a table with the previously created summary statistics:
# Create a table:
table = [['number nodes', nb_nodes], 
['number edges', nb_edges],
['diameter', diameter_graph],
['average distance', average_distance], 
['average clustering', av_clustering],
['average reciprocity', av_reciprocity]]

# print a lateX file:
print(tabulate(table, headers=['Statistic', 'Estimate'], floatfmt=".3f", tablefmt="latex"))

# save into a csv file:
with open(path_tables + '/table_summary_statistics.csv', 'w') as csvfile:
  writer = csv.writer(csvfile)
  [writer.writerow(r) for r in table]

########################
# Measurement of centrality
########################

# Degree Centrality:
# The degree centrality for a node v is the fraction of nodes it is connected to
deg_centrality = nx.degree_centrality(G)

# Betweenness
# Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v:
betw_centrality = nx.betweenness_centrality(G)

# Closeness
# Closeness centrality [1] of a node u is the reciprocal of the sum of the shortest path distances from u to all n-1 other nodes.
# Since the sum of distances depends on the number of nodes in the graph, closeness is normalized by the sum of minimum possible distances n-1.
clo_centrality = nx.closeness_centrality(G)

# Eigenvalue centrality
eig_centrality = nx.eigenvector_centrality(G)

# Convert the dictionary to a dataframe
# Easier to manipulate:

#initialisation:
ind = 0
list_index = []
list_degree_centrality = []
list_betweenness_centrality = []
list_closeness_centrality = []
list_eigen_value_centrality = []
list_label = [] #Twitter name of entities
list_state = [] #dummy variable, if State
list_non_state = [] #dummy variable, if Non State
list_other = [] #dummy variable, if Observers
list_color = [] #give a color to each type

for i in set(deg_centrality.keys()): #loop over the 'keys' of the dictionary 'deg_centrality'
  list_index.append(ind)
  ind = ind + 1
  list_degree_centrality.append(deg_centrality[i])
  list_betweenness_centrality.append(betw_centrality[i])
  list_closeness_centrality.append(clo_centrality[i])
  list_eigen_value_centrality.append(eig_centrality[i])
  list_label.append(labels[i])
  screen_name = user_json[i]['screen_name']
  if df.loc[df['@name'] == screen_name, 'States'].values == 1:
    list_state.append(1) 
    list_non_state.append(0) 
    list_other.append(0) 
    list_color.append('#ff6633') 
  elif df.loc[df['@name'] == screen_name, 'Non State'].values == 1:
    list_state.append(0) 
    list_non_state.append(1) 
    list_other.append(0) 
    list_color.append('#99ff33') 
  else:
    list_state.append(0) 
    list_non_state.append(0) 
    list_other.append(1) 
    list_color.append('#3399ff') 

#turn the lists into a dataframe:
df_centrality = pd.DataFrame(list_degree_centrality, index= list_index, columns=['degree_centrality'])

#A bit cumbersome. I successively merge several dataframes:
#must exist a more efficient way of doing this
df_centrality = pd.merge(df_centrality, pd.DataFrame(list_betweenness_centrality, index= list_index, columns=['betweenness_centrality']), left_index = True, right_index = True)
df_centrality = pd.merge(df_centrality, pd.DataFrame(list_closeness_centrality, index= list_index, columns=['closeness_centrality']), left_index = True, right_index = True)
df_centrality = pd.merge(df_centrality, pd.DataFrame(list_eigen_value_centrality, index= list_index, columns=['eigenvalue_centrality']), left_index = True, right_index = True)
df_centrality = pd.merge(df_centrality, pd.DataFrame(list_label, index= list_index, columns=['name']), left_index = True, right_index = True)
df_centrality = pd.merge(df_centrality, pd.DataFrame(list_state, index= list_index, columns=['State']), left_index = True, right_index = True)
df_centrality = pd.merge(df_centrality, pd.DataFrame(list_non_state, index= list_index, columns=['Non_State']), left_index = True, right_index = True)
df_centrality = pd.merge(df_centrality, pd.DataFrame(list_other, index= list_index, columns=['Other']), left_index = True, right_index = True)
df_centrality = pd.merge(df_centrality, pd.DataFrame(list_color, index= list_index, columns=['Color']), left_index = True, right_index = True)


#Save the dataframe
df_centrality.to_csv(path_tables + '/centrality_participants.csv')

df_centrality_sorted = df_centrality.sort_values(by = 'eigenvalue_centrality', ascending=0) #sort by eigenvalue centrality
#Print a lateX file:
print(df_centrality_sorted[['degree_centrality','betweenness_centrality', 'closeness_centrality', 'eigenvalue_centrality','name','State','Non_State','Other']].to_latex())

#Plot degree centrality and closeness centrality:
fig, ax = plt.subplots()
ax.scatter(df_centrality.loc[df_centrality['State'] ==1, 'degree_centrality'], df_centrality.loc[df_centrality['State'] ==1, 'closeness_centrality'], color = df_centrality.loc[df_centrality['State'] ==1, 'Color'], s = 100)
ax.scatter(df_centrality.loc[df_centrality['Non_State'] ==1, 'degree_centrality'], df_centrality.loc[df_centrality['Non_State'] ==1, 'closeness_centrality'], color = df_centrality.loc[df_centrality['Non_State'] ==1, 'Color'], s = 100)
ax.scatter(df_centrality.loc[df_centrality['Other'] ==1, 'degree_centrality'], df_centrality.loc[df_centrality['Other'] ==1, 'closeness_centrality'], color = df_centrality.loc[df_centrality['Other'] ==1, 'Color'], s = 100)
plt.xlabel('degree centrality')
plt.ylabel('closeness centrality')
#for i, row in df_centrality.iterrows(): 
#  ax.annotate(df_centrality['name'][i],(df_centrality['degree_centrality'][i],df_centrality['closeness_centrality'][i]))

plt.legend(['State','Non Sate','Other'],loc=4)
#plt.savefig(path_figures + '' +'/degree_closenness_centrality.png')
plt.show()

#Plot eigenvalue centrality and betweenness centrality:
fig, ax = plt.subplots()
ax.scatter(df_centrality.loc[df_centrality['State'] ==1, 'closeness_centrality'], df_centrality.loc[df_centrality['State'] ==1, 'eigenvalue_centrality'], color = df_centrality.loc[df_centrality['State'] ==1, 'Color'], s = 100)
ax.scatter(df_centrality.loc[df_centrality['Non_State'] ==1, 'closeness_centrality'], df_centrality.loc[df_centrality['Non_State'] ==1, 'eigenvalue_centrality'], color = df_centrality.loc[df_centrality['Non_State'] ==1, 'Color'], s = 100)
ax.scatter(df_centrality.loc[df_centrality['Other'] ==1, 'closeness_centrality'], df_centrality.loc[df_centrality['Other'] ==1, 'eigenvalue_centrality'], color = df_centrality.loc[df_centrality['Other'] ==1, 'Color'], s = 100)
plt.xlabel('closeness centrality')
plt.ylabel('eigenvalue centrality')

plt.legend(['State','Non Sate','Other'],loc=4)

#plt.savefig(path_figures + '' +'/eigenvalue_closenness_centrality.png')
plt.show()


################################################
#Regress the eigenvalue centrality_participants:
################################################
result = sm.ols(formula=" eigenvalue_centrality ~ State + Non_State", data=df_centrality).fit(cov_type='HC0')
print(result.summary().as_latex ())


##################################
# F test for equality of estimates
##################################
#Get the estimates:
estimates  = result.params

#number of observations:
nb_obs = result.nobs

#number of explanatory + constant:
k  = 3

#degree_freedom :
degree_freedom = nb_obs - 3

#Store the variance covariance matrix:
Var_cov = result.cov_HC0

#Build a F test to test the equality of estimates 'States' and 'Non States': 
F = abs(estimates[1]-estimates[2])/np.sqrt(Var_cov[1,1] + Var_cov[2,2] - 2*(Var_cov[1,2])) 

print(F)




