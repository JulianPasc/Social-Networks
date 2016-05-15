# Social-Networks
Repository that contains the codes and the data that were used to produce my research proposal for the class in Social Networks by Professor Emeric Henry: http://www.sciencespo.fr/ecole-doctorale/sites/sciencespo.fr.ecole-doctorale/files/Social_Network.pdf

## Summary
My research proposal offers a network analysis of the COP 21 Make It Work, which was a 3-day simulation of the COP 21 that involved 208 students from all around the world. The relationship between the Twitter network of participants and the outcome of the simulation is analysed, especially with regards to the centrality of delegations. This analysis underlines that statistics, such as the eigenvalue centrality, might be good indicators of bargaining power - i.e weight in the negotiations

### Example
Graph of the entire COP21 MIW network:
![GitHub Logo](https://github.com/JulianPasc/Social-Networks/blob/master/figures/plot_entire_network.png)
Source: author's calculations based on data from Twitter. Notes: State and Non State represent Twitter accounts of delegations that participated in the COP21 MIW simulation. Observers are other entities that posted on Twitter under the hash-tag #COP21MIW between 29th and 31stth May 2015, but could not be counted as delegations. 

## Requirements
- Python 3.5.1 |Anaconda 2.4.1 (64-bit)| (default, Dec 7 2015, 11:16:01) [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux was used. 
- Paths in "Create_network_myversion.py" and "get_followers_myversion.py" have to be adjusted to your environment
- One needs Twitter API keys to use "get_followers_myversion.py"

## Structure of the codes:
- "get_followers_myversion.py" collects data on Twitter accounts using the Twitter API
- "Create_network_myversion.py" uses NetworkX to analyse the network
