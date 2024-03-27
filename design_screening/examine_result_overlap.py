import pandas as pd
import networkx as nx
import numpy as np

res_1 = pd.read_csv('duet_all_designed_coils_malb_results.csv')
res_2 = pd.read_csv('duet_dhd0_dhd1_results.csv')
res_3 = pd.read_csv('duet_dhd2_dhd0_malb_results.csv')

set_1 = set(res_1.PPI.to_list())
set_2 = set(res_2.PPI.to_list())
set_3 = set(res_3.PPI.to_list())

#print (set_1, set_2, set_3)
print (len(set_1))
print (len(set_2))
print (len(set_3))

print (set_1.intersection(set_2))
print (set_1.intersection(set_3))
print (set_2.intersection(set_3))

#looking at shared interactions and proteins 

proteins_set_1 = np.array([x.split(':') for x in set_1]).flatten()
print (len(proteins_set_1))
proteins_set_2 = np.array([x.split(':') for x in set_2]).flatten()
print (len(proteins_set_2))
proteins_set_3 = np.array([x.split(':') for x in set_3]).flatten()
print (len(proteins_set_3))


print (len(set(proteins_set_1).intersection(set(proteins_set_2))))
print (len(set(proteins_set_1).intersection(set(proteins_set_3))))
