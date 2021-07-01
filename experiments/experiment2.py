from tps import TPS
from tps2 import TPS2
import setting
import train
import os
import math
import re
import statistics as st
import torch


# qDE 13, 15, 25 (DE 4.1, 5.1, 7.1)
tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
tensor_param = torch.tensor(
   [-3.0290,  1.0879,  0.2856,  0.5175,  0.6154, -0.2995,  0.8222, # DE 13
   
   -0.6920, 0.4323, -0.3823, -0.2302, -0.0772, -0.5366, -0.3650, 
   -0.2196,  0.4369, 0.2558,  1.1250,  0.4370,  0.0033,  0.1088,  
   0.4402,  0.5828,  0.8322,  0.8916,  1.1063,  0.8197,  0.2413, 
   -0.8434,  0.6201,  0.3956,  0.2308, -0.0526, -0.1066,  0.0896, 
   -2.8432, -0.2690, -0.7206, -0.5235, -0.2751, -0.8235, -0.8548, 
   -0.1116,  1.0191, -0.0207,  0.6347,  0.3095,  0.1902,  0.0505, 
   -1.1904, -0.1000, -0.1190,  0.5080, -0.0410, -0.3482, -0.0153, # DE 15
   
    -1.0923,  1.0923], # DE 25
    requires_grad=True) # 58 params
tps2.update_params(tensor_param)
tpl_list = train.load('./data2/autumn_leaves.txt')
int_list_x2 = train.get_chord_interpretation_list_x2(tpl_list)
graph = train.make_interpretation_graph([tps2, int_list_x2])
back_link_list_x3 = train.get_back_link_list_x3(tps2, graph)
train.debug_show_shortest_paths(tpl_list, graph, back_link_list_x3)
