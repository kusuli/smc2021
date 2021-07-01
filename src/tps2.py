# coding: utf-8

import heapq
import math
import setting
from tps import TPS
import numpy
import torch

class TPS2:
	DISTANCE_ELEMENT_COUNT = 28
	
	def __init__(self, _b_list, _b_pi_list = []):
		self.tps = TPS()
		self.b_list = [1 if i < len(_b_list) and _b_list[i] == 1 else 0 for i in range(self.DISTANCE_ELEMENT_COUNT)]
		self.matrix_list = []
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 2 * 12)]))         # qDE3, DE6.2
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 7 * 2 * 12 * 7)])) # qDE4, DE8.2
		self.matrix_list.append(numpy.array([1.0 for i in range(12)]))                 # qDE5, 
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 12)]))             # qDE6, 
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 12 * 12)]))        # qDE7, 
		self.matrix_list.append(numpy.array([1.0 for i in range(7 * 12)]))             # qDE8, DE7.2
		self.matrix_list.append(numpy.array([1.0 for i in range(4 * 4)]))              # qDE9, 
		self.matrix_list.append(numpy.array([1.0 for i in range(7)]))                  # qDE10, DE5.2
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 7 * 2)]))          # qDE11, 
		self.matrix_list.append(numpy.array([1.0 for i in range(12)]))                 # qDE12, DE5.4
		self.matrix_list.append(numpy.array([1.0 for i in range(7)]))                  # qDE13, DE5.1
		self.matrix_list.append(numpy.array([1.0 for i in range(12)]))                 # qDE14, DE5.3
		self.matrix_list.append(numpy.array([1.0 for i in range(7 * 7)]))              # qDE15, DE7.1
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 7 * 2 * 7 * 7)]))  # qDE16, 
		self.matrix_list.append(numpy.array([1.0 for i in range(3 * 7)]))              # qDE17, 
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 7)]))              # qDE18, DE6.1
		self.matrix_list.append(numpy.array([1.0 for i in range(3)]))                  # qDE19, 
		self.matrix_list.append(numpy.array([1.0 for i in range(4)]))                  # qDE20, DE4.2
		self.matrix_list.append(numpy.array([1.0 for i in range(4)]))                  # qDE21, 
		self.matrix_list.append(numpy.array([1.0 for i in range(2)]))                  # qDE22, 
		self.matrix_list.append(numpy.array([1.0 for i in range(3)]))                  # qDE23, 
		self.matrix_list.append(numpy.array([1.0 for i in range(3 * 7 * 7 * 7)]))      # qDE24, 
		self.matrix_list.append(numpy.array([1.0 for i in range(2)]))                  # qDE25, DE4.1
		self.matrix_list.append(numpy.array([1.0 for i in range(2 * 7 * 7 * 7)]))      # qDE26, DE8.1
		self.matrix_list.append(numpy.array([1.0 for i in range(7)]))                  # qDE27, 
		self.matrix_param_count = 0
		for i in range(self.DISTANCE_ELEMENT_COUNT - 3):
			if self.b_list[i + 3]:
				self.matrix_param_count += len(self.matrix_list[i])
		self.edge_store = {}
		self.edge_tensor_store = {}
	
	def get_distance2(self, keypos_1, scale_1, degree_1, other_1, keypos_2, scale_2, degree_2, other_2):
		dist_list = []
		if self.b_list[0] or self.b_list[1] or self.b_list[2]:
			temp_sum = self.tps.get_distance2(keypos_1, scale_1, degree_1, other_1, keypos_2, scale_2, degree_2, other_2)
		if self.b_list[0]:
			dist_list.append(temp_sum[0])
		if self.b_list[1]:
			dist_list.append(temp_sum[1])
		if self.b_list[2]:
			dist_list.append(temp_sum[2])
		for i in range(self.DISTANCE_ELEMENT_COUNT - 3):
			if self.b_list[i + 3]:
				#dist_list.extend([1.0 if idx == self.get_matrix_coefs(i + 3, keypos_1, scale_1, degree_1, other_1, keypos_2, scale_2, degree_2, other_2) else 0.0 for idx in range(len(self.matrix_list[i]))])
				dist_list.extend(self.get_matrix_coefs(i + 3, keypos_1, scale_1, degree_1, other_1, keypos_2, scale_2, degree_2, other_2))
		return dist_list
	
	def get_scalar_distance(self, dist_list):
		if not tuple(dist_list) in self.edge_store.keys():
			current_index = 0
			dist = 0.0
			if dist_list != TPS.ZERO_TPL:
				if self.b_list[0]:
					dist += dist_list[current_index]
					current_index += 1
				if self.b_list[1]:
					dist += dist_list[current_index]
					current_index += 1
				if self.b_list[2]:
					dist += dist_list[current_index]
					current_index += 1
				for i in range(self.DISTANCE_ELEMENT_COUNT - 3):
					if self.b_list[i + 3]:
						if i + 3 == 9: # root_elm_matrix
							dist += sum(dist_list[current_index : current_index + len(self.matrix_list[i])] * self.matrix_list[i])
							current_index += len(self.matrix_list[i])
						else: # one-hot系
							dist += self.matrix_list[i][dist_list[current_index]]
							current_index += 1
			self.edge_store[tuple(dist_list)] = dist
		return self.edge_store[tuple(dist_list)]
	
	def get_scalar_distance_tensor(self, dist_list, tensor_param):
		if not tuple(dist_list) in self.edge_tensor_store.keys():
			current_dist_index = 0
			current_param_index = 0
			dist_tensor = torch.tensor(0.0)
			if dist_list != TPS.ZERO_TPL:
				if self.b_list[0]:
					dist_tensor += dist_list[current_dist_index]
					current_dist_index += 1
				if self.b_list[1]:
					dist_tensor += dist_list[current_dist_index]
					current_dist_index += 1
				if self.b_list[2]:
					dist_tensor += dist_list[current_dist_index]
					current_dist_index += 1
				for i in range(self.DISTANCE_ELEMENT_COUNT - 3):
					if self.b_list[i + 3]:
						if i + 3 == 9: # root_elm_matrix
							for i2 in range(len(self.matrix_list[i])):
								dist_tensor += tensor_param[current_param_index + i2] * dist_list[current_dist_index + i2]
							current_dist_index += len(self.matrix_list[i])
							current_param_index += len(self.matrix_list[i])
						else: # one-hot系
							dist_tensor += tensor_param[current_param_index + dist_list[current_dist_index]]
							current_dist_index += 1
							current_param_index += len(self.matrix_list[i])
			self.edge_tensor_store[tuple(dist_list)] = dist_tensor
			#return torch.exp(-dist_tensor)
			#return -dist_tensor
		return self.edge_tensor_store[tuple(dist_list)]
	
	def get_matrix_coefs(self, distance_element_index, keypos_1, scale_1, degree_1, other_1, keypos_2, scale_2, degree_2, other_2):
		if distance_element_index == 3: # qDE3
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			key_distance = (keypos_2 - keypos_1) % 12
			temp = b_major_1 * 12 * 2 + b_major_2 * 12 + key_distance
			#return [1.0 if idx == temp else 0.0 for idx in range(len(self.matrix_list[distance_element_index - 3]))]
			return [temp]
		elif distance_element_index == 4: # qDE4
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			key_distance = (keypos_2 - keypos_1) % 12
			temp = b_major_1 * 7 * 2 * 12 * 7 + (degree_1 - 1) * 2 * 12 * 7 + b_major_2 * 12 * 7 + key_distance * 7 + (degree_2 - 1)
			#return [1.0 if idx == temp else 0.0 for idx in range(len(self.matrix_list[distance_element_index - 3]))]
			return [temp]
		elif distance_element_index == 5: # qDE5
			pc_1 = setting.SCALE_DISTANCE_DIC[scale_1][degree_1 - 1]
			pc_2 = (keypos_2 - keypos_1 + setting.SCALE_DISTANCE_DIC[scale_2][degree_2 - 1]) % 12
			temp = pc_1 * 12 + pc_2
			#return [1.0 if idx == temp else 0.0 for idx in range(len(self.matrix_list[distance_element_index - 3]))]
			return [temp]
		elif distance_element_index == 6: # qDE6
			pc_1 = (keypos_1 + setting.SCALE_DISTANCE_DIC[scale_1][degree_1 - 1])
			pc_2 = (keypos_2 + setting.SCALE_DISTANCE_DIC[scale_2][degree_2 - 1])
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			temp = b_major_1 * 12 + ((pc_2 - pc_1) % 12)
			#return [1.0 if idx == temp else 0.0 for idx in range(len(self.matrix_list[distance_element_index - 3]))]
			return [temp]
		elif distance_element_index == 7: # qDE7
			pc_1 = setting.SCALE_DISTANCE_DIC[scale_1][degree_1 - 1]
			pc_2 = (keypos_2 - keypos_1 + setting.SCALE_DISTANCE_DIC[scale_2][degree_2 - 1]) % 12
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			temp = b_major_1 * 12 * 12 + pc_1 * 12 + pc_2
			#return [1.0 if idx == temp else 0.0 for idx in range(len(self.matrix_list[distance_element_index - 3]))]
			return [temp]
		elif distance_element_index == 8: # qDE8
			pc_2 = (keypos_2 - keypos_1 + setting.SCALE_DISTANCE_DIC[scale_2][degree_2 - 1]) % 12 # 遷移元keyの1度の音に対して
			temp = (degree_1 - 1) * 12 + pc_2
			#return [1.0 if idx == temp else 0.0 for idx in range(len(self.matrix_list[distance_element_index - 3]))]
			return [temp]
		elif distance_element_index == 9: # qDE9
			bs1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
			bs2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
			scale1 = setting.SCALE_DISTANCE_DIC[scale_1]
			scale2 = setting.SCALE_DISTANCE_DIC[scale_2]
			bs1[(scale1[(0 + degree_1 - 1) % 7] + keypos_1) % 12] = 1 # root
			bs1[(scale1[(2 + degree_1 - 1) % 7] + keypos_1) % 12] = 2 # 3rd
			bs1[(scale1[(4 + degree_1 - 1) % 7] + keypos_1) % 12] = 3 # 5th
			bs2[(scale2[(0 + degree_2 - 1) % 7] + keypos_2) % 12] = 1 # root
			bs2[(scale2[(2 + degree_2 - 1) % 7] + keypos_2) % 12] = 2 # 3rd
			bs2[(scale2[(4 + degree_2 - 1) % 7] + keypos_2) % 12] = 3 # 5th
			ret = [0.0 for _ in range(4 * 4)]
			for i in range(12):
				ret[bs1[i] * 4 + bs2[i]] += 1.0
			return ret
		elif distance_element_index == 10: # qDE10
			temp = min((keypos_1 - keypos_2) % 12, (keypos_2 - keypos_1) % 12)
			return [temp]
		elif distance_element_index == 11: # qDE11
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			temp = b_major_1 * 7 * 2 + min((keypos_1 - keypos_2) % 12, (keypos_2 - keypos_1) % 12) * 2 + b_major_2
			return [temp]
		elif distance_element_index == 12: # qDE12
			temp = (keypos_2 - keypos_1) % 12
			return [temp]
		elif distance_element_index == 13: # qDE13
			major_keypos_1 = keypos_1
			if scale_1 != setting.SCALE_MAJOR:
				major_keypos_1 += 3
			major_keypos_2 = keypos_2
			if scale_2 != setting.SCALE_MAJOR:
				major_keypos_2 += 3
			temp = min((major_keypos_1 - major_keypos_2) % 12, (major_keypos_2 - major_keypos_1) % 12)
			return [temp]
		elif distance_element_index == 14: # qDE14
			major_keypos_1 = keypos_1
			if scale_1 != setting.SCALE_MAJOR:
				major_keypos_1 += 3
			major_keypos_2 = keypos_2
			if scale_2 != setting.SCALE_MAJOR:
				major_keypos_2 += 3
			temp = (major_keypos_2 - major_keypos_1) % 12
			return [temp]
		elif distance_element_index == 15: # qDE15
			pc_2 = (keypos_2 - keypos_1 + setting.SCALE_DISTANCE_DIC[scale_2][degree_2 - 1]) % 12 # 遷移元keyの1度の音に対して
			temp = (degree_1 - 1) * 7 + min(pc_2, 12 - pc_2)
			return [temp]
		elif distance_element_index == 16: # qDE16
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			key_distance = min((keypos_2 - keypos_1) % 12, (keypos_1 - keypos_2) % 12)
			temp = b_major_1 * 7 * 2 * 7 * 7 + (degree_1 - 1) * 2 * 7 * 7 + b_major_2 * 7 * 7 + key_distance * 7 + (degree_2 - 1)
			return [temp]
		elif distance_element_index == 17: # qDE17
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			temp = (b_major_1 + b_major_2) * 7 + min((keypos_1 - keypos_2) % 12, (keypos_2 - keypos_1) % 12)
			return [temp]
		elif distance_element_index == 18: # qDE18
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			temp1 = 1
			if b_major_1 == b_major_2:
				temp1 = 0
			temp = ((b_major_1 - b_major_2) % 2) * 7 + min((keypos_1 - keypos_2) % 12, (keypos_2 - keypos_1) % 12)
			return [temp]
		elif distance_element_index == 19: # qDE19
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			temp = b_major_1 + b_major_2
			return [temp]
		elif distance_element_index == 20: # qDE20
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			temp = b_major_1 * 2 + b_major_2
			return [temp]
		elif distance_element_index == 21: # qDE21
			degree_distance = min((degree_2 - degree_1) % 7, (degree_1 - degree_2) % 7)
			return [degree_distance]
		elif distance_element_index == 22: # qDE22
			temp1 = 0 # tonic
			if degree_1 == 5 or degree_1 == 7:
				temp1 = 1 # dominant
			elif degree_1 == 2 or degree_1 == 4:
				temp1 = 2 # sub dominant
			temp2 = 0 # tonic
			if degree_2 == 5 or degree_2 == 7:
				temp2 = 1 # dominant
			elif degree_2 == 2 or degree_2 == 4:
				temp2 = 2 # sub dominant
			temp = min((temp2 - temp1) % 3, (temp1 - temp2) % 3)
			return [temp]
		elif distance_element_index == 23: # qDE23
			temp1 = 0 # tonic
			if degree_1 == 5 or degree_1 == 7:
				temp1 = 1 # dominant
			elif degree_1 == 2 or degree_1 == 4:
				temp1 = 2 # sub dominant
			temp2 = 0 # tonic
			if degree_2 == 5 or degree_2 == 7:
				temp2 = 1 # dominant
			elif degree_2 == 2 or degree_2 == 4:
				temp2 = 2 # sub dominant
			temp = (temp2 - temp1) % 3
			return [temp]
		elif distance_element_index == 24: # qDE24
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			key_distance = min((keypos_2 - keypos_1) % 12, (keypos_1 - keypos_2) % 12)
			temp = (b_major_1 + b_major_2) * 7 * 7 * 7 + (degree_1 - 1) * 7 * 7 + (degree_2 - 1) * 7 + key_distance
			return [temp]
		elif distance_element_index == 25: # qDE25
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			temp = (b_major_1 - b_major_2) % 2
			return [temp]
		elif distance_element_index == 26: # qDE26
			b_major_1 = self.tps.scale_to_b_major(scale_1)
			b_major_2 = self.tps.scale_to_b_major(scale_2)
			scale_diff = (b_major_1 - b_major_2) % 2
			key_distance = min((keypos_2 - keypos_1) % 12, (keypos_1 - keypos_2) % 12)
			temp = scale_diff * 7 * 7 * 7 + (degree_1 - 1) * 7 * 7 + (degree_2 - 1) * 7 + key_distance
			return [temp]
		elif distance_element_index == 27: # qDE27
			degree_distance = (degree_2 - degree_1) % 7
			return [degree_distance]
		else:
			print('error', distance_element_index)
			return []

	def update_params(self, tensor_param):
		current_index = 0
		for i in range(self.DISTANCE_ELEMENT_COUNT - 3):
			if self.b_list[i + 3]:
				for i2 in range(len(self.matrix_list[i])):
					self.matrix_list[i][i2] = tensor_param[current_index].item()
					current_index += 1
		#print(self.pi_de_list)
		self.edge_store = {}
		self.edge_tensor_store = {}
