# coding: utf-8

import heapq
import math
import util
import setting
class TPS:
	INF_TPL = (999999,)
	ZERO_TPL = (999998,)
	
	def __init__(self):
		self.coef_i = 1/3
		self.coef_j = 1/3
		#self.coef_k = 1.0
		self.coef_sum = 1.0
		self.stored_distances = {}
	
	def get_distance(self, keynote_1, b_major_1, degree_1, keynote_2, b_major_2, degree_2):
		return self.get_scalar_distance(self.get_distance_bs(keynote_1, b_major_1, degree_1, [], keynote_2, b_major_2, degree_2, []))
	
	def get_distance_bs(self, keynote_1, b_major_1, degree_1, other_1, keynote_2, b_major_2, degree_2, other_2):
		keypos_1 = setting.NOTE_POS_DIC[keynote_1]
		keypos_2 = setting.NOTE_POS_DIC[keynote_2]
		if b_major_1:
			scale_1 = setting.SCALE_MAJOR
		else:
			scale_1 = setting.SCALE_NATURAL_MINOR
		if b_major_2:
			scale_2 = setting.SCALE_MAJOR
		else:
			scale_2 = setting.SCALE_NATURAL_MINOR
		return self.get_distance2(keypos_1, scale_1, degree_1, other_1, keypos_2, scale_2, degree_2, other_2)
	
	def scale_to_b_major(self, scale):
		return scale == setting.SCALE_MAJOR
	
	def get_distance2(self, keypos_1, scale_1, degree_1, other_1, keypos_2, scale_2, degree_2, other_2, b_close = False):
		if (keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2) in self.stored_distances:
			return self.stored_distances[(keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2)]
		b_major_1 = self.scale_to_b_major(scale_1)
		b_major_2 = self.scale_to_b_major(scale_2)
		if b_close or self.is_close_key(keypos_1, b_major_1, keypos_2, b_major_2): # related key
			sum1 = self.get_region_distance(keypos_1, b_major_1, keypos_2, b_major_2)
			sum2 = self.get_chord_distance(keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2)
			sum3 = self.get_basicspace_distance(keypos_1, scale_1, degree_1, other_1, keypos_2, scale_2, degree_2, other_2)
			sum_relative_key = self.get_relative_key_value(keypos_1, b_major_1, keypos_2, b_major_2)
			sum_other = self.get_other_distance(keypos_1, scale_1, degree_1, other_1, keypos_2, scale_2, degree_2, other_2)
			return (sum1, sum2, sum3, sum_relative_key, sum_other)
		else:
			sum_tpl = (0, 0, 0, 0, 0)
			reached = {}
			for close_key, _ in self.get_close_key_list(keypos_1, scale_1).items():
				reached[close_key] = self.get_distance2(keypos_1, scale_1, degree_1, other_1, close_key[0], close_key[1], 1, [], True)
			for i in range(100):
				k1 = min(reached, key=lambda x: max(0, self.get_scalar_distance(reached[x])))
				v1 = reached[k1]
				if (keypos_2, b_major_2) in reached and reached[(keypos_2, b_major_2)] == v1:
					sum_tpl = (sum_tpl[0] + v1[0], sum_tpl[1] + v1[1], sum_tpl[2] + v1[2], sum_tpl[3] + v1[3], sum_tpl[4] + v1[4])
					break
				close_key_list = self.get_close_key_list(k1[0], k1[1], True, reached)
				reached[k1] = TPS.INF_TPL
				for k2, v2 in close_key_list.items():
					if k2 == (keypos_2, scale_2 == setting.SCALE_MAJOR) and degree_2 != 1:
						v2 = self.get_distance2(k1[0], k1[1], 1, [], keypos_2, scale_2, degree_2, other_2, True)
					if (not k2 in reached) or (reached[k2] != TPS.INF_TPL and max(0, self.get_scalar_distance(reached[k2])) > max(0, self.get_scalar_distance(v1)) + max(0, self.get_scalar_distance(v2))):
						reached[k2] = (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2], v1[3] + v2[3], v1[4] + v2[4])
		self.stored_distances[(keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2)] = sum_tpl
		return sum_tpl
	
	def get_scalar_distance(self, dist_tpl):
		#print(dist_tpl)
		if dist_tpl == TPS.ZERO_TPL:
			return 0
		elif dist_tpl == TPS.INF_TPL:
			return 99999
		else:
			return dist_tpl[0] * self.coef_i + dist_tpl[1] * self.coef_j + dist_tpl[2] * (self.coef_sum - self.coef_i - self.coef_j)

	def is_close_key(self, keypos_1, b_major_1, keypos_2, b_major_2):
		if (keypos_1, b_major_1) == (keypos_2, b_major_2):
			return True
		else:
			close_key_list = self.get_close_key_list(keypos_1, b_major_1)
			#print(close_key_list)
			for k in close_key_list:
				if (keypos_2, b_major_2) == (k[0], k[1]):
					return True
			return False

	def get_close_key_list(self, keypos, b_major, b_with_distance = False, reached = []):
		ret = {(keypos, not b_major): -1, ((keypos + 7) % 12, b_major): -1, ((keypos + 5) % 12, b_major): -1};
		if b_major:
			ret[((keypos - 3) % 12, not b_major)] = -1
			ret[((keypos + 7 - 3) % 12, not b_major)] = -1
			ret[((keypos + 5 - 3) % 12, not b_major)] = -1
		else:
			ret[((keypos + 3) % 12, not b_major)] = -1
			ret[((keypos + 7 + 3) % 12, not b_major)] = -1
			ret[((keypos + 5 + 3) % 12, not b_major)] = -1
		if b_with_distance:
			for k in ret:
				if k not in reached or reached[k] != TPS.INF_TPL:
					scale_1 = setting.SCALE_MAJOR if b_major else setting.SCALE_NATURAL_MINOR
					scale_2 = setting.SCALE_MAJOR if k[1] else setting.SCALE_NATURAL_MINOR
					ret[k] = self.get_distance2(keypos, scale_1, 1, [], k[0], scale_2, 1, [], True)
		return ret

	get_region_distance_arr = [9, 2, 7, 0, 5, 10, 3, 8, 1, 6, 11, 4]
	def get_region_distance(self, keypos_1, b_major_1, keypos_2, b_major_2):
		modpos_1 = keypos_1
		if not b_major_1:
			modpos_1 = (modpos_1 + 3) % 12
		modpos_2 = keypos_2
		if not b_major_2:
			modpos_2 = (modpos_2 + 3) % 12
		mod = (self.get_region_distance_arr[modpos_1] - self.get_region_distance_arr[modpos_2]) % 12;
		return min(mod, 12 - mod)

	keypos_to_digree = {0: 0, 2: 1, 3: 2, 4: 2, 5: 3, 7: 4, 8: 5, 9: 5, 10: 6}
	get_chord_distance_arr = [0, 5, 3, 1, 6, 4, 2]
	def get_chord_distance(self, keypos_1, scale_1, degree_1, keypos_2, scale_2, degree_2):
		mod = (self.get_chord_distance_arr[((degree_2 + self.keypos_to_digree[(keypos_2 - keypos_1) % 12]) % 7) - (degree_1 % 7)]) % 7
		return min(mod, 7 - mod)

	def get_basicspace_distance(self, keypos_1, scale_1, degree_1, other_1, keypos_2, scale_2, degree_2, other_2):
		#return 0
		bs1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		bs2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		scale1 = setting.SCALE_DISTANCE_DIC[scale_1]
		scale2 = setting.SCALE_DISTANCE_DIC[scale_2]
		bs1[(scale1[0] + keypos_1) % 12] += 1
		bs1[(scale1[1] + keypos_1) % 12] += 1
		bs1[(scale1[2] + keypos_1) % 12] += 1
		bs1[(scale1[3] + keypos_1) % 12] += 1
		bs1[(scale1[4] + keypos_1) % 12] += 1
		bs1[(scale1[5] + keypos_1) % 12] += 1
		bs1[(scale1[6] + keypos_1) % 12] += 1
		bs2[(scale2[0] + keypos_2) % 12] += 1
		bs2[(scale2[1] + keypos_2) % 12] += 1
		bs2[(scale2[2] + keypos_2) % 12] += 1
		bs2[(scale2[3] + keypos_2) % 12] += 1
		bs2[(scale2[4] + keypos_2) % 12] += 1
		bs2[(scale2[5] + keypos_2) % 12] += 1
		bs2[(scale2[6] + keypos_2) % 12] += 1
		bs1[(scale1[(0 + degree_1 - 1) % 7] + keypos_1) % 12] += 3
		bs1[(scale1[(2 + degree_1 - 1) % 7] + keypos_1) % 12] += 1
		bs1[(scale1[(4 + degree_1 - 1) % 7] + keypos_1) % 12] += 2
		bs2[(scale2[(0 + degree_2 - 1) % 7] + keypos_2) % 12] += 3
		bs2[(scale2[(2 + degree_2 - 1) % 7] + keypos_2) % 12] += 1
		bs2[(scale2[(4 + degree_2 - 1) % 7] + keypos_2) % 12] += 2
		sum = 0
		for i in range(12):
			sum += max(0, bs2[i] - bs1[i])
		return sum
	
	def get_relative_key_value(self, keypos_1, b_major_1, keypos_2, b_major_2):
		modpos_1 = keypos_1
		if not b_major_1:
			modpos_1 = (modpos_1 + 3) % 12
		modpos_2 = keypos_2
		if not b_major_2:
			modpos_2 = (modpos_2 + 3) % 12
		mod = (self.get_region_distance_arr[modpos_1] - self.get_region_distance_arr[modpos_2]) % 12;
		if modpos_1 == modpos_2 and b_major_1 != b_major_2:
			return 1
		else:
			return 0
	
	def get_other_distance(self, keypos_1, scale_1, degree_1, other_1, keypos_2, scale_2, degree_2, other_2):
		return 0
