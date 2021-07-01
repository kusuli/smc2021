import os
import math
import re
import torch
import numpy
import statistics as st
import random
from multiprocessing import Pool

from tps import TPS
from tps2 import TPS2
import setting

denom_mode = 'default'

def load(data_path):
	tpl_list = []
	with open(data_path, "r") as f:
		for line in f:
			tpl = line.split(',')
			if len(tpl) == 3:
				tpl_list.append((int(tpl[0]), int(tpl[1]), int(tpl[2])))
	return tpl_list

def convert_tpl(tpl):
	root_pos = tpl[0] + setting.SCALE_DISTANCE_DIC[tpl[1]][tpl[2] - 1]
	root_pos %= 12
	chord_str = [k for k, v in setting.NOTE_POS_DIC.items() if v == root_pos][0]
	if tpl[1] == setting.SCALE_MAJOR:
		if tpl[2] == 1 or tpl[2] == 4 or tpl[2] == 5:
			chord_str += ':maj'
			chord_type = setting.CHORD_TYPE_MAJ
		elif tpl[2] == 2 or tpl[2] == 3 or tpl[2] == 6:
			chord_str += ':min'
			chord_type = setting.CHORD_TYPE_MIN
		elif tpl[2] == 7:
			chord_str += ':dim'
			chord_type = setting.CHORD_TYPE_DIM7
		else:
			print('convert_tpl_to_chord_name エラー: ' + tpl)
			raise Exception
	else:
		if tpl[2] == 3 or tpl[2] == 6 or tpl[2] == 7:
			chord_str += ':maj'
			chord_type = setting.CHORD_TYPE_MAJ
		elif tpl[2] == 1 or tpl[2] == 4 or tpl[2] == 5:
			chord_str += ':min'
			chord_type = setting.CHORD_TYPE_MIN
		elif tpl[2] == 2:
			chord_str += ':dim'
			chord_type = setting.CHORD_TYPE_DIM7
		else:
			print('convert_tpl_to_chord_name エラー: ' + tpl)
			raise Exception
	return (chord_str, root_pos, chord_type)

def get_chord_interpretation_list(tpl):
	(chord_str, root_pos, chord_type) = convert_tpl(tpl)
	#print(chord_str, root_pos, chord_type)
	if chord_type == setting.CHORD_TYPE_MAJ:
		return [
				((root_pos, setting.SCALE_MAJOR, 1), 0),
				(((root_pos + 9) % 12, setting.SCALE_NATURAL_MINOR, 3), 0),
				(((root_pos + 7) % 12, setting.SCALE_MAJOR, 4), 0),
				(((root_pos + 4) % 12, setting.SCALE_NATURAL_MINOR, 6), 0),
				(((root_pos + 5) % 12, setting.SCALE_MAJOR, 5), 0),
				(((root_pos + 2) % 12, setting.SCALE_NATURAL_MINOR, 7), 0)
		]
	elif chord_type == setting.CHORD_TYPE_MIN:
		return [
				((root_pos, setting.SCALE_NATURAL_MINOR, 1), 0),
				(((root_pos + 3) % 12, setting.SCALE_MAJOR, 6), 0),
				(((root_pos + 7) % 12, setting.SCALE_NATURAL_MINOR, 4), 0),
				(((root_pos + 10) % 12, setting.SCALE_MAJOR, 2), 0),
				(((root_pos + 5) % 12, setting.SCALE_NATURAL_MINOR, 5), 0),
				(((root_pos + 8) % 12, setting.SCALE_MAJOR, 3), 0)
		]
	elif chord_type == setting.CHORD_TYPE_DIM7:
		return [
				(((root_pos + 1) % 12, setting.SCALE_MAJOR, 7), 0),
				(((root_pos + 10) % 12, setting.SCALE_NATURAL_MINOR, 2), 0)
		]
	else:
		print('get_chord_interpretation エラー: ' + tpl)
		raise Exception

def get_chord_interpretation_list_x2(tpl_list):
	ret_list = []
	for tpl in tpl_list:
		ret_list.append(get_chord_interpretation_list(tpl))
		#print(tpl, ret_list[-1])
	return ret_list

def make_interpretation_graph(arg_list):
	(tps2, interpretation_list_x2) = arg_list
	node_list_x2 = [] # [layer index: [node index in the layer: [key, scale, degree, others, prob]]]
	edge_distance_list_x3 = [] # [src layer index: [src node index: [dest node index: (TPS2距離係数LIST)]]]
	# node_list_x2
	for i, interpretation_list in enumerate(interpretation_list_x2):
		node_list = []
		for i2, interpretation in enumerate(interpretation_list):
			node_list.append([interpretation[0][0], interpretation[0][1], interpretation[0][2], [], 0])
		node_list_x2.append(node_list)
	# edge_distance_list_x3
	for i, node_list in enumerate(node_list_x2):
		if i < len(node_list_x2) - 1:
			edge_distance_list_x2 = []
			for i2, node in enumerate(node_list):
				edge_distance_list = []
				for i3, next_node in enumerate(node_list_x2[i + 1]):
					sum_tpl = tps2.get_distance2(node[0], node[1], node[2], node[3], next_node[0], next_node[1], next_node[2], next_node[3])
					#edge_distance_list.append([tps.get_scalar_distance(sum_tpl), sum_tpl])
					edge_distance_list.append(sum_tpl)
				edge_distance_list_x2.append(edge_distance_list)
			edge_distance_list_x3.append(edge_distance_list_x2)
	# end node
	edge_distance_list_x2 = []
	for i, node in enumerate(node_list_x2[-1]):
		edge_distance_list_x2.append([TPS.ZERO_TPL])
	edge_distance_list_x3.append(edge_distance_list_x2)
	node_list_x2.append([[0, 0, 0, [], 0]])
	return (node_list_x2, edge_distance_list_x3)

#  returns: [layer index: [node index: [back node index]]]
def get_back_link_list_x3(tps2, graph):
	small_value = 0.00000001
	node_list_x2 = graph[0]
	edge_distance_list_x3 = graph[1]
	back_link_list_x3 = []
	node_cost_list_x2 = []
	# first layer
	back_link_list_x3.append([[] for i in node_list_x2[0]])
	node_cost_list_x2.append([0 for i in node_list_x2[0]])
	# following layers
	for n in range(len(node_list_x2) - 1):
		src_node_list = node_list_x2[n]
		dest_node_list = node_list_x2[n + 1]
		back_link_list_x2 = []
		node_cost_list = []
		for dest_node_index, dest_node in enumerate(dest_node_list):
			min_distance = math.inf
			cand_list = []
			for src_node_index, src_node in enumerate(src_node_list):
				#distance = node_cost_list_x2[n][src_node_index] + edge_distance_list_x3[n][src_node_index][dest_node_index][0]
				distance = node_cost_list_x2[n][src_node_index] + tps2.get_scalar_distance(edge_distance_list_x3[n][src_node_index][dest_node_index])
				if  (min_distance - small_value) <= distance <= (min_distance + small_value):
					cand_list.append(src_node_index)
				elif distance < min_distance + small_value:
					cand_list = [src_node_index]
					min_distance = distance
			back_link_list_x2.append(cand_list)
			node_cost_list.append(min_distance)
		back_link_list_x3.append(back_link_list_x2)
		node_cost_list_x2.append(node_cost_list)
	return back_link_list_x3

#  returns: [layer index: [node index: prob]]
def get_node_probability_list_x2(back_link_list_x3):
	node_probability_list_x2 = [[0 for node_index in back_link_list_x2] for back_link_list_x2 in back_link_list_x3]
	node_path_count_list_x2 = [[0 for node_index in back_link_list_x2] for back_link_list_x2 in back_link_list_x3] # [layer index: [node index: path count]]
	if True:
		if True:
			node_path_count_list_x2[-1][0] = 1
		for n0 in range(len(back_link_list_x3) - 1):
			n = len(back_link_list_x3) - 2 - n0
			back_link_list_x2 = back_link_list_x3[n + 1] # dest → src のリンクだよ
			for dest_node_index, back_link_list in enumerate(back_link_list_x2):
				path_count = node_path_count_list_x2[n + 1][dest_node_index]
				for src_node_index in back_link_list:
					node_path_count_list_x2[n][src_node_index] += path_count
	if True:
		if True:
			path_sum = sum([node_probability for node_probability in node_path_count_list_x2[0]])
			for node_index, path_count in enumerate(node_path_count_list_x2[0]):
				node_probability_list_x2[0][node_index] = path_count / path_sum
		for n in range(len(back_link_list_x3) - 1):
			back_link_list_x2 = back_link_list_x3[n + 1] # dest → src
			for src_node_index in range(len(node_path_count_list_x2[n])):
				path_count_sum = sum([node_path_count_list_x2[n + 1][dest_node_index] for dest_node_index, src_node_index_list in enumerate(back_link_list_x2) if src_node_index in src_node_index_list])
				if path_count_sum > 0:
					for dest_node_index, back_link_list in enumerate(back_link_list_x2):
						if src_node_index in back_link_list:
							node_probability_list_x2[n + 1][dest_node_index] += node_probability_list_x2[n][src_node_index] * (node_path_count_list_x2[n + 1][dest_node_index] / path_count_sum)
	return node_probability_list_x2

def make_computation_graph(graph, tps2, tensor_param):
	node_list_x2 = graph[0]
	edge_distance_list_x3 = graph[1]
	tensor_edge_list = []
	tensor_node_list = torch.tensor([1.0 / len(node_list_x2[0]) for elm in node_list_x2[0]])
	for n in range(len(node_list_x2) - 2):
		src_node_list = node_list_x2[n]
		dest_node_list = node_list_x2[n + 1]
		tensor_edge = torch.tensor(numpy.zeros((len(src_node_list), len(dest_node_list))))
		for src_node_index, src_node in enumerate(src_node_list):
			for dest_node_index, dest_node in enumerate(dest_node_list):
				tensor_edge[src_node_index][dest_node_index] = -tps2.get_scalar_distance_tensor(edge_distance_list_x3[n][src_node_index][dest_node_index], tensor_param)
		tensor_edge = torch.exp(tensor_edge)
		if denom_mode == 'hmm':
			tensor_Z = tensor_edge.sum(1)
			tensor_edge = (tensor_edge.t() / tensor_Z).t()
		else:
			tensor_Z = torch.sum(tensor_node_list * tensor_edge.sum(1))
			tensor_edge = tensor_edge / tensor_Z
		tensor_edge_list.append(tensor_edge)
		prev_tensor_node_list = tensor_node_list
		tensor_node_list = torch.tensor([0.0 for elm in node_list_x2[n + 1]])
		for dest_node_index, dest_node in enumerate(dest_node_list):
			tensor_node_list[dest_node_index] = sum([prev_tensor_node_list[src_node_index] * tensor_edge[src_node_index][dest_node_index] for src_node_index in range(len(src_node_list))])
	return tensor_edge_list

def get_accuracy(arg_list):
	(tps2, answer_tpl_list, interpretation_list_x2, graph) = arg_list
	#graph = make_interpretation_graph(tps2, interpretation_list_x2)
	node_list_x2 = graph[0]
	back_link_list_x3 = get_back_link_list_x3(tps2, graph)
	shortest_path_count = 0 # 未実装
	#print(back_link_list_x3)
	node_probability_list_x2 = get_node_probability_list_x2(back_link_list_x3)
	#print(node_probability_list_x2)
	answer_index_list = []
	for n, tpl in enumerate(answer_tpl_list):
		node_list = node_list_x2[n]
		answer_index = -1
		for i, node in enumerate(node_list):
			if tpl[0] == node[0] and tpl[1] == node[1] and tpl[2] == node[2]:
				answer_index = i
				break
		if answer_index < 0:
			print('error')
		answer_index_list.append(answer_index)
	correct_count = 0.0
	for n, tpl in enumerate(answer_tpl_list):
		correct_count += node_probability_list_x2[n][answer_index_list[n]]
	return correct_count / len(answer_tpl_list)

def get_average_accuracy(tps2, tpl_list_x2, int_list_x3, graph_list, parallel_count=8):
	acc_list = []
	#shortest_path_count_list = []
	file_count = len(tpl_list_x2)
	with Pool(parallel_count) as p:
		acc_list = p.map(func=get_accuracy, iterable=zip([tps2] * len(tpl_list_x2), tpl_list_x2, int_list_x3, graph_list))
	mean = st.mean(acc_list)
	stdev = st.stdev(acc_list)
	#mean_shortest_path_count = st.mean(shortest_path_count_list)
	#print('\rrget_average_accuracy mean:', mean, ', stdev:', stdev, '          ')
	#print('\r')
	return (mean, stdev)

def calc_gradient(graph, answer_tpl_list, tensor_param, tensor_edge_list):
	node_list_x2 = graph[0]
	edge_distance_list_x3 = graph[1]
	answer_index_list = []
	for n, tpl in enumerate(answer_tpl_list):
		node_list = node_list_x2[n]
		answer_index = -1
		for i, node in enumerate(node_list):
			if tpl[0] == node[0] and tpl[1] == node[1] and tpl[2] == node[2]:
				answer_index = i
				break
		if answer_index < 0:
			print('error')
		answer_index_list.append(answer_index)
	tensor_nl_prob = torch.tensor(0.0)
	for n in range(len(answer_index_list) - 1):
		edge_tensor = tensor_edge_list[n]
		tensor_nl_prob -= torch.log(edge_tensor[answer_index_list[n]][answer_index_list[n + 1]])
		#tensor_nl_prob = tensor_nl_prob - torch.log(edge_tensor[answer_index_list[n]][answer_index_list[n + 1]])
	tensor_nl_prob.backward()
	return tensor_nl_prob

def train(data_dir, max_epoch, batch_size, tps2, tensor_param, optimizer, max_length=100, b_refresh_graph=False, test_set_interval=10, wait_epoch=5, parallel_count=8):
	ret_str = ''
	# prepare tpl_list
	tpl_list_x2 = []
	for i, filename in enumerate(sorted(os.listdir(data_dir))):
		print('\rload データ:', (i + 1), end='')
		tpl_list = load(os.path.join(data_dir, filename))
		for i2 in range((len(tpl_list) // max_length) + 1):
			if len(tpl_list[i2:i2+max_length]) > 1:
				tpl_list_x2.append(tpl_list[i2:i2+max_length])
			else:
				pass
		#if i < 20: print("\r", i, filename)
	print('\rload finished ', len(tpl_list_x2), '                           ')
	# prepare int_list_x3
	int_list_x3 = []
	for i, tpl_list in enumerate(tpl_list_x2):
		print('\rget_chord_interpretation_list_x2 data:', (i + 1), '/', len(tpl_list_x2), end='')
		int_list_x3.append(get_chord_interpretation_list_x2(tpl_list))
	print('\rget_chord_interpretation_list_x2 finished                       ')
	# prepare graph_list
	graph_list = []
	zipped = list(zip([tps2] * len(int_list_x3), int_list_x3))
	with Pool(parallel_count) as p:
		(graph_list) = p.map(func=make_interpretation_graph, iterable=zipped)
	print('\rmake_interpretation_graph finished                              ')
	# prepare trainining/validation/test sets
	if test_set_interval > 0:
		validation_tpl_list_x2 = [v for i, v in enumerate(tpl_list_x2) if i % test_set_interval == 0]
		test_tpl_list_x2 = [v for i, v in enumerate(tpl_list_x2) if i % test_set_interval == 1]
		training_tpl_list_x2 = [v for i, v in enumerate(tpl_list_x2) if i % test_set_interval >= 2]
		validation_int_list_x3 = [v for i, v in enumerate(int_list_x3) if i % test_set_interval == 0]
		test_int_list_x3 = [v for i, v in enumerate(int_list_x3) if i % test_set_interval == 1]
		training_int_list_x3 = [v for i, v in enumerate(int_list_x3) if i % test_set_interval >= 2]
		validation_graph_list = [v for i, v in enumerate(graph_list) if i % test_set_interval == 0]
		test_graph_list = [v for i, v in enumerate(graph_list) if i % test_set_interval == 1]
		training_graph_list = [v for i, v in enumerate(graph_list) if i % test_set_interval >= 2]
	else:
		validation_tpl_list_x2 = tpl_list_x2
		test_tpl_list_x2 = tpl_list_x2
		training_tpl_list_x2 = tpl_list_x2
		validation_int_list_x3 = int_list_x3
		test_int_list_x3 = int_list_x3
		training_int_list_x3 = int_list_x3
		validation_graph_list = graph_list
		test_graph_list = graph_list
		training_graph_list = graph_list
	# first accuracy
	print('param: ', tensor_param)
	tps2.update_params(tensor_param)
	acc0 = get_average_accuracy(tps2, training_tpl_list_x2, training_int_list_x3, training_graph_list, parallel_count=parallel_count)
	acc1 = get_average_accuracy(tps2, validation_tpl_list_x2, validation_int_list_x3, validation_graph_list, parallel_count=parallel_count)
	acc2 = get_average_accuracy(tps2, test_tpl_list_x2, test_int_list_x3, test_graph_list, parallel_count=parallel_count)
	print('\rget_average_accuracy training mean:', acc0[0], ', stdev:', acc0[1], ', validation mean:', acc1[0], ', stdev:', acc1[1], ', test mean:', acc2[0], ', stdev:', acc2[1], '          ')
	max_acc_mean = 0.0
	max_acc_test_mean = 0.0
	max_acc_epoch = 0
	max_acc_tensor = 0
	# training
	for epoch in range(max_epoch):
		# finished?
		if epoch > max_acc_epoch + wait_epoch:
			ret_str += "\nmax mean accuracy: " + str(max_acc_test_mean) + " at epoch " + str(max_acc_epoch + 1) + "\n" + str(max_acc_tensor);
			return ret_str
		optimizer.zero_grad()
		nl_prob = 0
		zipped = list(zip([tps2] * len(training_tpl_list_x2), training_tpl_list_x2, training_int_list_x3, training_graph_list, [tensor_param] * len(training_tpl_list_x2)))
		random.shuffle(zipped)
		for i0 in range(math.ceil(len(training_tpl_list_x2) / batch_size)):
			with Pool(parallel_count) as p:
				(result_list) = p.map(func=train2, iterable=zipped[i0:i0 + batch_size])
			(nl_prob_list, grad_list) = zip(*result_list)
			tensor_param.grad = sum(grad_list)
			nl_prob = sum(nl_prob_list) / len(training_int_list_x3)
			optimizer.step()
			print('\repoch:', (epoch + 1), '-', ((i0 + 1) * batch_size), 'param:', tensor_param, ', grad:', tensor_param.grad, ', average nl_prob:', nl_prob, '                                               ')
			optimizer.zero_grad()
			tps2.update_params(tensor_param)
			if True:
				if b_refresh_graph:
					print('error: not implemented')
				acc0 = get_average_accuracy(tps2, training_tpl_list_x2, training_int_list_x3, training_graph_list, parallel_count=parallel_count)
				acc1 = get_average_accuracy(tps2, validation_tpl_list_x2, validation_int_list_x3, validation_graph_list, parallel_count=parallel_count)
				acc2 = get_average_accuracy(tps2, test_tpl_list_x2, test_int_list_x3, test_graph_list, parallel_count=parallel_count)
				print('\rget_average_accuracy training mean:', acc0[0], ', stdev:', acc0[1], ', validation mean:', acc1[0], ', stdev:', acc1[1], ', test mean:', acc2[0], ', stdev:', acc2[1], '          ')
				ret_str += '\nepoch:' + str(epoch + 1) + '-' + str((i0 + 1) * batch_size) + '  training mean:' + str(acc0[0]) + ', stdev:' + str(acc0[1]) + ', validation mean:' + str(acc1[0]) + ', stdev:' + str(acc1[1]) + ', test mean:' + str(acc2[0]) + ', stdev:' + str(acc2[1])
				if max_acc_mean < acc1[0]:
					max_acc_mean = acc1[0]
					max_acc_test_mean = acc2[0]
					max_acc_epoch = epoch
					max_acc_tensor = tensor_param.clone()

def train2(arg_list):
	(tps2, tpl_list, int_list_x2, graph, tensor_param) = arg_list
	tensor_edge_list = make_computation_graph(graph, tps2, tensor_param)
	nl_prob = calc_gradient(graph, tpl_list, tensor_param, tensor_edge_list).item()
	return nl_prob, tensor_param.grad

def debug_show_shortest_paths(answer_tpl_list, graph, back_link_list_x3):
	node_list_x2 = graph[0]
	edge_distance_list_x3 = graph[1]
	correct_count = 0.0
	node_probability_list_x2 = get_node_probability_list_x2(back_link_list_x3)
	for n, answer_tpl in enumerate(answer_tpl_list):
		answer_tpl2 = (answer_tpl[0], True if answer_tpl[1] == setting.SCALE_MAJOR else False, answer_tpl[2]) # scale → b_major
		print('chord', n, ':', debug_tpl(answer_tpl2))
		node_list = node_list_x2[n]
		back_link_list_x2 = back_link_list_x3[n]
		node_probability_list = node_probability_list_x2[n]
		for n2, node in enumerate(node_list):
			node2 = (node[0], True if node[1] == setting.SCALE_MAJOR else False, node[2]) # scale → b_major
			back_link_list = back_link_list_x2[n2]
			node_probability = node_probability_list[n2]
			print('　interpretation ', n2, ':', debug_tpl(node2), 'prob:', node_probability, 'prev: ', end='')
			if answer_tpl2 == node2:
				correct_count += node_probability / len(answer_tpl_list)
			for back_link in back_link_list:
				print(back_link, end=' ')
			print('')
		print('　　acc:', (correct_count * len(answer_tpl_list) / (n + 1)))
	back_link_list = back_link_list_x3[-1][0]
	print('from end node: ', end='')
	for back_link in back_link_list:
		print(back_link, end=' ')
	print('')

def debug_tpl(tpl):
	return '(key:' + str(tpl[0]) + ', ' + ('maj' if tpl[1] else 'min') + ', degree:' + str(tpl[2]) + ')'
