from tps import TPS
from tps2 import TPS2
import setting
import train
import os
import math
import re
import statistics as st
import torch

if __name__ == '__main__':
	
	def exp_d(title, tps2, tensor_param):
		data_dir = './data'
		out_file = 'output.txt'
		#op = torch.optim.SGD([tensor_param], lr=0.0003, momentum=0.0)
		op = torch.optim.SGD([tensor_param], lr=0.001, momentum=0.0)
		torch.set_printoptions(edgeitems=tensor_param.size()[0])
		#ret_str = train.train(data_dir, 200, 100, tps2, tensor_param, op, max_length=50, wait_epoch=10, parallel_count=4)
		ret_str = train.train(data_dir, 200, 100, tps2, tensor_param, op, max_length=50, wait_epoch=2, parallel_count=4)
		#print(title + '\n' + ret_str + '\n')
		with open(out_file, mode='a') as f:
			f.write('\n\n' + title + '\n' + ret_str + '\n')

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(2)], requires_grad=True) # 2 params
	exp_d('exp : qDE 25 (DE 4.1)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(4)], requires_grad=True) # 4 params
	exp_d('exp : qDE 20 (DE 4.2)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(7)], requires_grad=True) # 7 params
	exp_d('exp : qDE 13 (DE 5.1)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(7)], requires_grad=True) # 7 params
	exp_d('exp : qDE 10 (DE 5.2)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(12)], requires_grad=True) # 12 params
	exp_d('exp : qDE 14 (DE 5.3)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(12)], requires_grad=True) # 12 params
	exp_d('exp : qDE 12 (DE 5.4)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(2 * 7)], requires_grad=True) # 14 params
	exp_d('exp : qDE 18 (DE 6.1)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(2 * 12 * 2)], requires_grad=True) # 48 params
	exp_d('exp : qDE 3 (DE 6.2)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(7 * 7)], requires_grad=True) # 49 params
	exp_d('exp : qDE 15 (DE 7.1)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(7 * 12)], requires_grad=True) # 84 params
	exp_d('exp : qDE 8 (DE 7.2)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(2 * 7 * 7 * 7)], requires_grad=True) # 686 params
	exp_d('exp : qDE 26 (DE 8.1)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(2 * 7 * 2 * 12 * 7)], requires_grad=True) # 2352 params
	exp_d('exp : qDE 4 (DE 8.2)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(2 + 7)], requires_grad=True) # 9 params
	exp_d('exp : qDE 13, 25 (DE 4.1, 5.1)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(4 + 7)], requires_grad=True) # 11 params
	exp_d('exp : qDE 13, 20 (DE 4.2, 5.1)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(7 + 49)], requires_grad=True) # 56 params
	exp_d('exp : qDE 13, 15 (DE 5.1, 7.1)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(2 + 7 + 49)], requires_grad=True) # 58 params
	exp_d('exp : qDE 13, 15, 25 (DE 4.1, 5.1, 7.1)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(4 + 7 + 49)], requires_grad=True) # 60 params
	exp_d('exp : qDE 13, 15, 20 (DE 4.2, 5.1, 7.1)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(14 + 49)], requires_grad=True) # 63 params
	exp_d('exp : qDE 15, 18 (DE 6.1, 7.1)', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 1, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(2 * 12 * 2 + 7 * 12)], requires_grad=True) # 132 params
	exp_d('exp : qDE 3, 8 (DE 6.1, 7.2)', tps2, tensor_param)

	tps2 = TPS2([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(2 * 7)], requires_grad=True) # 14 params
	exp_d('exp : qDE 1, 2, 18 (DE 2, 3, 6.1)', tps2, tensor_param)

	tps2 = TPS2([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(7 * 7)], requires_grad=True) # 49 params
	exp_d('exp : qDE 0, 2, 15 (DE 1, 3, 7.1)', tps2, tensor_param)

	tps2 = TPS2([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(14 + 49)], requires_grad=True) # 63 params
	exp_d('exp : qDE 2, 15, 18 (DE 3, 6.1, 7.1)', tps2, tensor_param)
	
	'''
	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(3)], requires_grad=True) # 3 params
	exp_d('exp : qDE 19', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(3 * 7)], requires_grad=True) # 21 params
	exp_d('exp : qDE 17', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(2 * 7 * 2)], requires_grad=True) # 28 params
	exp_d('exp : qDE 11', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(4)], requires_grad=True) # 4 params
	exp_d('exp : qDE 21', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(7)], requires_grad=True) # 7 params
	exp_d('exp : qDE 27', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(3 * 7 * 7 * 7)], requires_grad=True) # 1029 params
	exp_d('exp : qDE 24', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0])
	tensor_param = torch.tensor([0.0 for _ in range(7 + 3)], requires_grad=True) # 10 params
	exp_d('exp : qDE 13, 19', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
	tensor_param = torch.tensor([0.0 for _ in range(7 + 3 + 7 * 7)], requires_grad=True) # 59 params
	exp_d('exp : qDE 13, 15, 19', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(21 + 49)], requires_grad=True) # 70 params
	exp_d('exp : qDE 15, 17', tps2, tensor_param)

	tps2 = TPS2([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0])
	tensor_param = torch.tensor([0.0 for _ in range(7 + 3)], requires_grad=True) # 10 params
	exp_d('exp : qDE 1, 2, 13, 19', tps2, tensor_param)

	tps2 = TPS2([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(3 * 7)], requires_grad=True) # 21 params
	exp_d('exp : qDE 1, 2, 17', tps2, tensor_param)

	tps2 = TPS2([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
	tensor_param = torch.tensor([0.0 for _ in range(7 + 3 + 7 * 7)], requires_grad=True) # 59 params
	exp_d('exp : qDE 2, 13, 15, 19', tps2, tensor_param)

	tps2 = TPS2([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(21 + 49)], requires_grad=True) # 70 params
	exp_d('exp : qDE 2, 15, 17', tps2, tensor_param)

	tps2 = TPS2([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(2 * 12 * 2 + 7 * 7)], requires_grad=True) # 97 params
	exp_d('exp : qDE 3, 15', tps2, tensor_param)

	tps2 = TPS2([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	tensor_param = torch.tensor([0.0 for _ in range(3 * 7 * 7 * 7)], requires_grad=True) # 1029 params
	exp_d('exp : qDE 2, 24', tps2, tensor_param)

	'''
