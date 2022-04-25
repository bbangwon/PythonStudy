import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
	#Init 함수 : 네트워크의 변수 정의
    # nn.Linear: 선형 연산(Wx + b)
	def __init__(self):
		super(Model, self).__init__()
		self.fc1 = nn.Linear(3,4)
		self.fc2 = nn.Linear(4,4)
		self.fc3 = nn.Linear(4,2)

	#Foward 함수 : 네트워크 구조 결정 및 연산 수행
	def forward(self, x):
		# 선형연산 수행 후 비선형 함수 통과
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		# 선형 연산만 수행
		x = self.fc3(x)
		# 결과값 반환
		return x	

# 네트워크 선언
model = Model().to(device)

#네트워크를 이용한 연산 수행 (tensor)
#tensor  : pytorch 연산 수행을 위한 데이터 자료형 (numpy <-> tensor 변환 중요)
x_tensor = torch.rand(2,3).to(device)
# 딥러닝 모델에 입력 tensor를 대입하여 결과값 도출 -> model의 forward 함수 연산
output = model(x_tensor)

print("x_tensor: {}".format(x_tensor))
print("output: {}".format(output))

# Numpy 변환
# 연산장치를 CPU로 변환해서 Gradient 전파가 안 되도록 함
# 출력이 다시 numpy로 바꾸려면 학습에 이용되지 않는 Gradient를 제외해야 되기 때문
output_np = output.cpu().detach().numpy()
print("output_np: {}".format(output_np))

#네트워크를 이용한 연산 수행 (np -> tensor)
# numpy array 형태로 랜덤값 도출
x_np = np.random.rand(2,3)
# numpy array를 FloatTensor로 변환하고 연산 장치 할당
x_tensor = torch.FloatTensor(x_np).to(device)
#해당 tensor를 입력으로 하여 결과 도출
output_tensor = model(x_tensor)

print('-------------')
print("x_np: {}".format(x_np))
print("x_tensor: {}".format(x_tensor))
print("output_tensor: {}".format(output_tensor))

