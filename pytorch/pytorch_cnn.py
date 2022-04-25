import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    # Init 함수 : 네트워크 변수 정의
    # nn.Conv2d: 합성곱 여산
    def __init__(self):
        super(Model, self).__init__()
        # In channel: 3, out channel : 32, kernel size=3, stride=2
        self.conv1 = nn.Conv2d(3, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, 2)
        # Linear -> Convolution
        # 4x4x32 벡터로 변환 후 10개의 결과 출력
        self.fc1 = nn.Linear(32*4*4, 10)

    # Forward 함수 : 네트워크 구조 결정 및 연산 수행
    def forward(self, x):
        #합성곱 연산 수행 후 비선형 함수 통과
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # 4x4x32 -> 4*4*32 벡터로 변환
        x = x.view(x.size(0), -1)
        #선형 연산 수행
        x = self.fc1(x)
        # 결과값 반환
        return x

# 네트워크 선언
# Model 클래스를 model로 정의 후 연산 장치 할당
model = Model().to(device)

# 네트워크를 이용한 연산 수행(np -> tensor)
# (1,3,20,20)의 랜덤 데이터를 torch로 생성 후 연산 장치 할당 (3 channel, 20x20 너비 높이)
x_np = np.random.rand(1,3,20,20)
# numpy array를 tensor로 변환하고 연산 장치 할당
x_tensor = torch.FloatTensor(x_np).to(device)
# 딥려닝 모델(model)에 입력 x를 대입하여 결과값(output) 도출 -> model의 forward 함수 연산
output_tensor = model(x_tensor)

print('------------------')
print("output_tensor: {}".format(output_tensor))