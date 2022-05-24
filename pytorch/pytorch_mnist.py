# 라이브러리 불러오기
import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # 최적화기 관련 라이브러리
from torchvision import (
    datasets,
    transforms,
)  # datasets : 다양한 데이터셋 제공, transforms : 데이터 전처리 등 수행

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 파라미터 설정
batch_size = 128  # 배치 사이즈 설정 -> 128 (한번 학습시 1284개의 데이터를 통해 학습)
# Batch Size를 한 번씩 한 번씩 연산을 해서 전체 데이터 한 번을 다 학습한게 epoch
# 모든 데이터에 대해 10번 학습 수행
num_epochs = 10

# 학습률(학습속도결정 -> 너무 값이 작으면 학습 속도가 느림, 값이 크면 최적으로 학습하지 못함)
learning_rate = 0.00025  # 학습률

# MNIST 데이터 다운로드
trn_dataset = datasets.MNIST(
    "./mnist_data/",  # 다운로드 경로
    download=True,  # True인 경우 데이터 셋을 다운 받음, 이미 다운받아진 경우 다운을 진행하지 않음
    train=True,  # True인 경우 학습 데이터로 사용, False인 경우 학습에 사용하지 않음
    transform=transforms.Compose(
        [transforms.ToTensor()]
    ),  # Pytorch 텐서(Tensor)의 형태로 데이터 출력 -> pytorch 연산의 경우 tensor의 형태로 연산 진행
)
val_dataset = datasets.MNIST(
    "./mnist_data/",
    download=False,
    train=False,
    transform=transforms.Compose([transforms.ToTensor()]),
)

# DataLoader 설정
# DataLoader를 이용해 학습 및 검증 데이터를 섞어준 후 미니 배치로 나눠줌
trn_loader = torch.utils.data.DataLoader(
    trn_dataset, batch_size=batch_size, shuffle=True  # 배치 사이즈 , 데이터를 epoch마다 섞어줄지 결정
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True
)

# CNN 네트워크
class CNNClassifier(nn.Module):
    # CNN 네트워크 변수 정의

    def __init__(self):
        super(CNNClassifier, self).__init__()
        # nn.Conv2d : convolution 연산 수행
        self.conv1 = nn.Conv2d(1, 16, 3, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)

        # nn.Linear : 선형 연산 수행
        # 64*4*4 -> 64채널수, 3번째 convolution 연산까지 수행한 후 feature map의 너비, 높이
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

        self.drop = nn.Dropout(p=0.2)

    # 네트워크 구조 결정 및 연산 수행
    def forward(self, x):
        # 3번의 Convolution 연산 수행 (input batch x 1 x 28 x 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 데이터 구조 변환 (batcv x 64 x 4 x 4) -> (batch x 64*4*4)

        x = F.relu(self.fc1(x))  # 선형 연산 수행 후 비선형함수 (relu) 통과
        x = self.drop(x)  # dropout 수행(overfitting 방지)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)  # 선형 연산 수행
        return x  # 최종 결과 반환


# 정확도를 도출하는 함수
# 입력 y(네트워크 연산 결과), label(실제 결과)
# Label은 현재 입력이 어떤 숫자인지의 값을 보여줌(batch size x 1) - (EX, 정답이 4이면 4, 7이면 7 이런식으로 도출을 해주는 것)
# Y는 각 숫자에 대한 확률을 나타냄
# 하나의 입력에 대해 10개의 값을 가짐 (batch size x 10)
def get_accuracy(y, label):
    # argmax 연산을 통해 확률 중 가장 큰 값의 인덱스를 반환하여 label과 동일한 형식으로 변환
    y_idx = torch.argmax(y, dim=1)
    result = y_idx - label

    num_correct = 0
    for i in range(len(result)):
        if result[i] == 0:
            num_correct += 1

    return num_correct / y.shape[0]  # 모든 입력에 대해 정답을 맞춘 개수를 전체 개수로 나눠주어 정확도를 반환


# 네트워크, 손실함수, 최적화기 선언
cnn = CNNClassifier().to(device)  # 연산잔치device 설정
criterion = nn.CrossEntropyLoss()  # 손실함수
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)  # 최적화기 Adam Optimizer

num_batches = len(trn_loader)  # 한 Epoch동안 몇 batch가 있느냐

# 학습 수행, 검증 및 삭습 진행상황 출력
for epoch in range(num_epochs):  # 전체 epoch 반복문
    # 학습시 손실함수 값과 정확도를 기록하기 위한 리스트
    trn_loss_list = []
    trn_acc_list = []
    for i, data in enumerate(
        trn_loader
    ):  # 1 epoch연산을 위한 반복문(여기서 data는 각 배치로 나누어진 데이터와 정답을 포함)
        # 데이터 처리
        cnn.train()  # cnn 네트워크를 학습을 위한 모드로 설정

        x, label = data  # 학습 데이터(x: 입력, label: 정답)를 받아온 후 device 할당
        x = x.to(device)
        label = label.to(device)

        # 네트워크 연산 및 손실함수 계신
        model_output = cnn(x)  # 네트워크 연산 수행후 출력값 도출(입력 x, 출력 model_output)
        loss = criterion(model_output, label)  # 손실함수 값 계산

        # 네트워크 업데이트(Pytorchㄹ를 이용한 딥러닝 학습에 있어서는 이 3개의 과정을 반드시 거쳐야 한다.)
        optimizer.zero_grad()  # 학습 수행 전 미분값을 0으로 초기화
        loss.backward()  # 네트워크 변수(가중치 W, 편향 b)에 대한 기울기 계산
        optimizer.step()  # 네트워크 변수 업데이트(adam optimizer를 최적화기로 사용)

        # 학습 정확도 및 손실함수 값 기록
        # 네트워크의 연산 결과와 실제 정답 결과를 비교하여 정확도를 도출
        trn_acc = get_accuracy(model_output, label)

        # 손실함수 값을 trn_loss_list에 추가(item: 하나의 값으로 된 tensor를 일반 값으로 바꿔줌)
        trn_loss_list.append(loss.item())
        trn_acc_list.append(trn_acc)  # 정확도 값을 trn_acc_list에 추가

        # 학습 진행 상황 출력 및 검증셋 연산 수행
        if (i + 1) % 100 == 0:  # 매 100번째 미니배치 연산마다 진행상황 출력
            cnn.eval()  # 네트워크를 검증 모드로 설정
            with torch.no_grad():  # 학습에 사용하지 않는 코드들은 해당 블록 내에 기입
                val_loss_list = []  # 검증시 손실함수 값과 정확도를 저장하기 위한 리스트
                val_acc_list = []

                # 검증 셋에 대한 연산 수행
                for j, val in enumerate(val_loader):
                    val_x, val_label = val

                    val_x = val_x.to(device)
                    val_label = val_label.to(device)

                    val_output = cnn(val_x)

                    val_loss = criterion(val_output, val_label)
                    val_acc = get_accuracy(val_output, val_label)

                    val_loss_list.append(val_loss.item())
                    val_acc_list.append(val_acc)

            # 학습과 검증 과정에 대한 진행상황 출력
            print(
                "epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | trn acc: {:.4f} | val acc: {:.4f}".format(
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    num_batches,
                    np.mean(trn_loss_list),
                    np.mean(val_loss_list),
                    np.mean(trn_acc_list),
                    np.mean(val_acc_list),
                )
            )
