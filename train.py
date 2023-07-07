import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import argparse
import logging
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--epoch', type=int, default=100, help='num of epochs to train')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--profiling', action="store_true", default=False, help="profile one batch")
    args = parser.parse_args()

    # 定义VGG模型
    VGG = models.vgg16(pretrained=False)

    # 设置训练参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = args.epoch
    batch_size = args.b
    learning_rate = args.lr

    # 加载CIFAR-10数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建VGG模型实例
    model = VGG.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_epoch_time = time.time() - epoch_start_time
        one_batch_time = time.time() - epoch_start_time
        logging.info("epoch = {epoch}, iteration = {i}, trained_samples = {1}, total_samples = {1}, loss = {loss}, lr = {lr}, current_epoch_wall-clock_time = {current_epoch_time}")
        if args.profiling:
            logging.info(f"PROFILING: dataset total number {len(train_loader.dataset)}, training one batch costs {one_batch_time} seconds")
            break

