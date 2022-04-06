import argparse

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import torchvision
import tqdm


def train(images, targets, device, scaler, amp_enabled):
    images, targets = images.to(device), targets.to(device)
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(amp_enabled):
        outputs = model(images)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--amp', type=bool, default=False, help='Specify whether to use an amp.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.resnet18(pretrained=True).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    tensorboard_logdir = 'logs/resnet18'
    if args.amp:
        tensorboard_logdir += '_amp'
    if args.batch_size != 128:
        tensorboard_logdir += f'_{args.batch_size}bs'
    if args.num_workers != 0:
        tensorboard_logdir += f'_{args.num_workers}workers'

    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_logdir),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
    ) as profiler:
        for step, (images, targets) in enumerate(trainloader):
            if step >= (2 + 2 + 5) * 1:
                break
            train(images, targets, device, scaler, args.amp)
            profiler.step()
