import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import torchvision


def train(images, targets, device):
    images, targets = images.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet18(pretrained=True).to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

with torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=True,
                            with_flops=True) as profiler:
    for step, (images, targets) in enumerate(trainloader):
        if step >= (1 + 1 + 3) * 2:
            break
        train(images, targets, device)
        profiler.step()
