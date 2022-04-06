import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import torchvision


amp_enabled = True


def train(images, targets, device, scaler, amp_enabled):
    images, targets = images.to(device), targets.to(device)
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(amp_enabled):
        outputs = model(images)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


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
scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

with torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=True,
                            with_flops=True) as profiler:
    for step, (images, targets) in enumerate(trainloader):
        if step >= (1 + 1 + 3) * 2:
            break
        train(images, targets, device, scaler, amp_enabled)
        profiler.step()
