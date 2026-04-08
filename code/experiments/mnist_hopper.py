"""MNIST generator tuning experiment for Hopper (multi-rep, GPU).
Usage: python3 -u mnist_hopper.py --rep 1
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import pandas as pd
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--rep', type=int, required=True)
args = parser.parse_args()

seed = args.rep * 31 + 7
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}  Rep: {args.rep}  Seed: {seed}")

os.makedirs("results", exist_ok=True)

# --- Config ---
batch_size = 128
epochs = 5
lr_generator = 1e-3
lambda_range = [1e-5, 1e-1]
n_wbb_dim = 16

# --- Index-aware dataset wrapper ---
class IndexedSubset(Dataset):
    def __init__(self, subset):
        self.subset = subset
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        data, label = self.subset[idx]
        return data, label, idx

# --- Models ---
class TargetMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=64, num_classes=10):
        super().__init__()
        self.input_size = input_size
        self.param_shapes = {
            'w1': (hidden_size, input_size), 'b1': (hidden_size,),
            'w2': (num_classes, hidden_size), 'b2': (num_classes,)
        }
        self.total_params = sum(np.prod(s) for s in self.param_shapes.values())

    def forward(self, x, weights):
        idx = 0
        parts = {}
        for name, shape in self.param_shapes.items():
            size = int(np.prod(shape))
            parts[name] = weights[idx:idx+size].view(shape)
            idx += size
        x = x.view(-1, self.input_size)
        x = torch.relu(torch.mm(x, parts['w1'].t()) + parts['b1'])
        x = torch.mm(x, parts['w2'].t()) + parts['b2']
        return x

class HyperGenerator(nn.Module):
    def __init__(self, target_params_size, n_omega=16, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + n_omega, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, target_params_size)
        )
    def forward(self, x):
        return self.net(x)

def sample_block_weights(n_blocks, device):
    alpha = torch.ones(n_blocks, device=device)
    w = torch.distributions.Dirichlet(alpha).sample()
    return w * n_blocks

# --- Data ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

n_train = 50000
indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(42))
train_subset = Subset(full_train, indices[:n_train])
val_dataset = Subset(full_train, indices[n_train:60000])

train_indexed = IndexedSubset(train_subset)
train_loader = DataLoader(train_indexed, batch_size=batch_size, shuffle=True)
train_loader_plain = DataLoader(Subset(full_train, indices[:n_train]),
                                 batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Stable block assignment by observation index
block_assignments = torch.randint(0, n_wbb_dim, (n_train,), device=device)

# --- Train generator ---
target = TargetMLP().to(device)
generator = HyperGenerator(target.total_params, n_omega=n_wbb_dim).to(device)
optimizer = optim.Adam(generator.parameters(), lr=lr_generator)
criterion = nn.CrossEntropyLoss(reduction='none')

min_log = np.log10(lambda_range[0])
max_log = np.log10(lambda_range[1])

print("Training generator...")
t0 = time.time()
for epoch in range(epochs):
    generator.train()
    for batch_idx, (data, target_labels, obs_indices) in enumerate(train_loader):
        data, target_labels = data.to(device), target_labels.to(device)
        obs_indices = obs_indices.to(device)

        log_l = np.random.uniform(min_log, max_log)
        l_val = 10**log_l
        omega_blocks = sample_block_weights(n_wbb_dim, device)
        # Stable block lookup by observation index
        batch_block_ids = block_assignments[obs_indices]
        omega_obs = omega_blocks[batch_block_ids]

        log_l_norm = (log_l - min_log) / (max_log - min_log) * 2 - 1
        gen_input = torch.cat([
            torch.tensor([log_l_norm], dtype=torch.float32, device=device),
            omega_blocks
        ]).unsqueeze(0)

        optimizer.zero_grad()
        weights = generator(gen_input).squeeze(0)
        outputs = target(data, weights)
        per_sample_loss = criterion(outputs, target_labels)
        data_loss = (omega_obs * per_sample_loss).mean()
        reg_loss = l_val * torch.sum(weights**2)
        loss = data_loss + reg_loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} done")

train_time = time.time() - t0
print(f"Generator training time: {train_time:.1f}s")

# --- Evaluate on validation set ---
generator.eval()
lambda_grid = np.logspace(min_log, max_log, 20)
omega_ones = torch.ones(n_wbb_dim, device=device)
criterion_eval = nn.CrossEntropyLoss()
val_accs = []

t0_eval = time.time()
with torch.no_grad():
    for l_val in lambda_grid:
        log_l = np.log10(l_val)
        log_l_norm = (log_l - min_log) / (max_log - min_log) * 2 - 1
        gen_input = torch.cat([
            torch.tensor([log_l_norm], dtype=torch.float32, device=device),
            omega_ones
        ]).unsqueeze(0)
        weights = generator(gen_input).squeeze(0)
        correct = 0
        for data, target_labels in val_loader:
            data, target_labels = data.to(device), target_labels.to(device)
            pred = target(data, weights).argmax(dim=1, keepdim=True)
            correct += pred.eq(target_labels.view_as(pred)).sum().item()
        val_accs.append(correct / len(val_loader.dataset))
eval_time = time.time() - t0_eval

best_idx = np.argmax(val_accs)
best_lambda = lambda_grid[best_idx]
best_val_acc = val_accs[best_idx]
print(f"Selected lambda={best_lambda:.2e}, val_acc={best_val_acc:.4f}, eval_time={eval_time:.1f}s")

# --- Test accuracy at selected lambda ---
with torch.no_grad():
    log_l = np.log10(best_lambda)
    log_l_norm = (log_l - min_log) / (max_log - min_log) * 2 - 1
    gen_input = torch.cat([
        torch.tensor([log_l_norm], dtype=torch.float32, device=device),
        omega_ones
    ]).unsqueeze(0)
    weights = generator(gen_input).squeeze(0)
    test_correct = 0
    for data, target_labels in test_loader:
        data, target_labels = data.to(device), target_labels.to(device)
        pred = target(data, weights).argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target_labels.view_as(pred)).sum().item()
    test_acc_gen = test_correct / len(test_loader.dataset)

# --- WBB uncertainty ---
M_wbb = 50
wbb_accs = []
with torch.no_grad():
    for m in range(M_wbb):
        omega_m = sample_block_weights(n_wbb_dim, device)
        gen_input = torch.cat([
            torch.tensor([log_l_norm], dtype=torch.float32, device=device),
            omega_m
        ]).unsqueeze(0)
        weights_m = generator(gen_input).squeeze(0)
        correct_m = 0
        for data, target_labels in test_loader:
            data, target_labels = data.to(device), target_labels.to(device)
            pred = target(data, weights_m).argmax(dim=1, keepdim=True)
            correct_m += pred.eq(target_labels.view_as(pred)).sum().item()
        wbb_accs.append(correct_m / len(test_loader.dataset))

wbb_mean = np.mean(wbb_accs)
wbb_std = np.std(wbb_accs)

# --- Baseline ---
class StandardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.view(-1, 784))))

baseline_net = StandardMLP().to(device)
optimizer_bl = optim.Adam(baseline_net.parameters(), lr=1e-3)
criterion_bl = nn.CrossEntropyLoss()
t0_bl = time.time()
for epoch in range(epochs):
    baseline_net.train()
    for data, target_labels in train_loader_plain:
        data, target_labels = data.to(device), target_labels.to(device)
        optimizer_bl.zero_grad()
        loss = criterion_bl(baseline_net(data), target_labels)
        l2_reg = sum(p.pow(2).sum() for p in baseline_net.parameters())
        (loss + best_lambda * l2_reg).backward()
        optimizer_bl.step()
baseline_time = time.time() - t0_bl

baseline_net.eval()
bl_correct = 0
with torch.no_grad():
    for data, target_labels in test_loader:
        data, target_labels = data.to(device), target_labels.to(device)
        pred = baseline_net(data).argmax(dim=1, keepdim=True)
        bl_correct += pred.eq(target_labels.view_as(pred)).sum().item()
baseline_acc = bl_correct / len(test_loader.dataset)

# --- Save ---
res = pd.DataFrame([{
    'rep': args.rep,
    'best_lambda': best_lambda,
    'best_val_acc': best_val_acc,
    'test_acc_gen': test_acc_gen,
    'baseline_acc': baseline_acc,
    'wbb_mean': wbb_mean,
    'wbb_std': wbb_std,
    'train_time': train_time,
    'eval_time': eval_time,
    'baseline_time': baseline_time,
}])
res.to_csv(f'results/mnist_rep{args.rep}.csv', index=False)
print(f"Done. test_acc_gen={test_acc_gen:.4f}, baseline={baseline_acc:.4f}, "
      f"wbb={wbb_mean:.4f}+/-{wbb_std:.4f}")
