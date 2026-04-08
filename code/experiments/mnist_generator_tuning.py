import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- Configuration ---
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 5
lr_generator = 1e-3
lambda_range = [1e-5, 1e-1]
n_wbb_dim = 16  # dimensionality of bootstrap weight input (block weights)

# --- Index-aware dataset wrapper ---
class IndexedSubset(Dataset):
    """Wraps a Subset to also return the within-subset index."""
    def __init__(self, subset):
        self.subset = subset
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        data, label = self.subset[idx]
        return data, label, idx  # idx is stable observation identity

# --- Target Model Definition ---
class TargetMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=64, num_classes=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.param_shapes = {
            'w1': (hidden_size, input_size),
            'b1': (hidden_size,),
            'w2': (num_classes, hidden_size),
            'b2': (num_classes,)
        }
        self.total_params = sum(np.prod(s) for s in self.param_shapes.values())

    def forward(self, x, weights):
        idx = 0
        w1_size = np.prod(self.param_shapes['w1'])
        w1 = weights[idx:idx+w1_size].view(self.param_shapes['w1'])
        idx += w1_size

        b1_size = np.prod(self.param_shapes['b1'])
        b1 = weights[idx:idx+b1_size].view(self.param_shapes['b1'])
        idx += b1_size

        w2_size = np.prod(self.param_shapes['w2'])
        w2 = weights[idx:idx+w2_size].view(self.param_shapes['w2'])
        idx += w2_size

        b2_size = np.prod(self.param_shapes['b2'])
        b2 = weights[idx:idx+b2_size].view(self.param_shapes['b2'])

        x = x.view(-1, self.input_size)
        x = torch.relu(torch.mm(x, w1.t()) + b1)
        x = torch.mm(x, w2.t()) + b2
        return x

# --- Generator Model Definition ---
class HyperGenerator(nn.Module):
    def __init__(self, target_params_size, n_omega=16, hidden_dim=128):
        super().__init__()
        input_dim = 1 + n_omega
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_params_size)
        )

    def forward(self, x):
        return self.net(x)

def sample_block_weights(n_blocks, device):
    """Sample Dirichlet(1,...,1) weights scaled by n_blocks (WBB-style)."""
    alpha = torch.ones(n_blocks, device=device)
    w = torch.distributions.Dirichlet(alpha).sample()
    return w * n_blocks  # E[w_j] = 1

def train_generator():
    print(f"Using device: {device}")

    # --- Data: train / val / test split ---
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Split train into train (50k) and val (10k)
    n_train = 50000
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(42))
    train_subset = Subset(full_train, indices[:n_train])
    val_dataset = Subset(full_train, indices[n_train:60000])

    # Wrap training set to return observation indices
    train_indexed = IndexedSubset(train_subset)
    train_loader = DataLoader(train_indexed, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Assign each training observation to a block (stable, by observation index)
    block_assignments = torch.randint(0, n_wbb_dim, (n_train,), device=device)

    target = TargetMLP().to(device)
    generator = HyperGenerator(target.total_params, n_omega=n_wbb_dim).to(device)
    optimizer = optim.Adam(generator.parameters(), lr=lr_generator)
    criterion = nn.CrossEntropyLoss(reduction='none')  # per-sample for weighting

    min_log = np.log10(lambda_range[0])
    max_log = np.log10(lambda_range[1])

    print("Starting generator training...")
    for epoch in range(epochs):
        generator.train()
        total_loss = 0
        for batch_idx, (data, target_labels, obs_indices) in enumerate(train_loader):
            data, target_labels = data.to(device), target_labels.to(device)
            obs_indices = obs_indices.to(device)

            # Sample random lambda and WBB weights for this batch
            log_l = np.random.uniform(min_log, max_log)
            l_val = 10**log_l

            omega_blocks = sample_block_weights(n_wbb_dim, device)
            # Look up block ID for each observation by its stable index
            batch_block_ids = block_assignments[obs_indices]
            omega_obs = omega_blocks[batch_block_ids]

            # Build generator input: [normalized_log_lambda, omega_blocks]
            log_l_norm = (log_l - min_log) / (max_log - min_log) * 2 - 1
            gen_input = torch.cat([
                torch.tensor([log_l_norm], dtype=torch.float32, device=device),
                omega_blocks
            ]).unsqueeze(0)  # (1, 1 + n_wbb_dim)

            optimizer.zero_grad()

            weights = generator(gen_input).squeeze(0)
            outputs = target(data, weights)
            per_sample_loss = criterion(outputs, target_labels)
            # Weighted data loss
            data_loss = (omega_obs * per_sample_loss).mean()
            reg_loss = l_val * torch.sum(weights**2)

            loss = data_loss + reg_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} "
                      f"(Data: {data_loss.item():.4f}, Reg: {reg_loss.item():.4f})")

        print(f"Epoch {epoch} Average Loss: {total_loss/len(train_loader):.4f}")

    # --- Evaluation on validation set (for tuning) ---
    print("Evaluating generator over lambda grid on VALIDATION set...")
    start_time_eval = time.time()
    generator.eval()
    lambda_grid = np.logspace(min_log, max_log, 20)
    val_losses = []
    val_accs = []

    # Use unit weights (omega = 1) for point estimation
    omega_ones = torch.ones(n_wbb_dim, device=device)

    criterion_eval = nn.CrossEntropyLoss()

    with torch.no_grad():
        for l_val in lambda_grid:
            log_l = np.log10(l_val)
            log_l_norm = (log_l - min_log) / (max_log - min_log) * 2 - 1
            gen_input = torch.cat([
                torch.tensor([log_l_norm], dtype=torch.float32, device=device),
                omega_ones
            ]).unsqueeze(0)
            weights = generator(gen_input).squeeze(0)

            curr_loss = 0
            correct = 0
            for data, target_labels in val_loader:
                data, target_labels = data.to(device), target_labels.to(device)
                outputs = target(data, weights)
                curr_loss += criterion_eval(outputs, target_labels).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target_labels.view_as(pred)).sum().item()

            val_losses.append(curr_loss / len(val_loader))
            val_accs.append(correct / len(val_loader.dataset))

    eval_time = time.time() - start_time_eval
    print(f"Generator grid evaluation time: {eval_time:.2f}s")

    # Select best lambda on validation set
    best_idx = np.argmax(val_accs)
    best_lambda = lambda_grid[best_idx]
    best_acc_gen_val = val_accs[best_idx]
    print(f"Generator selected lambda (val): {best_lambda:.2e} (Val Acc: {best_acc_gen_val:.4f})")

    # --- Evaluate selected lambda on TEST set ---
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
            outputs = target(data, weights)
            pred = outputs.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target_labels.view_as(pred)).sum().item()
        test_acc_gen = test_correct / len(test_loader.dataset)
    print(f"Generator test accuracy at selected lambda: {test_acc_gen:.4f}")

    # --- WBB uncertainty demo: resample omega at fixed lambda ---
    print("Computing WBB uncertainty summaries...")
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
                outputs = target(data, weights_m)
                pred = outputs.argmax(dim=1, keepdim=True)
                correct_m += pred.eq(target_labels.view_as(pred)).sum().item()
            wbb_accs.append(correct_m / len(test_loader.dataset))

    wbb_mean = np.mean(wbb_accs)
    wbb_std = np.std(wbb_accs)
    print(f"WBB posterior accuracy: mean={wbb_mean:.4f}, std={wbb_std:.4f}")

    # --- Baseline Comparison ---
    print(f"Training baseline model from scratch at lambda={best_lambda:.2e}...")
    start_time_baseline = time.time()

    class StandardMLP(nn.Module):
        def __init__(self, input_size=784, hidden_size=64, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = x.view(-1, 784)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Use the same train loader (without indices) for baseline
    train_loader_plain = DataLoader(Subset(full_train, indices[:n_train]),
                                     batch_size=batch_size, shuffle=True)
    baseline_net = StandardMLP().to(device)
    optimizer_baseline = optim.Adam(baseline_net.parameters(), lr=1e-3)
    criterion_baseline = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        baseline_net.train()
        for data, target_labels in train_loader_plain:
            data, target_labels = data.to(device), target_labels.to(device)
            optimizer_baseline.zero_grad()
            output = baseline_net(data)
            loss = criterion_baseline(output, target_labels)
            l2_reg = sum(p.pow(2).sum() for p in baseline_net.parameters())
            loss = loss + best_lambda * l2_reg
            loss.backward()
            optimizer_baseline.step()

    # Evaluate baseline on VALIDATION set (for figure comparison)
    baseline_net.eval()
    baseline_val_correct = 0
    with torch.no_grad():
        for data, target_labels in val_loader:
            data, target_labels = data.to(device), target_labels.to(device)
            output = baseline_net(data)
            pred = output.argmax(dim=1, keepdim=True)
            baseline_val_correct += pred.eq(target_labels.view_as(pred)).sum().item()
    baseline_val_acc = baseline_val_correct / len(val_loader.dataset)

    # Evaluate baseline on TEST set (for table)
    baseline_test_correct = 0
    with torch.no_grad():
        for data, target_labels in test_loader:
            data, target_labels = data.to(device), target_labels.to(device)
            output = baseline_net(data)
            pred = output.argmax(dim=1, keepdim=True)
            baseline_test_correct += pred.eq(target_labels.view_as(pred)).sum().item()
    baseline_test_acc = baseline_test_correct / len(test_loader.dataset)

    baseline_time = time.time() - start_time_baseline
    print(f"Baseline training time (1 lambda): {baseline_time:.2f}s")
    print(f"Baseline val accuracy: {baseline_val_acc:.4f}")
    print(f"Baseline test accuracy: {baseline_test_acc:.4f}")

    # --- Save Results ---
    os.makedirs('fig', exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.semilogx(lambda_grid, val_losses, 'b-o', label='Generator proxy')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Validation loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.semilogx(lambda_grid, val_accs, 'r-o', label='Generator proxy')
    plt.axvline(best_lambda, color='gray', linestyle='--', alpha=0.7,
                label=rf'Selected $\lambda$={best_lambda:.1e}')
    # Plot baseline on validation set (consistent with y-axis)
    plt.plot(best_lambda, baseline_val_acc, 'k*', markersize=12,
             label='Baseline (from scratch)')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Validation accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('fig/mnist_generator_tuning.pdf')
    print("Figure saved to fig/mnist_generator_tuning.pdf")

    # Save CSV
    import pandas as pd
    df = pd.DataFrame({
        'lambda': lambda_grid,
        'val_loss': val_losses,
        'val_acc': val_accs
    })
    df.to_csv('fig/mnist_generator_results.csv', index=False)

    # Generate LaTeX table
    with open('fig/mnist_tuning_table.tex', 'w') as f:
        f.write("\\begin{tabular}{ccc}\n")
        f.write("\\hline\n")
        f.write("$\\lambda$ & Val.~Loss & Val.~Acc. \\\\\n")
        f.write("\\hline\n")
        sel_indices = [0, 5, 10, 15, 19]
        for i in sel_indices:
            f.write(f"{lambda_grid[i]:.1e} & {val_losses[i]:.4f} & {val_accs[i]:.4f} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Selected ($\\lambda={best_lambda:.1e}$), test & - & {test_acc_gen:.4f} \\\\\n")
        f.write(f"Baseline ($\\lambda={best_lambda:.1e}$), test & - & {baseline_test_acc:.4f} \\\\\n")
        f.write(f"WBB posterior mean, test & - & {wbb_mean:.4f} $\\pm$ {wbb_std:.4f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    print("Done.")

if __name__ == "__main__":
    train_generator()
