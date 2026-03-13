import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import Function
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cuda_kernel_verifier import equivalent, EquivalenceChecker

@triton.jit
def _sum_rows_fwd(x_ptr, out_ptr, N, stride_xm, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    tl.store(out_ptr + row, tl.sum(tl.load(x_ptr + row * stride_xm + cols, mask=cols < N, other=0.0), axis=0))

@triton.jit
def _sum_rows_bwd(grad_out_ptr, grad_in_ptr, N, stride_xm, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    tl.store(grad_in_ptr + row * stride_xm + tl.arange(0, BLOCK_N), tl.load(grad_out_ptr + row), mask=tl.arange(0, BLOCK_N) < N)

def sum_ground_truth(_, x):
    return x.sum(dim=1)

def on_mismatch(args):
    diff = (args.original_result - args.ground_truth_result).abs().max().item()
    raise AssertionError(f"row_sum kernel diverged - max abs diff: {diff:.6f}")

class CudaRowSum(Function):
    @staticmethod
    @equivalent(sum_ground_truth, on_mismatch, rtol=1e-1, atol=1e-6)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        M, N = x.shape
        out = torch.zeros(M, device=x.device, dtype=x.dtype)
        _sum_rows_fwd[(M,)](x, out, N, x.stride(0), BLOCK_N=triton.next_power_of_2(N))
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (x,) = ctx.saved_tensors
        M, N = x.shape
        grad_input = torch.empty_like(x)
        _sum_rows_bwd[(M,)](grad_outputs[0], grad_input, N, x.stride(0), BLOCK_N=triton.next_power_of_2(N))
        return grad_input

cuda_row_sum = CudaRowSum.apply

class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)
        self.gate1 = nn.Linear(1, 1, bias=False)
        self.gate2 = nn.Linear(1, 1, bias=False)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def _gate(self, feat, gate):
        return gate(cuda_row_sum(feat.flatten(1)).unsqueeze(-1)).squeeze(-1).view(-1, 1, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x)) * (1 + torch.sigmoid(self._gate(x, self.gate1)))
        x = self.pool(F.relu(self.conv2(x)) * (1 + torch.sigmoid(self._gate(x, self.gate2))))
        x = self.pool(self.drop(x))
        return self.fc2(self.drop(F.relu(self.fc1(x.flatten(1)))))

def get_loaders(batch_size=256):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return (
        DataLoader(datasets.MNIST("data", train=True,  download=True, transform=tf), batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(datasets.MNIST("data", train=False, download=True, transform=tf), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = correct = seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += logits.argmax(1).eq(y).sum().item()
        seen += len(y)
    return total_loss / seen, correct / seen

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(1).eq(y).sum().item()
        seen += len(y)
    return correct / seen

def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    EquivalenceChecker.start(execution_sample_probability=0.5)
    train_dl, test_dl = get_loaders()
    model = torch.compile(MnistNet().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    print(f"{'Epoch':>5}  {'Loss':>8}  {'Train Acc':>9}  {'Test Acc':>8}\n" + "-" * 40)
    for epoch in range(1, 11):
        loss, train_acc = train_epoch(model, train_dl, optimizer, device)
        print(f"{epoch:5d}  {loss:8.4f}  {train_acc:9.4f}  {evaluate(model, test_dl, device):8.4f}")
        scheduler.step()
    EquivalenceChecker.stop()

if __name__ == "__main__":
    main()
