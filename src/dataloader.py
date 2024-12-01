import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

class BatchLoader:
    def __init__(self,         
        X,
        y,
        batch_size: int):
        self.X = X
        self.y = y
        self.data_shape = self.X.shape[1:]
        self.num_train_samples = X.shape[0]
        self.batch_size = batch_size
        if self.batch_size > self.num_train_samples:
            self.batch_size = self.num_train_samples
        self.num_batches = np.ceil((self.num_train_samples/self.batch_size)).astype(int)
    
    def sample(self, key):
        """Sample a single batch of X, y data."""
        sample_idx = np.random.default_rng(key).choice(
            np.arange(self.num_train_samples),
            (self.batch_size,),
            replace=False,
        )
        return (
            np.take(self.X, sample_idx, axis=0),
            np.take(self.y, sample_idx, axis=0),
        )
    
    def next(self, i):
        """Next batch of X, y data."""
        start = ((i) * self.batch_size)
        end = ((i+1) * self.batch_size)
        if end > self.num_train_samples:
            end = self.num_train_samples
        sample_idx = np.arange(start, end)
        return (
            np.take(self.X, sample_idx, axis=0),
            np.take(self.y, sample_idx, axis=0),
        )
    
def fashion_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_batch = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)))
    train_loader = BatchLoader(*train_batch, batch_size=128)

    forget_indices = range(1000)
    retain_indices = range(1000, len(train_dataset))
    forget_loader = BatchLoader(X=train_batch[0][forget_indices], y=train_batch[1][forget_indices], batch_size=1024)
    retain_loader = BatchLoader(X=train_batch[0][retain_indices], y=train_batch[1][retain_indices], batch_size=1024)

    test_batch = next(iter(torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)))
    test_loader = BatchLoader(*test_batch, batch_size=128)

    shadow_train_indices = range(5000)
    shadow_unseen_indices = range(5000, len(test_dataset))
    shadow_train_loader = BatchLoader(X=test_batch[0][shadow_train_indices], y=test_batch[1][shadow_train_indices], batch_size=128)
    shadow_unseen_loader = BatchLoader(X=test_batch[0][shadow_unseen_indices], y=test_batch[1][shadow_unseen_indices], batch_size=128)
    
    return train_loader, test_loader, forget_loader, retain_loader, shadow_train_loader, shadow_unseen_loader