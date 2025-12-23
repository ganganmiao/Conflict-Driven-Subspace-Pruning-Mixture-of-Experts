import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class TaskTaggedDataset(Dataset):
    """
    功能：
    1. 注入 task_id
    2. 执行标签重映射 (Label Remapping)，将 0-9 映射到 0-1
    """

    def __init__(self, dataset, task_id):
        self.dataset = dataset
        self.task_id = task_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # === 标签重映射 (Label Remapping) ===
        # 将原始的 0-9 标签映射到 0-1，以适配共享的 10 维输出空间
        final_label = label

        if self.task_id == 1:
            # Task 1: Parity (KMNIST)
            # 偶数->0, 奇数->1
            final_label = label % 2

        elif self.task_id == 2:
            # Task 2: Magnitude (FashionMNIST)
            # 0-4->0 (Small/Tops), 5-9->1 (Large/Shoes)
            final_label = 1 if label >= 5 else 0

        # 对于 Task 0 (MNIST)，final_label 保持 0-9

        return image, self.task_id, final_label


def get_mixed_task_loaders(batch_size=64, root='./data'):
    # 统一预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 1. 加载数据集
    mnist_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    kmnist_train = datasets.KMNIST(root=root, train=True, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root=root, train=False, download=True, transform=transform)

    fmnist_train = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    # 2. 包装
    train_datasets = [
        TaskTaggedDataset(mnist_train, task_id=0),
        TaskTaggedDataset(kmnist_train, task_id=1),
        TaskTaggedDataset(fmnist_train, task_id=2)
    ]
    test_datasets = [
        TaskTaggedDataset(mnist_test, task_id=0),
        TaskTaggedDataset(kmnist_test, task_id=1),
        TaskTaggedDataset(fmnist_test, task_id=2)
    ]

    # 3. 混合
    full_train_set = ConcatDataset(train_datasets)
    full_test_set = ConcatDataset(test_datasets)

    train_loader = DataLoader(full_train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(full_test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader