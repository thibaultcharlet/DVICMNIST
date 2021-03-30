from torch.optim import AdamW
from tqdm import tqdm
from webmnist.data import MNISTDataset, MNISTLoader
from webmnist.model import LeNet5

import torch
import torch.nn as nn


def train(path: str, epochs: int = 3) -> None:
    dataset = MNISTDataset()
    loader = MNISTLoader()

    model = LeNet5(n_classes=10).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = AdamW(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()
        total_loss, total_acc = 0, 0

        pbar = tqdm(loader.train, desc="Train")
        for img, label in pbar:
            img, label = img.cuda(), label.cuda()
            optimizer.zero_grad()
            
            preds = model(img)
            loss = criterion(preds, label)
            acc = (torch.argmax(torch.softmax(
                preds, dim=1,
            ), dim=1) == label).sum()
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item() / len(loader.train)
            total_acc += acc.item() / len(dataset.train)

            pbar.set_postfix(
                loss=f"{total_loss:.2e}",
                acc=f"{total_acc * 100:.2f}%",
            )

        model.eval()
        total_loss, total_acc = 0, 0

        pbar = tqdm(loader.test, desc="Test")
        for img, label in pbar:
            img, label = img.cuda(), label.cuda()
            
            preds = model(img)
            loss = criterion(preds, label)
            acc = (torch.argmax(torch.softmax(
                preds, dim=1,
            ), dim=1) == label).sum()
            
            total_loss += loss.item() / len(loader.test)
            total_acc += acc.item() / len(dataset.test)

            pbar.set_postfix(
                loss=f"{total_loss:.2e}",
                acc=f"{total_acc * 100:.2f}%",
            )

    torch.save(model.state_dict(), path)