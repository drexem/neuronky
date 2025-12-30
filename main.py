from src.dataset import ATPMatchesDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn

import torch
import torch.optim as optim
import pickle
import os

if __name__ == '__main__':
    print("Started training script.")

    dataset = ATPMatchesDataset('data/processed/atp_matches_2000_2024_final.csv')

    num_workers = 2
    train_loader = DataLoader(dataset.TRAIN, batch_size=32, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dataset.DEV, batch_size=32, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset.TEST, batch_size=32, shuffle=False, num_workers=num_workers)

    class ATPMatchPredictor(nn.Module):
        def __init__(self, net : torch.nn.Sequential):
            super(ATPMatchPredictor, self).__init__()

            self.network = net

        def forward(self, x):
            return self.network(x)

    # Check if CUDA is available and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get input size from first batch
    sample_batch = next(iter(train_loader))
    features = sample_batch['features']
    labels = sample_batch['target']
    input_size = features.shape[1]

    print(f"Input size: {input_size}")


    hidden_sizes=[128, 64, 32]
    layers = []
    prev_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        prev_size = hidden_size

    # Output layer (binary classification)
    layers.append(nn.Linear(prev_size, 1))
    first_arch = nn.Sequential(*layers)


    architectures = [
        first_arch,
        nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.BatchNorm1d(256),
                    nn.GELU(),
                    nn.Dropout(0.15),

                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.GELU(),
                    nn.Dropout(0.15),

                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.GELU(),

                    nn.Linear(64, 1)
                ),
        # Deep residual-style network
        nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1)
        ),
        # Wide shallow network
        nn.Sequential(
            nn.Linear(input_size, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(0.1),

            nn.Linear(192, 1)
        ),
        # Funnel architecture with smaller dropouts
        nn.Sequential(
            nn.Linear(input_size, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(320, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(160, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(80, 40),
            nn.BatchNorm1d(40),
            nn.ReLU(),

            nn.Linear(40, 1)
        ),
        # Narrow deep network
        nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.15),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Linear(128, 1)
        ),
        # Aggressive dropout network
        nn.Sequential(
            nn.Linear(input_size, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(100, 1)
        )
    ]

    num_epochs = 50

    optimizer_classes = [
        optim.AdamW,
        optim.SGD,
        # Add more optimizers if needed
    ]

    scheduler_classes = [
        lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs),
        lambda opt: optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1),
        # Add more schedulers if needed
    ]


    for arch_idx, arch in enumerate(architectures):
        for opt_idx, opt_cls in enumerate(optimizer_classes):
            for sched_idx, sched_fn in enumerate(scheduler_classes):
                print(
                    f"Testing architecture {arch_idx}, optimizer {opt_cls.__name__}, scheduler {sched_fn.__name__ if hasattr(sched_fn, '__name__') else sched_fn}")
                model = ATPMatchPredictor(arch).to(device)
                optimizer = opt_cls(model.parameters(), lr=3e-4, weight_decay=1e-4)
                scheduler = sched_fn(optimizer)

                # Loss and optimizer
                criterion = nn.BCEWithLogitsLoss()

                for epoch in range(num_epochs):
                    model.train()
                    train_loss = 0.0

                    for batch in train_loader:
                        features = batch['features'].to(device)  # Move to device
                        labels = batch['target'].float().to(device)  # Move to device

                        # Forward pass
                        outputs = model(features)
                        loss = criterion(outputs.squeeze(), labels)

                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()

                    # Validation
                    model.eval()
                    dev_loss = 0.0
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for batch in dev_loader:
                            features = batch['features'].to(device)  # Move to device
                            labels = batch['target'].float().to(device)

                            outputs = model(features)
                            loss = criterion(outputs.squeeze(), labels)
                            dev_loss += loss.item()

                            predicted = (outputs.squeeze() > 0.5).float()
                            correct += (predicted == labels).sum().item()
                            total += labels.size(0)

                    accuracy = 100 * correct / total
                    print(f'Epoch [{epoch+1}/{num_epochs}], '
                          f'Train Loss: {train_loss/len(train_loader):.4f}, '
                          f'Dev Loss: {dev_loss/len(dev_loader):.4f}, '
                          f'Dev Accuracy: {accuracy:.2f}%')



                # Create models directory if it doesn't exist
                os.makedirs('models', exist_ok=True)

                # Save the entire model as pickle
                model_data = {
                    'model': model,
                    'input_size': input_size,
                    'hidden_sizes': [128, 64, 32],
                    'epoch': num_epochs,
                    'device': str(device),
                    'optimizer' : optimizer,
                    'scheduler' : scheduler
                }

                with open(f'models/atp_match_predictor_a{arch_idx}_o{opt_idx}_s{sched_idx}.pkl', 'wb') as f:
                    pickle.dump(model_data, f)

                print(f"Model saved as pickle to models/atp_match_predictor.pkl")