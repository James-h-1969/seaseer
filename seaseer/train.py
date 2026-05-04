import torch
from dataloader import create_dataloader
from model import SeaSeer

LEARNING_RATE = 0.001
AMOUNT_OF_TRAINING_EPOCHS = 100
BATCH_SIZE = 8


def train():
    train_loader, val_loader, channel_names = create_dataloader(
        batch_size=BATCH_SIZE,
    )
    in_channels = len(channel_names)
    print(f"Loaded {in_channels} channels: {channel_names}")

    model = SeaSeer(in_channels=in_channels, out_channels=in_channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(AMOUNT_OF_TRAINING_EPOCHS):
        print(f"Beginning epoch {epoch + 1}/{AMOUNT_OF_TRAINING_EPOCHS}")
        running_loss = 0.0

        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                avg_loss = running_loss / 10
                print(f"  Batch {i + 1}, Loss: {avg_loss:.4f}")
                running_loss = 0.0

        print(f"Epoch {epoch + 1} complete.")

    print("Training finished.")
    torch.save(model.state_dict(), "model_checkpoints/seaseer_model.pth")


if __name__ == "__main__":
    train()
