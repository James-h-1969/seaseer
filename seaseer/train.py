import torch
from model import SeaSeer

LEARNING_RATE = 0.001
AMOUNT_OF_TRAINING_EPOCHS = 100


def load_data() -> torch.utils.data.DataLoader:
    """Function that pulls the required data into memory"""
    return ""


def train():
    dataloader: torch.utils.data.DataLoader = load_data()
    model = SeaSeer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(AMOUNT_OF_TRAINING_EPOCHS):
        print(f"Beginning epoch {epoch + 1}/{AMOUNT_OF_TRAINING_EPOCHS}")
        running_loss = 0.0

        model.train()
        for i, (inputs, targets) in enumerate(dataloader):
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
