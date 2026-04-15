import torch
from model import SeaSeer

MODEL_PATH = "model_checkpoints/seaseer_model.pth"


def load_data() -> torch.utils.data.DataLoader:
    """Load evaluation dataset."""
    # TODO: implement data loading
    raise NotImplementedError


def evaluate():
    """Evaluate a trained SeaSeer model on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = SeaSeer()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    dataloader = load_data()
    loss_fn = torch.nn.MSELoss()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Evaluation MSE: {avg_loss:.4f}")
    return avg_loss


if __name__ == "__main__":
    evaluate()
