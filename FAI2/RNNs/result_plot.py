import os
import torch
import matplotlib.pyplot as plt

# Load saved losses
model_save_dir = "C://Users//Sideeq Bello//CSC7809_FoundationModels//Project2//models//saved"
losses = torch.load(os.path.join(model_save_dir, "losses.pth"))

# Plot training and validation loss curves
def plot_loss_curves(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"{model_name} Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(model_save_dir, f"{model_name}_loss_curve.png"))
    plt.show()

# Plot for each model
plot_loss_curves(losses["rnn_train_losses"], losses["rnn_val_losses"], "RNN")
#plot_loss_curves(losses["lstm_train_losses"], losses["lstm_val_losses"], "LSTM")
#plot_loss_curves(losses["transformer_train_losses"], losses["transformer_val_losses"], "Transformer")