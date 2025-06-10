import csv
import matplotlib.pyplot as plt

epochs = []
losses = []

with open("../results/training_log.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        losses.append(float(row["loss"]))

plt.plot(epochs, losses, marker='o')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/training_loss_plot.png")
plt.show()
