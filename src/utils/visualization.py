import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curve(history):
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'], label='Training Loss', color='b', marker='o', markersize=4)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='r', marker='x', markersize=4)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Learning Curve', fontsize=16)
    plt.grid(visible=True, which='both', linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_error_distribution(y_true, y_pred):
    errors = y_true - y_pred
    plt.figure(figsize=(12, 8))
    sns.histplot(errors, bins=30, kde=True, color='blue', alpha=0.6)
    plt.xlabel('Prediction Error', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Error Distribution', fontsize=16)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()