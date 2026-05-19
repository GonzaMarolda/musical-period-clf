import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Paths ---
PROCESSED_DIR = Path("data/processed")
FEATURES_CSV = PROCESSED_DIR / "maestro_features.csv"
COLUMNS_JSON = PROCESSED_DIR / "feature_columns.json"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

# --- Dataset Definition ---
class MaestroDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Model Definition ---
class MusicPeriodMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicPeriodMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)

            # nn.Linear(input_size, 2),
            # nn.Linear(2, num_classes)

            
            # nn.Linear(input_size, 512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Linear(128,64),
            # nn.ReLU(),
            # nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP for Music Period Classification")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--run-name", type=str, default="default", help="Name of the run for saving results separately")
    return parser.parse_args()

def main():
    args = parse_args()
    
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("[1/5] Loading data...")
    if not FEATURES_CSV.exists() or not COLUMNS_JSON.exists():
        print("Features or columns file not found. Run preprocess_maestro.py first.")
        sys.exit(1)

    with open(COLUMNS_JSON, "r") as f:
        feature_cols = json.load(f)

    df = pd.read_csv(FEATURES_CSV)
    
    # Filter valid rows
    valid_df = df[df["feature_status"] == "ok"].copy()
    print(f"Loaded {len(valid_df)} valid samples out of {len(df)} total.")

    if len(valid_df) == 0:
        print("No valid samples found!")
        sys.exit(1)
    
    # Check if target column exists
    if "period_id" not in valid_df.columns:
        print("Target column 'period_id' not found in dataset!")
        sys.exit(1)

    # --- Preprocessing ---
    print("[2/5] Preprocessing and Splitting data...")
    
    # Extract features (X) and labels (y)
    # Some features might be NaN if something went wrong but status is OK, let's fillna with 0
    X = valid_df[feature_cols].fillna(0).values
    
    # Get labels
    # ensure zero-indexed labels
    y = valid_df["period_id"].values
    unique_labels = sorted(list(set(y)))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_mapped = np.array([label_map[label] for label in y])
    num_classes = len(unique_labels)
    print(f"Found {num_classes} classes: {unique_labels}")

    # Use official_split if possible
    if "official_split" in valid_df.columns:
        train_mask = valid_df["official_split"] == "train"
        val_mask = valid_df["official_split"] == "validation"
        test_mask = valid_df["official_split"] == "test"
        
        X_train, y_train = X[train_mask], y_mapped[train_mask]
        X_val, y_val = X[val_mask], y_mapped[val_mask]
        X_test, y_test = X[test_mask], y_mapped[test_mask]
        
        # If any split is empty, fallback to random split
        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            print("Official split missing some sets. Falling back to random split.")
            X_train, X_temp, y_train, y_temp = train_test_split(X, y_mapped, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y_mapped, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Create DataLoaders
    train_dataset = MaestroDataset(X_train, y_train)
    val_dataset = MaestroDataset(X_val, y_val)
    test_dataset = MaestroDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # --- Training ---
    print(f"[3/5] Initializing Model (Epochs: {args.epochs}, LR: {args.learning_rate})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MusicPeriodMLP(input_size=len(feature_cols), num_classes=num_classes).to(device)
    
    # Save and show model topology
    topology_str = str(model)
    print("\n--- Model Topology ---")
    print(topology_str)
    print("----------------------\n")
    
    hyperparameters_str = (
        f"\n--- Hyperparameters ---\n"
        f"Epochs: {args.epochs}\n"
        f"Learning Rate: {args.learning_rate}\n"
        f"Batch Size: {args.batch_size}\n"
        f"-----------------------\n"
    )
    
    with open(RESULTS_DIR / f"topology_{args.run_name}.txt", "w") as f:
        f.write(topology_str + "\n" + hyperparameters_str)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("[4/5] Training...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(args.epochs):
        # Train phase
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass (Backpropagation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = correct / total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{args.epochs:03d} - "
                  f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Validation Acc')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'training_history_{args.run_name}.png')
    
    # Save Model
    model_path = MODELS_DIR / f"mlp_model_{args.run_name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # --- Evaluation ---
    def evaluate_and_plot_cm(loader, dataset_name):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        print(f"\n[{dataset_name}] Accuracy: {acc:.4f}")
        
        report_str = classification_report(all_labels, all_preds, target_names=[str(x) for x in unique_labels], zero_division=0)
        report_dict = classification_report(all_labels, all_preds, target_names=[str(x) for x in unique_labels], output_dict=True, zero_division=0)
        
        if dataset_name == "Test Set":
            print(f"[{dataset_name}] Classification Report:")
            print(report_str)

        # Save precision and recall per class
        metrics_per_class = {}
        for label in [str(x) for x in unique_labels]:
            metrics_per_class[label] = {
                "precision": float(report_dict[label]["precision"] * 100),
                "recall": float(report_dict[label]["recall"] * 100)
            }
            
        metrics_file = RESULTS_DIR / f'metrics_{dataset_name.replace(" ", "_").lower()}_{args.run_name}.json'
        with open(metrics_file, "w") as f:
            json.dump(metrics_per_class, f, indent=4)

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(x) for x in unique_labels])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f'confusion_matrix_{dataset_name.replace(" ", "_").lower()}_{args.run_name}.png')
        plt.close()

    print("[5/5] Evaluating and generating confusion matrices...")
    evaluate_and_plot_cm(train_loader, "Train Set")
    evaluate_and_plot_cm(test_loader, "Test Set")

if __name__ == "__main__":
    main()
