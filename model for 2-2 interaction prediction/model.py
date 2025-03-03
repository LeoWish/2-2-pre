import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression as LR
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(3)

# Load data from an Excel file
def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    df_features = df.iloc[:, 7:]
    data_value = np.array(df_features, dtype=object)
    transfer = StandardScaler()
    data_value = transfer.fit_transform(data_value)
    return data_value

# Define the Generator model for GAN
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator model for GAN
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Train the GAN model
def train_gan(generator, discriminator, X_train, latent_dim, epochs=100, batch_size=128):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(epochs):
        # Train the discriminator
        real_labels = torch.ones((batch_size, 1))
        fake_labels = torch.zeros((batch_size, 1))

        # Train with real data
        indices = torch.randint(0, X_train.shape[0], (batch_size,))
        real_data = X_train[indices]
        optimizer_D.zero_grad()
        outputs = discriminator(real_data)
        d_loss_real = criterion(outputs, real_labels)

        # Train with fake data
        random_latent_vectors = torch.randn((batch_size, latent_dim))
        fake_data = generator(random_latent_vectors)
        outputs = discriminator(fake_data)
        d_loss_fake = criterion(outputs, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train the generator
        random_latent_vectors = torch.randn((batch_size, latent_dim))
        fake_data = generator(random_latent_vectors)
        outputs = discriminator(fake_data)
        g_loss = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Discriminator Loss: {d_loss.item()}, Generator Loss: {g_loss.item()}")

# Generate new data using the trained generator
def generate_data(generator, num_samples, latent_dim):
    random_latent_vectors = torch.randn((num_samples, latent_dim))
    generated_data = generator(random_latent_vectors).detach()
    return generated_data

# Train base models (e.g., Random Forest, SVM, etc.)
def train_base_models(X_train, y_train, X_test):
    base_models = [
        ("rf", RandomForestClassifier(n_estimators=1000, max_depth=15, random_state=7, min_samples_split=4)),
        ("svm", SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=7)),
        ("knn", KNN(n_neighbors=75, weights='uniform', metric='euclidean')),
        ("lr", LR(solver='saga', class_weight='balanced', max_iter=100, random_state=7)),
        ("xgb", xgb.XGBClassifier(n_estimators=1200, max_depth=15, learning_rate=0.1, random_state=7, use_label_encoder=False)),
        ("lgbm", lgb.LGBMClassifier(n_estimators=1200, max_depth=12, learning_rate=0.1, random_state=7)),
    ]

    base_model_predictions_train = []
    base_model_predictions_test = []

    for name, model in base_models:
        model.fit(X_train.numpy(), y_train.numpy().ravel())
        train_pred = model.predict_proba(X_train.numpy())[:, 1].reshape(-1, 1)
        base_model_predictions_train.append(train_pred)
        test_pred = model.predict_proba(X_test.numpy())[:, 1].reshape(-1, 1)
        base_model_predictions_test.append(test_pred)

    return np.hstack(base_model_predictions_train), np.hstack(base_model_predictions_test)

# Define the optimized MLP model
class OptimizedMLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5, hidden_layers=[512, 256, 128, 64, 32]):
        super(OptimizedMLP, self).__init__()
        layers = []
        previous_layer_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_layer_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            previous_layer_dim = hidden_dim

        layers.append(nn.Linear(previous_layer_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Train the MLP model
def train_mlp(model, X_train, y_train, epochs=200, batch_size=128, patience=5):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# Main function
def main():
    # Load data
    file_path = 'dataset/dataset.xlsx'
    data_value = load_data(file_path)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(data_value[:22], dtype=torch.float32)
    X_test = torch.tensor(data_value[22:], dtype=torch.float32)
    y_train = torch.tensor([1] * 11 + [0] * 11, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor([1] * 11 + [0] * 19, dtype=torch.float32).unsqueeze(1)

    # GAN parameters
    latent_dim = 64
    generator = Generator(latent_dim, X_train.shape[1])
    discriminator = Discriminator(X_train.shape[1])

    # Train GAN
    train_gan(generator, discriminator, X_train, latent_dim)

    # Generate new data
    num_samples = 2000
    generated_data = generate_data(generator, num_samples, latent_dim)
    generated_labels = torch.cat([torch.ones((num_samples // 2, 1)), torch.zeros((num_samples // 2, 1))], dim=0)

    # Augment training data
    X_train = torch.cat([X_train, generated_data], dim=0)
    y_train = torch.cat([y_train, generated_labels], dim=0)

    # Train base models
    stacked_X_train, stacked_X_test = train_base_models(X_train, y_train, X_test)

    # Train MLP model
    mlp_model = OptimizedMLP(input_dim=stacked_X_train.shape[1])
    train_mlp(mlp_model, torch.tensor(stacked_X_train, dtype=torch.float32), y_train)

    # Make predictions using the MLP model
    mlp_model.eval()
    with torch.no_grad():
        predictions = mlp_model(torch.tensor(stacked_X_test, dtype=torch.float32)).numpy()
        stacked_pred = (predictions > 0.5).astype(int).flatten()

    # Evaluate performance
    accuracy = accuracy_score(y_test.numpy(), stacked_pred)
    print(f"Stacking Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test.numpy(), stacked_pred))
    # Print final predictions
    print("\nFinal Predictions:")
    print(stacked_pred)

if __name__ == "__main__":
    main()