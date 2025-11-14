import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from sklearn.model_la_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import optuna
from optuna.trial import TrialState
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Preprocessing metadata
metadata_df = pd.read_csv('fundus/Fundus Dataset B/dataset_B.csv')

le_sex = LabelEncoder()
metadata_df['sex'] = le_sex.fit_transform(metadata_df['sex'])

le_treatment = LabelEncoder()
metadata_df['treatment'] = le_treatment.fit_transform(metadata_df['treatment'])

le_dosing_regimen = LabelEncoder()
metadata_df['dosing_regimen'] = le_dosing_regimen.fit_transform(metadata_df['dosing_regimen'])

le_clinical_center = LabelEncoder()
metadata_df['clinical_center'] = le_clinical_center.fit_transform(metadata_df['clinical_center'])

metadata_df['age'] = metadata_df['age'].astype(float)

# Inspect original 'amd' column
print("Unique values in original 'amd' column:", metadata_df['amd'].unique())

# Clean and map the 'amd' column (Normal=0, Wet=1)
mapping = {'Normal': 0.0, 'Wet': 1.0}
metadata_df['amd'] = metadata_df['amd'].map(mapping).astype(float)
print("Unique values in 'amd' after mapping:", metadata_df['amd'].unique())

# Convert 'amd' to numeric again to avoid issues
metadata_df['amd'] = pd.to_numeric(metadata_df['amd'], errors='coerce')
print("Unique values in 'amd' after numeric conversion:", metadata_df['amd'].unique())

# Fill remaining NaN values
metadata_df = metadata_df.fillna(0)

print(metadata_df.head())

input_dim_metadata = metadata_df.shape[1] - 1
print(f"Number of metadata columns (excluding image_file): {input_dim_metadata}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_paths = [os.path.join(row['image_file']) for idx, row in metadata_df.iterrows()]

# Split dataset into train, validation, and test sets
train_img_paths, temp_img_paths, train_metadata_df, temp_metadata_df = train_test_split(
    image_paths, metadata_df.drop(columns=['image_file']), test_size=0.3, random_state=42, stratify=metadata_df['amd']
)

val_img_paths, test_img_paths, val_metadata_df, test_metadata_df = train_test_split(
    temp_img_paths, temp_metadata_df, test_size=0.5, random_state=42, stratify=temp_metadata_df['amd']
)

print(f"Number of training samples: {len(train_img_paths)}")
print(f"Number of validation samples: {len(val_img_paths)}")
print(f"Number of test samples: {len(test_img_paths)}")

# Corrected FundusDataset to return the whole metadata row
class FundusDataset(Dataset):
    def __init__(self, image_paths, metadata_df, transform=None, target_column='amd'):
        self.image_paths = image_paths
        self.metadata_df = metadata_df
        self.transform = transform
        self.target_column = target_column

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.metadata_df.iloc[idx][self.target_column], dtype=torch.float32)
        metadata = torch.tensor(self.metadata_df.iloc[idx].drop(self.target_column).values, dtype=torch.float32)
        return image, metadata, label

class ImageEncoder(nn.Module):
    def __init__(self, vit_model_name='google/vit-base-patch16-224-in21k'):
        super(ImageEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name)

    def forward(self, images):
        outputs = self.vit(images)
        return outputs.last_hidden_state

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, image_features, metadata_features, return_attention=False):
        metadata_features = metadata_features.unsqueeze(1)
        attention_output, attn_weights = self.attention(metadata_features, image_features, image_features)
        combined_features = attention_output + metadata_features
        combined_features = self.fc(combined_features)
        combined_features = self.norm(combined_features)

        if return_attention:
            return combined_features, attn_weights
        return combined_features

class MetadataEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_dim):
        super(MetadataEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc_target = nn.Linear(hidden_dim, target_dim)

    def forward(self, metadata):
        metadata = F.relu(self.fc(metadata))
        return self.fc_target(metadata)

class CrossAttentionModel(nn.Module):
    def __init__(self, image_encoder, metadata_encoder, cross_attention, hidden_size, output_dim):
        super(CrossAttentionModel, self).__init__()
        self.image_encoder = image_encoder
        self.metadata_encoder = metadata_encoder
        self.cross_attention = cross_attention
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(hidden_size, output_dim)

    def forward(self, images, metadata, return_attention=False):
        image_features = self.image_encoder(images)  
        metadata_features = self.metadata_encoder(metadata) 
        if return_attention:
            combined_features, attn_weights = self.cross_attention(image_features, metadata_features, return_attention=True)
        else:
            combined_features = self.cross_attention(image_features, metadata_features)

        pooled_features = self.pooler(combined_features.transpose(1, 2)).squeeze(2)
        output = self.fc_out(pooled_features)

        if return_attention:
            return output, attn_weights
        return output

# ---------- OPTUNA OBJECTIVE ----------
def objective(trial):
    # Hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_int("epochs", 5, 15)
    hidden_dim_metadata = trial.suggest_categorical("hidden_dim_metadata", [64, 128, 256])
    num_heads = trial.suggest_categorical("num_heads", [4, 8, 12])

    train_dataset = FundusDataset(train_img_paths, train_metadata_df.reset_index(drop=True), transform=transform, target_column='amd')
    val_dataset = FundusDataset(val_img_paths, val_metadata_df.reset_index(drop=True), transform=transform, target_column='amd')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Build model
    image_encoder = ImageEncoder()
    metadata_encoder = MetadataEncoder(input_dim_metadata - 1, hidden_dim=hidden_dim_metadata, target_dim=768)
    cross_attention = CrossAttention(hidden_size=768, num_heads=num_heads)
    model = CrossAttentionModel(image_encoder, metadata_encoder, cross_attention, hidden_size=768, output_dim=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_auc = 0
    best_model_path = f"Model_3/best_model_3_trial_{trial.number}.pt"

    for epoch in range(epochs):
        model.train()
        for images, metadata, labels in train_loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, metadata)
            
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_outputs, val_labels = [], []
        with torch.no_grad():
            for images, metadata, labels in val_loader:
                images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
                outputs = model(images, metadata)
                print(outputs.shape)

                val_outputs.extend(torch.sigmoid(outputs).squeeze().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_outputs = np.array(val_outputs)
        val_labels = np.array(val_labels)
        val_auc = roc_auc_score(val_labels, val_outputs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

        trial.report(val_auc, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_auc

# ---------- MAIN ----------
if __name__ == '__main__':
    study_name = "fundus_amd_classification_multimetric_model_3_1"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        study.optimize(objective, n_trials=3)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Best Validation AUC): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # ----------- Evaluate best model ----------
    best_batch_size = trial.params['batch_size']
    best_hidden_dim_metadata = trial.params['hidden_dim_metadata']
    best_num_heads = trial.params['num_heads']

    test_dataset = FundusDataset(test_img_paths, test_metadata_df.reset_index(drop=True), transform=transform, target_column='amd')
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False, drop_last=True)

    best_model_3 = CrossAttentionModel(
        ImageEncoder(),
        MetadataEncoder(input_dim_metadata - 1, hidden_dim=best_hidden_dim_metadata, target_dim=768),
        CrossAttention(hidden_size=768, num_heads=best_num_heads),
        hidden_size=768,
        output_dim=1
    ).to(device)

    best_model_path_3 = f"Model_3/best_model_3_trial_{trial.number}.pt"
    best_model_3.load_state_dict(torch.load(best_model_path_3))
    best_model_3.eval()

    test_outputs, test_labels = [], []
    test_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, metadata, labels in test_loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            outputs = best_model_3(images, metadata)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()
            test_outputs.extend(torch.sigmoid(outputs).squeeze().cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_outputs = np.array(test_outputs)
    test_labels = np.array(test_labels)
    test_predictions = test_outputs > 0.5

    print(f"\nTest Results (with best model):")
    print(f"  Loss: {test_loss / len(test_loader):.4f}")
    print(f"  Accuracy: {accuracy_score(test_labels, test_predictions):.4f}")
    print(f"  Precision: {precision_score(test_labels, test_predictions):.4f}")
    print(f"  Recall: {recall_score(test_labels, test_predictions):.4f}")
    print(f"  F1: {f1_score(test_labels, test_predictions):.4f}")
    print(f"  AUC: {roc_auc_score(test_labels, test_outputs):.4f}")

import matplotlib.pyplot as plt
import seaborn as sns

# Get a batch from the test set
images, metadata, labels = next(iter(test_loader))
images, metadata = images.to(device), metadata.to(device)

# Forward with attention
with torch.no_grad():
    outputs, attn_weights = best_model_3(images, metadata, return_attention=True)

# attn_weights: (B, num_heads, target_len=1, source_len=image_tokens)
weights = attn_weights[0].mean(dim=0).squeeze(0).cpu().numpy()

plt.figure(figsize=(10, 4))
sns.heatmap(weights[np.newaxis, :], cmap="viridis", cbar=True, xticklabels=False, yticklabels=["Attention"])
plt.title("Cross-Attention Map (metadata → image features)")
plt.xlabel("Image Patch Tokens")
plt.tight_layout()
plt.show()
