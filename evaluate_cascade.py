import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Custom Dataset
# -------------------------
class FundusDataset(Dataset):
    """Dataset for fundus images and associated metadata."""
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
        label = self.metadata_df.iloc[idx][self.target_column]
        metadata = torch.tensor(self.metadata_df.iloc[idx].drop(self.target_column).values, dtype=torch.float32)
        return image, metadata, label

# -------------------------
# Image Encoder
# -------------------------
class ImageEncoder(nn.Module):
    """Vision Transformer encoder."""
    def __init__(self, vit_model_name='google/vit-base-patch16-224-in21k'):
        super(ImageEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name)

    def forward(self, images):
        outputs = self.vit(images)
        return outputs.last_hidden_state

# -------------------------
# Cross-Attention Module
# -------------------------
class CrossAttention(nn.Module):
    """Cross-attention between image and metadata features."""
    def __init__(self, hidden_size, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, image_features, metadata_features, return_attention=False):
        metadata_features = metadata_features.unsqueeze(1)
        attn_output, attn_weights = self.attention(metadata_features, image_features, image_features)
        combined = attn_output + metadata_features
        combined = self.fc(combined)
        combined = self.norm(combined)

        if return_attention:
            return combined, attn_weights
        return combined

# -------------------------
# Metadata Encoder
# -------------------------
class MetadataEncoder(nn.Module):
    """Simple MLP for metadata encoding."""
    def __init__(self, input_dim, hidden_dim, target_dim):
        super(MetadataEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc_target = nn.Linear(hidden_dim, target_dim)

    def forward(self, metadata):
        x = F.relu(self.fc(metadata))
        return self.fc_target(x)

# -------------------------
# Cross-Attention Model
# -------------------------
class CrossAttentionModel(nn.Module):
    """Full model combining image and metadata via cross-attention."""
    def __init__(self, image_encoder, metadata_encoder, cross_attention, hidden_size, output_dim):
        super(CrossAttentionModel, self).__init__()
        self.image_encoder = image_encoder
        self.metadata_encoder = metadata_encoder
        self.cross_attention = cross_attention
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(hidden_size, output_dim)

    def forward(self, images, metadata, return_attention=False):
        image_features = self.image_encoder(images)  # (B, seq_len, 768)
        metadata_features = self.metadata_encoder(metadata)  # (B, 768)
        if return_attention:
            combined_features, attn_weights = self.cross_attention(image_features, metadata_features, return_attention=True)
        else:
            combined_features = self.cross_attention(image_features, metadata_features)

        pooled_features = self.pooler(combined_features.transpose(1, 2)).squeeze(2)
        output = self.fc_out(pooled_features)

        if return_attention:
            return output, attn_weights
        return output

# -------------------------
# Load and preprocess metadata
# -------------------------
metadata_a = pd.read_csv('fundus/Fundus Dataset A/dataset_A.csv')
metadata_b = pd.read_csv('fundus/Fundus Dataset B/dataset_B.csv')
combined_metadata = pd.concat([metadata_a, metadata_b], ignore_index=True)

# Encode categorical columns
for col in ['sex', 'treatment', 'dosing_regimen', 'clinical_center']:
    le = LabelEncoder()
    combined_metadata[col] = le.fit_transform(combined_metadata[col])

combined_metadata['age'] = combined_metadata['age'].astype(float)
amd_map = {'Normal': 0, 'Wet': 1, 'Intermediate': 2, 'GA': 3}
combined_metadata['amd'] = combined_metadata['amd'].map(amd_map)

input_dim_metadata = combined_metadata.shape[1] - 1  # Exclude 'image_file'

# -------------------------
# Image transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

image_paths = [os.path.join(row['image_file']) for _, row in combined_metadata.iterrows()]

# -------------------------
# Train/Validation/Test split
# -------------------------
train_img_paths, temp_img_paths, train_metadata_df, temp_metadata_df = train_test_split(
    image_paths, combined_metadata.drop(columns=['image_file']), 
    test_size=0.3, random_state=42, stratify=combined_metadata['amd']
)

val_img_paths, test_img_paths, val_metadata_df, test_metadata_df = train_test_split(
    temp_img_paths, temp_metadata_df,
    test_size=0.5, random_state=42, stratify=temp_metadata_df['amd']
)

test_dataset = FundusDataset(test_img_paths, test_metadata_df.reset_index(drop=True), transform=transform, target_column='amd')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -------------------------
# Function to load trained model
# -------------------------
def load_trained_model(model_path, trial):
    model = CrossAttentionModel(
        ImageEncoder(),
        MetadataEncoder(input_dim_metadata-1, hidden_dim=trial.params['hidden_dim_metadata'], target_dim=768),
        CrossAttention(hidden_size=768, num_heads=trial.params['num_heads']),
        hidden_size=768,
        output_dim=1
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# -------------------------
# Load best models from Optuna studies
# -------------------------
study1 = optuna.load_study(study_name="fundus_amd_classification_multimetric_14",
                           storage="sqlite:///fundus_amd_classification_multimetric_14.db")
trial1 = study1.best_trial
model_1 = load_trained_model(f"Model_1/best_model_1_trial_{trial1.number}.pt", trial1)

study2 = optuna.load_study(study_name="fundus_amd_classification_multimetric_model_2_3",
                           storage="sqlite:///fundus_amd_classification_multimetric_model_2_3.db")
trial2 = study2.best_trial
model_2 = load_trained_model(f"Model_2/best_model_2_trial_{trial2.number}.pt", trial2)

study3 = optuna.load_study(study_name="fundus_amd_classification_multimetric_model_3_1",
                           storage="sqlite:///fundus_amd_classification_multimetric_model_3_1.db")
trial3 = study3.best_trial
model_3 = load_trained_model(f"Model_3/best_model_3_trial_{trial3.number}.pt", trial3)

# -------------------------
# Cascade evaluation
# -------------------------
true_labels = []
final_preds = []
final_probs = []

n_classes = 4

with torch.no_grad():
    for images, metadata, labels in test_loader:
        images, metadata = images.to(device), metadata.to(device)
        labels_np = labels.cpu().numpy()
        true_labels.extend(labels_np)

        out1 = torch.sigmoid(model_1(images, metadata)).squeeze()
        pred1 = (out1 > 0.5).long().cpu().numpy()

        for i in range(len(pred1)):
            if pred1[i] == 1:  # A -> Intermediate/GA
                out2 = torch.sigmoid(model_2(images[i:i+1], metadata[i:i+1])).squeeze()
                final_class = 2 if out2 > 0.5 else 3
                probs = [0, 0, 1 - out2.cpu().item(), out2.cpu().item()]
            else:  # B -> Wet/Normal
                out3 = torch.sigmoid(model_3(images[i:i+1], metadata[i:i+1])).squeeze()
                final_class = 1 if out3 > 0.5 else 0
                probs = [1 - out3.cpu().item(), out3.cpu().item(), 0, 0]

            final_preds.append(final_class)
            final_probs.append(probs)

true_labels = np.array(true_labels)
final_preds = np.array(final_preds)
final_probs = np.array(final_probs)

# -------------------------
# Compute metrics
# -------------------------
accuracy = accuracy_score(true_labels, final_preds)
precision = precision_score(true_labels, final_preds, average='weighted')
recall = recall_score(true_labels, final_preds, average='weighted')
f1 = f1_score(true_labels, final_preds, average='weighted')
roc_auc = roc_auc_score(true_labels, final_probs, average='weighted', multi_class='ovr')

print("\nCascade Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print(f"AUC-ROC (weighted): {roc_auc:.4f}")
