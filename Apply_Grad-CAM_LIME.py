import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lime.lime_tabular import LimeTabularExplainer
import pickle

# -------------------------
# Grad-CAM & LIME Utilities
# -------------------------

def get_vit_attention_map(model, image_tensor, device):
    """Generate the attention map from a ViT model for a single image."""
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs, attn_weights = model(
            image_tensor, 
            torch.zeros((1, model.metadata_encoder.fc.in_features)).to(device), 
            return_attention=True
        )
    
    attn = attn_weights[0]  # shape: (num_heads, 1, seq_len)
    attn_mean = attn.mean(0).squeeze(0).cpu().numpy()  # shape: (seq_len,)
    return attn_mean


def visualize_attention_overlay(image_tensor, attention_weights, title="Attention Overlay"):
    """Overlay attention weights on the image and display using matplotlib."""
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    height, width = 224, 224
    num_patches = int(np.sqrt(len(attention_weights)))
    
    if num_patches * num_patches != len(attention_weights):
        print(f"Warning: expected {num_patches*num_patches} patches, got {len(attention_weights)}")
        num_patches = int(np.sqrt(len(attention_weights) - 1))

    attn_map = attention_weights[:num_patches * num_patches].reshape(num_patches, num_patches)
    attn_map = cv2.resize(attn_map, (width, height))
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    overlay = 0.6 * image_np + 0.4 * heatmap[..., ::-1] / 255.0

    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.show()


def explain_metadata_with_lime(model, metadata_array, feature_names, batch_size=1):
    """Generate LIME explanations for tabular metadata."""
    model.eval()

    def predict_fn(x):
        with torch.no_grad():
            batch_data = torch.tensor(x, dtype=torch.float32).to(next(model.parameters()).device)
            dummy_images = torch.zeros((batch_data.shape[0], 3, 224, 224)).to(next(model.parameters()).device)
            outputs = model(dummy_images, batch_data)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            return np.vstack([1 - probs, probs]).T

    explainer = LimeTabularExplainer(
        metadata_array,
        feature_names=feature_names,
        class_names=["0", "1"],
        discretize_continuous=True
    )

    num_samples = metadata_array.shape[0]
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = metadata_array[start_idx:end_idx]
        exp = explainer.explain_instance(batch_data[0], predict_fn, num_features=10, num_samples=100)
        exp.show_in_notebook()
        print(f"LIME explanation shown for batch {start_idx // batch_size + 1}.")
        save_lime_explanation(exp, start_idx // batch_size + 1)

    return exp


def save_lime_explanation(explainer_obj, batch_idx):
    """Save LIME explanation object to a pickle file."""
    with open(f'lime_explanation_batch_{batch_idx}.pkl', 'wb') as f:
        pickle.dump(explainer_obj, f)


def load_lime_explanation(batch_idx):
    """Load LIME explanation object from a pickle file."""
    with open(f'lime_explanation_batch_{batch_idx}.pkl', 'rb') as f:
        return pickle.load(f)


# -------------------------
# Main Workflow
# -------------------------

def main():
    # Load trained model
    best_model_path = f"Model_3/best_model_3_trial_{trial.number}.pt"
    best_model_3.to(device)
    
    # Extract sample for visualization
    sample_image, sample_metadata, _ = test_dataset[4]  # Replace with your actual dataset variable
    attention_weights = get_vit_attention_map(best_model_3, sample_image, device)
    
    # Visualize attention overlay
    visualize_attention_overlay(sample_image, attention_weights)
    
    # LIME explanation on metadata
    metadata_np = test_metadata_df.drop(columns='amd').values  # Replace 'amd' if needed
    feature_names = test_metadata_df.drop(columns='amd').columns.tolist()
    explain_metadata_with_lime(best_model_3, metadata_np, feature_names, batch_size=10)


if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your datasets here
    # test_dataset = ...
    # test_metadata_df = ...
    
    main()
