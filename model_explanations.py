import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shap
from lime.lime_image import LimeImageExplainer
from lime.lime_tabular import LimeTabularExplainer
from skimage.segmentation import mark_boundaries
import pickle

# -------------------------
# Grad-CAM & LIME/SHAP Utilities
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
    attn = attn_weights[0]  # (num_heads, 1, seq_len)
    return attn.mean(0).squeeze(0).cpu().numpy()  # Attention vector per patch


def visualize_attention_overlay(image_tensor, attention_weights, title="Attention Overlay"):
    """Overlay attention weights on an image and display with matplotlib."""
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    height, width = 224, 224
    num_patches = int(np.sqrt(len(attention_weights)))
    if num_patches * num_patches != len(attention_weights):
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


# -------------------------
# LIME for Images
# -------------------------
def explain_lime_image(model, image_tensor, class_names, device):
    """Generate LIME explanations for a single image."""
    explainer = LimeImageExplainer()

    if image_tensor.shape[0] == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)
    elif image_tensor.shape[0] != 3:
        raise ValueError("Image must have 3 channels (RGB).")

    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    def predict_fn(x):
        x_tensor = torch.tensor(x).permute(0, 3, 1, 2).float().to(device) / 255.0
        with torch.no_grad():
            dummy_metadata = torch.zeros((x_tensor.shape[0], model.metadata_encoder.fc.in_features)).to(device)
            outputs = model(x_tensor, dummy_metadata)
            probs = torch.sigmoid(outputs).cpu().numpy().reshape(-1)
            return np.vstack([1 - probs, probs]).T

    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=100,
        batch_size=20
    )

    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(mark_boundaries(temp, mask))
    plt.title(f"LIME explanation (class {class_names[explanation.top_labels[0]]})")
    plt.axis("off")
    plt.show()


# -------------------------
# SHAP for Images
# -------------------------
def explain_shap_image(model, image_tensor, device):
    """Generate SHAP explanations for a single image."""
    model.eval()

    background = image_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().numpy()  # (1, 224, 224, 3)
    background_flat = background.reshape((1, -1))

    dummy_metadata = torch.zeros((1, model.metadata_encoder.fc.in_features)).to(device)

    def model_wrapper(x_flat_np):
        x_img = torch.tensor(x_flat_np.reshape((-1, 224, 224, 3)), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        x_meta = dummy_metadata.repeat(x_img.shape[0], 1)
        with torch.no_grad():
            output = model(x_img, x_meta)
            prob = torch.sigmoid(output).cpu().numpy()
        return prob

    explainer = shap.KernelExplainer(model_wrapper, background_flat)
    shap_values = explainer.shap_values(background_flat, nsamples=50)

    shap_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    seq_len = 14 * 14
    if shap_array.shape[1] != seq_len:
        raise ValueError(f"SHAP values are not the expected size: {shap_array.shape[1]} != {seq_len}")

    shap_map = np.abs(shap_array).mean(axis=1).reshape((14, 14))
    shap_map = cv2.resize(shap_map, (224, 224))
    shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min())

    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * shap_map), cv2.COLORMAP_JET)
    overlay = 0.6 * image_np + 0.4 * heatmap[..., ::-1] / 255.0

    plt.imshow(overlay)
    plt.title("SHAP Overlay on Image")
    plt.axis("off")
    plt.show()


# -------------------------
# LIME & SHAP for Metadata
# -------------------------
def explain_lime_metadata(model, metadata_array, feature_names, batch_size=10):
    """Generate LIME explanations for tabular metadata."""
    model.eval()

    def predict_fn(x):
        x_tensor = torch.tensor(x).float().to(next(model.parameters()).device)
        with torch.no_grad():
            dummy_metadata = torch.zeros((x_tensor.shape[0], model.metadata_encoder.fc.in_features)).to(next(model.parameters()).device)
            outputs = model(torch.zeros((x_tensor.shape[0], 3, 224, 224)).to(next(model.parameters()).device), dummy_metadata)
            probs = torch.sigmoid(outputs).cpu().numpy().reshape(-1)
            return np.vstack([1 - probs, probs]).T

    explainer = LimeTabularExplainer(metadata_array, feature_names=feature_names, class_names=["0", "1"], discretize_continuous=True)

    num_samples = metadata_array.shape[0]
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = metadata_array[start_idx:end_idx]
        exp = explainer.explain_instance(batch_data[0], predict_fn, num_features=10)
        exp.show_in_notebook()
        print(f"LIME explanation shown for batch {start_idx // batch_size + 1}.")
        save_lime_explanation(exp, start_idx // batch_size + 1)

    return exp


def explain_shap_metadata(model, metadata_array):
    """Generate SHAP explanations for tabular metadata."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(metadata_array)
    shap.summary_plot(shap_values, metadata_array)


# -------------------------
# Save & Load LIME Explanations
# -------------------------
def save_lime_explanation(exp_obj, batch_idx):
    with open(f'lime_explanation_batch_{batch_idx}.pkl', 'wb') as f:
        pickle.dump(exp_obj, f)


def load_lime_explanation(batch_idx):
    with open(f'lime_explanation_batch_{batch_idx}.pkl', 'rb') as f:
        return pickle.load(f)


# -------------------------
# Main Workflow
# -------------------------
def main():
    # Load trained model
    best_model_path = f"Model_3/best_model_3_trial_{trial.number}.pt"
    best_model_3.to(device)

    # Sample for visualization
    sample_image, sample_metadata, _ = test_dataset[0]  # Replace with your actual dataset
    attention_weights = get_vit_attention_map(best_model_3, sample_image, device)

    # Visualize attention overlay
    visualize_attention_overlay(sample_image, attention_weights)

    # LIME explanation on image
    explain_lime_image(best_model_3, sample_image, class_names=["0", "1"], device=device)

    # SHAP explanation on image (optional)
    # explain_shap_image(best_model_3, sample_image, device=device)

    # LIME explanation on metadata (optional)
    metadata_np = test_metadata_df.drop(columns='amd').values
    feature_names = test_metadata_df.drop(columns='amd').columns.tolist()
    # explain_lime_metadata(best_model_3, metadata_np, feature_names, batch_size=10)

    # SHAP explanation on metadata (optional)
    # explain_shap_metadata(best_model_3, metadata_np)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load your datasets here
    # test_dataset = ...
    # test_metadata_df = ...
    main()
