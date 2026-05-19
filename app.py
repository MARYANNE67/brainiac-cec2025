from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from transformers import ViTForImageClassification, ViTConfig

load_dotenv()

app = Flask(__name__)

# Project Title
PROJECT_TITLE = "Brainiac: Tumor Detection AI"

# Folder to store uploaded images
UPLOAD_FOLDER = "static/uploads"
HIGHLIGHT_FOLDER = "static/highlighted"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HIGHLIGHT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["HIGHLIGHT_FOLDER"] = HIGHLIGHT_FOLDER

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model configuration and weights
def load_model(model_path=None):
    model_path = model_path or os.getenv("MODEL")
    if not model_path:
        raise RuntimeError("MODEL is not set. Add the model path to your .env file.")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file was not found at: {model_path}")

    config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
    config.num_labels = 2  # Assuming binary classification

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        config=config,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Load the model globally to avoid reloading.
model = None
model_error = None
try:
    model = load_model()
except Exception as exc:
    model_error = str(exc)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    return input_tensor

def colorize_heatmap(heatmap):
    heatmap = np.clip(heatmap, 0, 1)
    red = np.clip(1.8 * heatmap, 0, 1)
    green = np.clip(1.8 * (1 - np.abs(heatmap - 0.55) * 2), 0, 1)
    blue = np.clip(1.4 * (1 - heatmap), 0, 1)
    return np.stack([red, green, blue], axis=-1)

def generate_gradcam(image_path, predicted_class, filename):
    activations = {}

    def save_activation(module, inputs, output):
        output.retain_grad()
        activations["tokens"] = output

    target_layer = model.vit.encoder.layer[-1].layernorm_before
    hook = target_layer.register_forward_hook(save_activation)

    try:
        input_tensor = preprocess_image(image_path)
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor).logits
        logits[0, predicted_class].backward()

        token_activations = activations["tokens"]
        token_gradients = token_activations.grad
        if token_gradients is None:
            return None

        # Drop the class token and reshape ViT patch tokens back into a 2D map.
        patches = token_activations.detach()[0, 1:, :]
        gradients = token_gradients.detach()[0, 1:, :]
        weights = gradients.mean(dim=0)
        cam = torch.relu((patches * weights).sum(dim=-1))

        grid_size = int(cam.numel() ** 0.5)
        if grid_size * grid_size != cam.numel():
            return None

        cam = cam.reshape(grid_size, grid_size).cpu().numpy()
        cam -= cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam /= max_value

        original = Image.open(image_path).convert("RGB")
        heatmap = Image.fromarray(np.uint8(cam * 255), mode="L").resize(original.size, Image.Resampling.BICUBIC)
        heatmap_array = np.asarray(heatmap).astype(np.float32) / 255.0
        color_heatmap = colorize_heatmap(heatmap_array)
        original_array = np.asarray(original).astype(np.float32) / 255.0
        alpha = (0.18 + 0.45 * heatmap_array)[..., None]
        overlay = original_array * (1 - alpha) + color_heatmap * alpha

        name, _ = os.path.splitext(filename)
        heatmap_filename = f"heatmap_{name}.png"
        heatmap_path = os.path.join(app.config["HIGHLIGHT_FOLDER"], heatmap_filename)
        Image.fromarray(np.uint8(np.clip(overlay, 0, 1) * 255)).save(heatmap_path)
        return heatmap_filename
    finally:
        hook.remove()

# Function to predict tumor
def predict_tumor(image_path):
    if model is None:
        raise RuntimeError(model_error or "The model is not available.")

    # Preprocess the image
    input_tensor = preprocess_image(image_path)

    # Make a prediction
    with torch.no_grad():
        outputs = model(input_tensor).logits
        probs = torch.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, dim=1)

    # Map the prediction to a label
    labels = ["no tumor", "tumor"]
    prediction_label = labels[preds.item()]
    confidence_value = confidence.item()

    return {
        "pred": prediction_label,
        "confidence": confidence_value,
        "image": image_path,
        "heatmap": None
    }

# Load the page
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if model is None:
            return render_template(
                "index.html",
                title=PROJECT_TITLE,
                filename=None,
                prediction=None,
                model_error=model_error
            ), 503

        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Process the image and predict
            prediction = predict_tumor(file_path)
            if prediction["pred"] == "tumor":
                prediction["heatmap"] = generate_gradcam(file_path, 1, filename)

            return render_template(
                "index.html",
                title=PROJECT_TITLE,
                filename=filename,
                prediction=prediction,
                model_error=model_error
            )

    return render_template(
        "index.html",
        title=PROJECT_TITLE,
        filename=None,
        prediction=None,
        model_error=model_error
    )

if __name__ == "__main__":
    app.run(debug=True)
