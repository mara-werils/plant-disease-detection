"""
PlantGuard AI — FastAPI ML Inference Server
Real ResNet50 inference + Grad-CAM + Mistral-7B LLM recommendations
"""

import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import MobileNetV2ImageProcessor, AutoModelForImageClassification
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="PlantGuard AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Disease Classes (38 from PlantVillage) ────────────────────────────
CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

CLASS_DESCRIPTIONS = {
    'Apple___Apple_scab': "Apple plant affected by Apple scab",
    'Apple___Black_rot': "Apple plant affected by Black rot",
    'Apple___Cedar_apple_rust': "Apple plant affected by Cedar apple rust",
    'Apple___healthy': "Apple plant — healthy and disease-free",
    'Blueberry___healthy': "Blueberry plant — healthy and disease-free",
    'Cherry_(including_sour)___Powdery_mildew': "Cherry plant affected by Powdery mildew",
    'Cherry_(including_sour)___healthy': "Cherry plant — healthy and disease-free",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Corn plant affected by Gray leaf spot",
    'Corn_(maize)___Common_rust_': "Corn plant affected by Common rust",
    'Corn_(maize)___Northern_Leaf_Blight': "Corn plant affected by Northern Leaf Blight",
    'Corn_(maize)___healthy': "Corn plant — healthy and disease-free",
    'Grape___Black_rot': "Grape plant affected by Black rot",
    'Grape___Esca_(Black_Measles)': "Grape plant affected by Esca (Black Measles)",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Grape plant affected by Leaf blight",
    'Grape___healthy': "Grape plant — healthy and disease-free",
    'Orange___Haunglongbing_(Citrus_greening)': "Orange plant affected by Citrus greening",
    'Peach___Bacterial_spot': "Peach plant affected by Bacterial spot",
    'Peach___healthy': "Peach plant — healthy and disease-free",
    'Pepper,_bell___Bacterial_spot': "Bell pepper plant affected by Bacterial spot",
    'Pepper,_bell___healthy': "Bell pepper plant — healthy and disease-free",
    'Potato___Early_blight': "Potato plant affected by Early blight",
    'Potato___Late_blight': "Potato plant affected by Late blight",
    'Potato___healthy': "Potato plant — healthy and disease-free",
    'Raspberry___healthy': "Raspberry plant — healthy and disease-free",
    'Soybean___healthy': "Soybean plant — healthy and disease-free",
    'Squash___Powdery_mildew': "Squash plant affected by Powdery mildew",
    'Strawberry___Leaf_scorch': "Strawberry plant affected by Leaf scorch",
    'Strawberry___healthy': "Strawberry plant — healthy and disease-free",
    'Tomato___Bacterial_spot': "Tomato plant affected by Bacterial spot",
    'Tomato___Early_blight': "Tomato plant affected by Early blight",
    'Tomato___Late_blight': "Tomato plant affected by Late blight",
    'Tomato___Leaf_Mold': "Tomato plant affected by Leaf Mold",
    'Tomato___Septoria_leaf_spot': "Tomato plant affected by Septoria leaf spot",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Tomato plant affected by Spider mites",
    'Tomato___Target_Spot': "Tomato plant affected by Target Spot",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Tomato plant affected by Yellow Leaf Curl Virus",
    'Tomato___Tomato_mosaic_virus': "Tomato plant affected by Mosaic virus",
    'Tomato___healthy': "Tomato plant — healthy and disease-free"
}

# No HF_TO_ORIGINAL mapping needed — ozair23 model outputs native Crop___Disease labels

# ─── Model Loading ─────────────────────────────────────────────────────
_model = None
_processor = None

def get_model():
    global _model, _processor
    if _model is None:
        custom_weights_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_final.pth")
        
        try:
            _processor = MobileNetV2ImageProcessor()
            
            if os.path.exists(custom_weights_path):
                print(f"🌟 Loading CUSTOM trained MobileNetV2 from {custom_weights_path}...")
                import torchvision.models as models
                import torch.nn as nn
                
                # Reconstruct the exact architecture trained on RunPod
                _model = models.mobilenet_v2()
                _model.classifier = nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(_model.last_channel, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, len(CLASSES))
                )
                
                # Load the trained weights
                state_dict = torch.load(custom_weights_path, map_location=torch.device('cpu'), weights_only=True)
                _model.load_state_dict(state_dict)
                _model.eval()
                
                # We need to attach the config object to standard torchvision model so the rest of the file works
                class DummyConfig:
                    pass
                _model.config = DummyConfig()
                _model.config.id2label = {i: name for i, name in enumerate(CLASSES)}
                
                print("✅ Custom model loaded successfully!")
                
            else:
                print("🔄 Loading MobileNetV2 (ozair23) from HuggingFace...")
                model_name = "ozair23/mobilenet_v2_1.0_224-finetuned-plantdisease"
                _model = AutoModelForImageClassification.from_pretrained(model_name)
                _model.eval()
                print("✅ HuggingFace Model loaded successfully!")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
            
    return _model, _processor


# ─── Test-Time Augmentation (TTA) for Real-World Robustness ────────────
def create_tta_crops(pil_image, crop_size=224):
    """
    Generate multiple augmented views of the image to improve prediction
    robustness on real-world (non-lab) photos. Uses:
    - Center crop (most important)
    - 4 corner crops
    - Horizontal flip of center crop
    This mimics the TenCrop strategy used in research papers.
    """
    w, h = pil_image.size
    
    # Make image square by cropping to center
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    square = pil_image.crop((left, top, left + min_dim, top + min_dim))
    
    # Resize to slightly larger than crop_size for multi-crop
    resize_dim = int(crop_size * 1.15)  # ~257 pixels
    resized = square.resize((resize_dim, resize_dim), Image.LANCZOS)
    
    crops = []
    # Center crop
    center_crop = T.CenterCrop(crop_size)(resized)
    crops.append(center_crop)
    
    # 4 corner crops
    for (t, l) in [(0, 0), (0, resize_dim - crop_size), 
                    (resize_dim - crop_size, 0), 
                    (resize_dim - crop_size, resize_dim - crop_size)]:
        corner = resized.crop((l, t, l + crop_size, t + crop_size))
        crops.append(corner)
    
    # Horizontal flip of center crop
    crops.append(T.functional.hflip(center_crop))
    
    return crops


def predict_with_tta(pil_image, model, processor):
    """
    Run prediction with Test-Time Augmentation:
    Average softmax probabilities across multiple augmented views.
    """
    crops = create_tta_crops(pil_image)
    
    all_probs = []
    for crop in crops:
        inputs = processor(images=crop, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        with torch.no_grad():
            outputs = model(pixel_values)
            # HF models return an object with logits, standard PyTorch models return the tensor directly
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = F.softmax(logits, dim=-1)[0]
        all_probs.append(probs)
    
    # Average predictions across all augmented views
    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs

# ─── Saliency Map / XAI ────────────────────────────────────────────────
def make_saliency_map(img_tensor, model):
    """Generic Input Saliency Map using PyTorch gradients"""
    model.eval()
    input_tensor = img_tensor.clone().detach().requires_grad_(True)
    # Forward pass
    outputs = model(input_tensor)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    
    # Get the class with the highest probability
    class_idx = logits[0].argmax().item()
    score = logits[0, class_idx]
    score.backward()
    
    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
    saliency = saliency[0].cpu().numpy()
    
    # Normalize
    saliency = np.maximum(saliency, 0)
    saliency = saliency - np.min(saliency)
    saliency = saliency / (np.max(saliency) + 1e-8)
    return saliency

def create_gradcam_overlay(original_img, heatmap, alpha=0.4):
    """Create Saliency overlay image, returns base64 string"""
    img_cv = np.array(original_img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_cv, 0.6, heatmap_color, alpha, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    # Encode as base64
    pil_img = Image.fromarray(superimposed_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def create_heatmap_only(heatmap):
    """Create standalone heatmap visualization, returns base64"""
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(heatmap_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ─── Image to base64 ──────────────────────────────────────────────────
def image_to_base64(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ─── API Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "MobileNetV2-PlantVillage-TTA", "classes": len(CLASSES)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a leaf image → returns prediction, confidence, top-5, Saliency heatmap
    """
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        resized = pil_image.resize((224, 224))

        model, processor = get_model()
        
        # Use center-cropped version for saliency map display
        center_crop = T.Compose([
            T.CenterCrop(min(pil_image.size)),
            T.Resize((224, 224)),
        ])(pil_image)
        inputs = processor(images=center_crop, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        
        # Saliency Map extraction (on center-cropped image)
        heatmap = make_saliency_map(pixel_values, model)
        gradcam_overlay = create_gradcam_overlay(resized, heatmap)
        heatmap_only = create_heatmap_only(heatmap)
        original_b64 = image_to_base64(resized)

        # Predict with Test-Time Augmentation for robustness
        probs = predict_with_tta(pil_image, model, processor)
            
        confidence = float(torch.max(probs))
        class_idx = int(torch.argmax(probs))
        
        # ozair23 model outputs native Crop___Disease format directly
        predicted_class = model.config.id2label[class_idx]
        description = CLASS_DESCRIPTIONS.get(predicted_class, predicted_class)
        is_healthy = "healthy" in predicted_class.lower()

        # Parse crop and disease
        parts = predicted_class.split("___")
        crop = parts[0].replace("_", " ").replace("(", "").replace(")", "").strip()
        disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
        
        # Severity estimation based on heatmap coverage
        heatmap_thresh = (heatmap > 0.5).sum() / heatmap.size
        if is_healthy:
            severity = "None"
        elif heatmap_thresh > 0.3:
            severity = "High"
        elif heatmap_thresh > 0.15:
            severity = "Moderate"
        else:
            severity = "Low"

        # Top-5 predictions
        top5_prob, top5_catid = torch.topk(probs, 5)
        top5 = []
        for i in range(top5_prob.size(0)):
            lbl = model.config.id2label[top5_catid[i].item()]
            top5.append({
                "class": lbl,
                "description": CLASS_DESCRIPTIONS.get(lbl, lbl),
                "confidence": float(top5_prob[i].item())
            })
            
        return JSONResponse({
            "prediction": {
                "class": predicted_class,
                "description": description,
                "confidence": confidence,
                "crop": crop,
                "disease": disease,
                "isHealthy": is_healthy,
                "severity": severity,
            },
            "top5": top5,
            "gradcam": {
                "overlay": gradcam_overlay,
                "heatmap": heatmap_only,
                "original": original_b64,
            },
            "allProbabilities": {
                model.config.id2label[i]: float(probs[i]) for i in range(len(probs))
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RecommendRequest(BaseModel):
    prediction: str
    isHealthy: bool = False
    context: dict = None


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    """
    Get LLM-powered plant care recommendations via Mistral-7B
    """
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    if not HF_TOKEN:
        print("⚠️ Warning: No HF_TOKEN provided. Using fallback deterministic recommendations.")
        # Return structured fallback recommendations
        return JSONResponse({
            "source": "fallback",
            "recommendations": get_fallback_recommendations(req.prediction, req.isHealthy)
        })

    try:
        from huggingface_hub import InferenceClient
        # Standard HuggingFace Inference API
        client = InferenceClient(api_key=HF_TOKEN)

        condition_type = "healthy" if req.isHealthy else "diseased"
        
        # Multi-Modal Context Injection
        env_context = ""
        if req.context:
            temp = req.context.get("temperature", "Unknown")
            hum = req.context.get("humidity", "Unknown")
            soil = req.context.get("soilType", "Unknown")
            env_context = f"Current Meteorological Context: {temp}°C, {hum}% Humidity, {soil} Soil Type."

        prompt = f"""
        You are an expert, professional agricultural pathologist and agronomist.
        The computer vision system has diagnosed the plant specimen as: **{req.prediction}** (Condition: {condition_type}).
        {env_context}
        
        Please provide a highly professional, academic, and actionable treatment protocol or preventive care strategy.
        Critically: You MUST adapt your recommendations based on the Current Meteorological Context provided above. Explain WHY the current temperature/humidity/soil affects the disease spread or treatment choice.
        Do not use filler phrases (e.g., "Here is your advice", "Of course"). Go straight into the protocol.
        Format your response as a numbered list with detailed, scientifically sound points.
        """

        completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[
                {"role": "system", "content": "You are a concise, professional agricultural expert and plant pathologist. Respond with direct, academic care instructions only. No conversational filler."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        response = completion.choices[0].message["content"]
        return JSONResponse({
            "source": "mistral-7b",
            "recommendations": response
        })

    except Exception as e:
        print(f"❌ Mistral API Error: {str(e)}")
        return JSONResponse({
            "source": "fallback",
            "recommendations": get_fallback_recommendations(req.prediction, req.isHealthy),
            "error": str(e)
        })


def get_fallback_recommendations(prediction: str, is_healthy: bool) -> str:
    """Structured fallback when no API key available"""
    if is_healthy:
        return """1. **Continue Current Care Routine** — Your plant appears healthy. Maintain consistent watering schedule, proper sunlight exposure, and soil nutrition to preserve this condition.

2. **Preventive Monitoring** — Regularly inspect leaves (both upper and lower surfaces) for early signs of discoloration, spots, or unusual patterns. Early detection is key.

3. **Soil & Nutrition Management** — Ensure balanced NPK fertilization and maintain soil pH between 6.0-7.0. Consider periodic soil testing to optimize plant nutrition.

4. **Environmental Control** — Maintain adequate spacing between plants for proper air circulation. Avoid overhead watering to reduce moisture on leaf surfaces."""
    else:
        disease_name = prediction.split("—")[-1].strip() if "—" in prediction else prediction
        return f"""1. **Immediate Isolation** — Separate affected plants from healthy ones to prevent disease spread. Remove and safely dispose of severely infected leaves or plant parts.

2. **Targeted Treatment for {disease_name}** — Apply appropriate fungicide or bactericide based on the specific pathogen identified. Consult local agricultural extension services for region-specific treatment protocols.

3. **Environmental Modification** — Adjust watering practices to avoid leaf wetness. Ensure proper drainage and air circulation around affected plants. Consider adjusting planting density.

4. **Integrated Disease Management** — Implement crop rotation strategies, use disease-resistant varieties where available, and maintain proper sanitation of gardening tools to prevent reinfection.

5. **Long-term Prevention** — After treatment, strengthen plant immunity through balanced nutrition (avoid excess nitrogen), proper pruning, and mulching to prevent soil-borne pathogen splash."""


# ─── Startup ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Pre-load model on server start"""
    print("🌿 PlantGuard AI API starting...")
    get_model()
    print("🚀 Server ready!")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
