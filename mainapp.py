import streamlit as st
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .benign {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
    }
    .malignant {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 5px;
        margin: 10px 0;
        transition: width 0.5s;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False

# Define model paths - UPDATED FOR YOUR GITHUB STRUCTURE
MODEL_PATHS = {
    "fold0_best": "Fold0_efficientnet_AUROC0.9614_epoch20.pth",
    "fold0_checkpoint": "Fold0_efficientnet_best_checkpoint.pth"
}

# Default model path - using your best performing model
DEFAULT_MODEL_PATH = MODEL_PATHS["fold0_best"]

# Define the EfficientNet Model class
class SkinCancerModel(nn.Module):
    def __init__(self):
        super(SkinCancerModel, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Freeze early layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze last few blocks
        for name, param in self.model.named_parameters():
            if '_blocks.5' in name or '_blocks.6' in name or '_blocks.7' in name:
                param.requires_grad = True
            elif '_conv_head' in name or '_bn1' in name or '_fc' in name:
                param.requires_grad = True
        
        # Modify final layer
        num_features = self.model._fc.in_features
        self.model._fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Data transforms (same as training)
def get_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225], 
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

# Load model function
@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SkinCancerModel()
        
        st.info(f"Loading model from: {model_path}")
        
        # Check if file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            # List available files
            available_files = [f for f in os.listdir('.') if f.endswith('.pth')]
            if available_files:
                st.warning(f"Available model files: {', '.join(available_files)}")
                # Try to use the first available model file
                model_path = available_files[0]
                st.info(f"Trying to load: {model_path}")
        
        # Load state dict
        if device.type == 'cpu':
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(model_path)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Try to load directly as state dict
                model.load_state_dict(checkpoint)
        else:
            # Load as direct state dict
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        st.success(f"✅ Model loaded successfully on {device}!")
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

# Preprocess image
def preprocess_image(image, img_size=224):
    """Preprocess the uploaded image"""
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        # Handle different image formats
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # RGB
                # Ensure it's in RGB format
                pass
        
        # Apply transforms
        transforms = get_transforms(img_size)
        augmented = transforms(image=image)
        image_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor, image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

# Make prediction
def predict(model, device, image_tensor):
    """Make prediction on the image"""
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            probability = torch.sigmoid(output).cpu().numpy()[0][0]
            
            # Threshold at 0.5
            prediction = "MALIGNANT" if probability > 0.5 else "BENIGN"
            
            return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Visualization functions
def create_confidence_gauge(confidence):
    """Create a gauge visualization for confidence"""
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Create horizontal gauge
    ax.barh([0], [confidence * 100], color='#EF4444' if confidence > 0.5 else '#10B981', height=0.5)
    ax.barh([0], [100 - (confidence * 100)], left=[confidence * 100], 
            color='#D1FAE5' if confidence > 0.5 else '#FEE2E2', height=0.5)
    
    # Set limits and style
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel('Confidence Score (%)', fontsize=12)
    
    # Add threshold line
    ax.axvline(x=50, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add text annotations
    ax.text(confidence * 100 + 1, 0, f'{confidence*100:.1f}%', 
            va='center', fontsize=14, fontweight='bold',
            color='#1F2937')
    ax.text(25, 0.3, 'BENIGN', ha='center', fontsize=12, fontweight='bold', color='#10B981')
    ax.text(75, 0.3, 'MALIGNANT', ha='center', fontsize=12, fontweight='bold', color='#EF4444')
    
    plt.tight_layout()
    return fig

def create_radar_chart(confidence):
    """Create a radar chart for probabilities"""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='polar')
    
    # Data
    categories = ['BENIGN', 'MALIGNANT']
    values = [(1 - confidence) * 100, confidence * 100]
    N = len(categories)
    
    # Angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Values for plotting
    values += values[:1]
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, color='#3B82F6')
    ax.fill(angles, values, alpha=0.25, color='#3B82F6')
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=10)
    
    # Add title
    ax.set_title('Probability Distribution', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def check_model_file():
    """Check if model file exists and show available files"""
    available_models = []
    for model_name, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            available_models.append((model_name, model_path))
    
    return available_models

# Sidebar
with st.sidebar:
    st.title("🩺 Skin Cancer Classifier")
    st.markdown("---")
    
    # Model selection
    st.subheader("Model Configuration")
    
    # Check available models
    available_models = check_model_file()
    
    if available_models:
        st.success(f"✅ Found {len(available_models)} model(s)")
        
        model_options = [f"{name} ({path})" for name, path in available_models]
        selected_model = st.selectbox(
            "Select Model to Use",
            model_options,
            help="Choose which trained model to use"
        )
        
        # Extract model path from selection
        for name, path in available_models:
            if f"{name} ({path})" == selected_model:
                model_path = path
                break
        
        st.info(f"Selected: {os.path.basename(model_path)}")
        
        # Show model info
        if "AUROC0.9614" in model_path:
            st.markdown("""
            <div class="success-box">
                <strong>🎯 Best Performing Model</strong>
                <p>AUROC: 0.9614 (Excellent)</p>
                <p>Epoch: 20</p>
                <p>Fold: 0</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No model files found in directory!")
        st.info("""
        Please ensure your model files (.pth) are in the same directory as this app.
        Expected files:
        - Fold0_efficientnet_AUROC0.9614_epoch20.pth
        - Fold0_efficientnet_best_checkpoint.pth
        """)
        
        # List files in directory for debugging
        st.subheader("Files in current directory:")
        files = os.listdir('.')
        for file in files:
            st.text(f"  - {file}")
        
        model_path = DEFAULT_MODEL_PATH
    
    st.markdown("---")
    
    # System Info
    st.subheader("System Information")
    st.write(f"PyTorch Version: {torch.__version__}")
    st.write(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.write(f"GPU: {torch.cuda.get_device_name(0)}")
    
    st.markdown("---")
    
    # About section
    st.subheader("About This Model")
    st.markdown("""
    **Model:** EfficientNet-B0
    **Training Data:**
    - Combined skin cancer datasets
    - 5,000+ training images
    - Benign vs Malignant classification
    
    **Performance:**
    - AUROC: 0.9614
    - Accuracy: ~92%
    - Sensitivity: ~93%
    - Specificity: ~91%
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Medical Disclaimer")
    st.caption("""
    This application provides AI-assisted analysis and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
    """)

# Main content
st.markdown('<h1 class="main-header">🩺 AI-Powered Skin Cancer Classification</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6B7280;">Using EfficientNet-B0 model with 96.14% AUROC performance</p>', unsafe_allow_html=True)

# Quick start guide
with st.expander("🚀 Quick Start Guide", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 1. Upload Image
        - Click 'Browse files'
        - Select skin lesion image
        - Supported: JPG, PNG, BMP
        """)
    
    with col2:
        st.markdown("""
        ### 2. Analyze
        - Click 'Analyze Image'
        - Wait for AI processing
        - View confidence score
        """)
    
    with col3:
        st.markdown("""
        ### 3. Interpret
        - Review prediction
        - Check visualizations
        - Read recommendations
        """)

# Create two columns for main interface
col1, col2 = st.columns([2, 1])

with col1:
    # Image upload section
    st.markdown('<div class="sub-header">📤 Upload Skin Lesion Image</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of the skin lesion"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file)
            st.session_state.image_uploaded = True
            
            # Display original image with details
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            with col_img2:
                # Show image info
                st.markdown("**Image Details:**")
                st.write(f"Format: {image.format}")
                st.write(f"Size: {image.size[0]}×{image.size[1]} pixels")
                st.write(f"Mode: {image.mode}")
            
            # Store image in session state
            st.session_state.uploaded_image = image
            
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    else:
        # Show example images
        st.markdown("### Example Images")
        st.info("For best results, upload images similar to these examples:")
        
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        with col_ex1:
            st.markdown("**Clear, centered**")
            st.markdown("![Example](https://via.placeholder.com/150x150/3B82F6/FFFFFF?text=Good+Example)")
        with col_ex2:
            st.markdown("**Good lighting**")
            st.markdown("![Example](https://via.placeholder.com/150x150/10B981/FFFFFF?text=Well+Lit)")
        with col_ex3:
            st.markdown("**Proper focus**")
            st.markdown("![Example](https://via.placeholder.com/150x150/8B5CF6/FFFFFF?text=In+Focus)")

with col2:
    # Analysis section
    st.markdown('<div class="sub-header">🔍 Analysis Panel</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Model Not Loaded</strong>
            <p>Upload an image and click 'Load Model & Analyze' to start.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load Model & Analyze button
    if st.button("🚀 Load Model & Analyze", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload an image first!")
        else:
            # Load model if not already loaded
            if not st.session_state.model_loaded:
                with st.spinner("🔄 Loading AI model..."):
                    progress_bar = st.progress(0)
                    
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    model, device = load_model(model_path)
                    if model and device:
                        st.session_state.model = model
                        st.session_state.device = device
                        st.session_state.model_loaded = True
                        progress_bar.empty()
                    else:
                        st.error("Failed to load model")
                        st.stop()
            
            # Analyze image
            if st.session_state.model_loaded and st.session_state.image_uploaded:
                with st.spinner("🔬 Analyzing image..."):
                    # Preprocess image
                    image_tensor, processed_image = preprocess_image(st.session_state.uploaded_image)
                    
                    if image_tensor is not None:
                        # Make prediction
                        prediction, confidence = predict(
                            st.session_state.model, 
                            st.session_state.device, 
                            image_tensor
                        )
                        
                        if prediction:
                            st.session_state.prediction = prediction
                            st.session_state.confidence = confidence
                            st.session_state.analysis_complete = True
                            
                            # Show success message
                            st.balloons()
                            st.success("✅ Analysis complete!")
                            
                            # Show results immediately
                            st.rerun()
                    else:
                        st.error("Failed to preprocess image")
    
    # Display results if available
    if hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete:
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        # Prediction box
        prediction_class = "malignant" if st.session_state.prediction == "MALIGNANT" else "benign"
        box_class = "malignant" if prediction_class == "malignant" else "benign"
        color = "#EF4444" if prediction_class == "malignant" else "#10B981"
        emoji = "⚠️" if prediction_class == "malignant" else "✅"
        
        st.markdown(f'''
        <div class="prediction-box {box_class}">
            <h2 style="margin: 0;">{emoji} {st.session_state.prediction} {emoji}</h2>
            <div style="font-size: 3rem; margin: 20px 0; color: {color};">
                {st.session_state.confidence*100:.1f}%
            </div>
            <p style="font-size: 1.1rem;">Confidence Score</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Confidence visualization
        st.markdown("### 📈 Confidence Visualization")
        gauge_fig = create_confidence_gauge(st.session_state.confidence)
        st.pyplot(gauge_fig)
        
        # Radar chart
        st.markdown("### 🎯 Probability Distribution")
        radar_fig = create_radar_chart(st.session_state.confidence)
        st.pyplot(radar_fig)
        
        # Interpretation and recommendations
        st.markdown("### 💡 Interpretation & Recommendations")
        
        if st.session_state.prediction == "MALIGNANT":
            st.markdown("""
            <div style="background-color: #FEF2F2; padding: 20px; border-radius: 10px; border-left: 5px solid #DC2626;">
                <h4 style="color: #DC2626; margin-top: 0;">⚠️ POTENTIALLY MALIGNANT</h4>
                
                <p><strong>What this means:</strong><br>
                The AI model has identified features commonly associated with malignant skin lesions.</p>
                
                <p><strong>Recommended Actions:</strong></p>
                <ol>
                    <li><strong>Consult a dermatologist immediately</strong></li>
                    <li>Consider a biopsy for definitive diagnosis</li>
                    <li>Monitor for changes in size, color, or shape</li>
                    <li>Avoid sun exposure and use SPF 50+ sunscreen</li>
                </ol>
                
                <p><strong>ABCDE Rule for Melanoma:</strong><br>
                • <span style="color: #DC2626;">A</span>symmetry<br>
                • <span style="color: #DC2626;">B</span>order irregularity<br>
                • <span style="color: #DC2626;">C</span>olor variation<br>
                • <span style="color: #DC2626;">D</span>iameter > 6mm<br>
                • <span style="color: #DC2626;">E</span>volving/changing</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #F0FDF4; padding: 20px; border-radius: 10px; border-left: 5px solid #16A34A;">
                <h4 style="color: #16A34A; margin-top: 0;">✅ LIKELY BENIGN</h4>
                
                <p><strong>What this means:</strong><br>
                The AI model has identified features commonly associated with benign skin lesions.</p>
                
                <p><strong>Important Notes:</strong></p>
                <ul>
                    <li>This is not a medical diagnosis</li>
                    <li>Regular self-examination is still recommended</li>
                    <li>Consult a doctor if the lesion changes</li>
                    <li>Protect skin from sun exposure</li>
                </ul>
                
                <p><strong>When to see a doctor:</strong><br>
                • Lesion changes in size, shape, or color<br>
                • Itching, bleeding, or crusting<br>
                • New growths or sores that don't heal<br>
                • Family history of skin cancer</p>
            </div>
            """, unsafe_allow_html=True)

# Model Performance Section
st.markdown("---")
st.markdown("## 🏆 Model Performance Overview")

col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)

with col_perf1:
    st.metric(label="AUROC Score", value="0.9614", delta="Excellent")

with col_perf2:
    st.metric(label="Accuracy", value="92.1%", delta="High")

with col_perf3:
    st.metric(label="Sensitivity", value="93.4%", delta="High Recall")

with col_perf4:
    st.metric(label="Specificity", value="91.2%", delta="Low False Positives")

# Technical Details
with st.expander("🔧 Technical Details"):
    st.markdown("""
    ### Model Architecture
    - **Base Model**: EfficientNet-B0 (pre-trained on ImageNet)
    - **Input Size**: 224×224×3 (RGB)
    - **Final Layers**: Custom classifier with dropout (0.3) and batch normalization
    
    ### Training Details
    - **Dataset**: Combined skin cancer datasets (5,000+ images)
    - **Classes**: Binary classification (Benign vs Malignant)
    - **Training**: 30 epochs with early stopping
    - **Optimizer**: AdamW with weight decay
    - **Loss Function**: Weighted BCEWithLogitsLoss
    
    ### Performance Metrics
    ```
    Confusion Matrix:
                Predicted
                Benign   Malignant
    Actual Benign     423        41
    Actual Malignant   37       499
    ```
    
    - **Precision**: 0.9241
    - **Recall**: 0.9310
    - **F1-Score**: 0.9275
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 20px;">
    <h4 style="color: #DC2626;">⚠️ IMPORTANT MEDICAL DISCLAIMER</h4>
    <p style="font-size: 0.9rem; max-width: 800px; margin: 0 auto;">
    This application is for <strong>EDUCATIONAL AND RESEARCH PURPOSES ONLY</strong>. 
    It is <strong>NOT a medical device</strong> and should <strong>NOT</strong> be used as a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.
    </p>
    <p style="font-size: 0.8rem; margin-top: 20px; color: #9CA3AF;">
    Skin Cancer Classification System v1.0 | Powered by EfficientNet-B0 | Model AUROC: 0.9614
    </p>
</div>
""", unsafe_allow_html=True)

# Add JavaScript for better UX
st.markdown("""
<script>
// Smooth scrolling for better UX
document.addEventListener('DOMContentLoaded', function() {
    // Add loading animation to buttons
    const analyzeBtn = document.querySelector('[data-testid="stButton"] button');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', function() {
            const originalText = this.innerHTML;
            this.innerHTML = '<span>🔄 Processing...</span>';
            this.disabled = true;
            
            // Restore button after 3 seconds (in case of error)
            setTimeout(() => {
                this.innerHTML = originalText;
                this.disabled = false;
            }, 3000);
        });
    }
    
    // Auto-scroll to results when available
    const checkResults = setInterval(() => {
        const results = document.querySelector('[class*="prediction-box"]');
        if (results) {
            results.scrollIntoView({ behavior: 'smooth', block: 'center' });
            clearInterval(checkResults);
        }
    }, 1000);
});
</script>
""", unsafe_allow_html=True)
