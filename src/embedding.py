import torch
from torchvision import models
from torchvision import transforms
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def load_model(model_name: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Loads a pre-trained CNN/ViT model and prepares it for feature extraction."""
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final classification layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
    # Add support for Vision Transformer (ViT) or other models here
    elif model_name == 'vit-base':
        # Example for ViT
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # Assuming the ViT structure, we take the output before the final classification head
        model.heads = torch.nn.Identity()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.eval()
    model.to(device)
    return model

def generate_embeddings(source_dir: str, model, recursive: bool = True, batch_size: int = 64):
    """Processes images in a directory to generate embeddings."""
    device = next(model.parameters()).device
    
    # Standard image transformation for pre-trained ImageNet models
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = []
    # (File system logic to collect paths based on recursive flag)

    all_embeddings = []
    metadata_list = []

    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing Batches"):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        current_metadata = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                batch_tensors.append(transform(img))
                current_metadata.append({'path': path, 'size': os.path.getsize(path)})
            except Exception:
                # Skip corrupted or non-image files
                continue

        if not batch_tensors: continue

        inputs = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            outputs = model(inputs).squeeze() # Generate the embedding vectors
        
        # Normalize the embeddings (Crucial for Cosine Similarity)
        normalized_embeddings = outputs / outputs.norm(dim=1, keepdim=True)
        
        all_embeddings.append(normalized_embeddings.cpu().numpy())
        metadata_list.extend(current_metadata)

    return metadata_list, np.vstack(all_embeddings)
          
