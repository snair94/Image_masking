import gradio as gr
from PIL import Image
import numpy as np
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from matplotlib.colors import to_rgb
import re
import cv2

# Load model
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

def parse_color(color_str):
    """
    Converts a color string (hex, name, or rgba(...)) to an RGB tuple.
    """
    try:
        if isinstance(color_str, str):
            if color_str.startswith("rgba("):
                # Extract the 3 RGB components
                numbers = list(map(float, re.findall(r"[\d.]+", color_str)))
                if len(numbers) >= 3:
                    r, g, b = numbers[:3]
                    return int(r), int(g), int(b)
            else:
                # Use named or hex color
                return tuple(int(255 * c) for c in to_rgb(color_str))
    except Exception:
        pass
    raise ValueError(f"Invalid color format: {color_str}. Use hex like '#ff0000', color name like 'red', or rgba format.")

def apply_mask(image: Image.Image, prompt: str, color: str) -> Image.Image:
    # Process the input image and prompt
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    outputs = model(**inputs)
    preds = outputs.logits[0]

    # Get the binary mask from predictions
    mask = preds.sigmoid().detach().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)

    # Convert image to RGBA
    image_np = np.array(image.convert("RGBA"))

    # Resize mask to match image size
    mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
    mask_3d = np.stack([mask_resized] * 4, axis=-1)  # Extend mask to 3D

    # Convert the color string to an RGB tuple
    color_rgb = parse_color(color)
    overlay_color = np.array([*color_rgb, 128], dtype=np.uint8)  # RGBA with alpha 128

    # Create an overlay with the selected color
    overlay = np.zeros_like(image_np, dtype=np.uint8)
    overlay[:] = overlay_color

    # Apply the mask to the image
    masked_image = np.where(mask_3d == 1, overlay, image_np)
    return Image.fromarray(masked_image)

# Gradio Interface
iface = gr.Interface(
    fn=apply_mask,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(label="Segmentation Prompt", placeholder="e.g., helmet, road, sky"),
        gr.ColorPicker(label="Mask Color", value="#ff0000")
    ],
    outputs=gr.Image(type="pil", label="Segmented Image"),
    title="CLIPSeg Image Masking",
    description="Upload an image, input a prompt (e.g., 'person', 'sky'), and pick a mask color."
)

iface.launch()
