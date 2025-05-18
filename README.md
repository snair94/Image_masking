AI Challenge Brief 2 – Plus Also Studios
Image Masking 
This project is a semantic segmentation web app that allows users to upload a product image, provide a text prompt to identify an item within the image, and apply a solid-colored mask to the segmented region. The user can download the masked image or the binary mask.
Built using Gradio and the CLIPSeg model by CIDAS.
________________________________________Features
Upload any product image (clothing, cans, shoes, etc.)
Input a prompt like "shirt", "label", "logo", "bottle" to mask a specific part of the image
Select any color for the mask using a color picker
View and download the masked image or the binary mask
Simple, clean UI powered by Gradio
________________________________________Project Structure
clipseg-masker/
│
├── app.py              # Main application with Gradio interface
├── requirements.txt    # Python dependencies
└── README.md           # You're here!
________________________________________Setup Instructions
1. Clone the Repository
2. Create a Python Virtual Environment (Optional but Recommended)
3. Install Dependencies
pip install -r requirements.txt

Run this on hugging-face -
Open the displayed URL in your browser. (CLIPSeg Image Masking)
This will start a local Gradio interface.
Upload an image, write a prompt, and select the color you want the mask in
View and download your AI-generated image.
________________________________________Dependencies
All dependencies are listed in requirements.txt. Key packages include:
transformers
gradio
Pillow
torch
matplotlib
opencv-python
Install all with:
pip install -r requirements.txt
________________________________________Testing (Manual)
Since this is a front-end tool:
Launch the app.
Upload a sample product image.
Use prompts like "bottle", "label", "can", "shirt" to test segmentation.
Try with various mask colors.
Confirm the output image shows a correctly colored overlay and that it is downloadable.
________________________________________Notes
The segmentation model is zero-shot, meaning it works without needing a labeled dataset.
Ensure the prompt is accurate and clearly refers to a visual object in the image for best results.
Mask quality can vary depending on image complexity and prompt specificity.
Use images of .png or .jpeg format
________________________________________Future Improvements
Provide options to download the binary mask separately.
Add mask transparency controls.
Add support for multi-label prompts (e.g., "shirt and jeans").
Export results in .png with alpha channel for design workflows.
________________________________________GitHub: snair94/Image_masking: This project is a semantic segmentation web app that allows users to upload a product image, provide a text prompt to identify an item within the image, and apply a solid-colored mask to the segmented region.
Hugging-face stage link: CLIPSeg Image Masking
