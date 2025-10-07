Model Card

1. Model Details
Model name: HgbDetector 
Model: 1.0
Date : October 2025
Model type: CNN based (ResNet18 backbone)
Training Algorithms: Input data 
Framework: PyTorch
Architecture: Hybrid ResNet-18 and handcrafted skin features
Owner: Sutolimin Widjaja 

2. Intended Users
Primary purpose: Non-invasive Hb level estimation from lip images for preliminary screening and research use, and educational purpose 
Intended users: Healthcare researchers, data scientists, and medical screening system developers
Not intended for: Direct clinical diagnosis or medical decision-making without proper validation

3. Factors
Demographics: Works on multiple skin tones, ethnicity captured in CSV 
Device Type: differ between iPhone and Android, preprocessing required
Lighting: model may be sensitive to very dark or very bright images, refer to analyze_lighting() function that helps flag issues. (able to warn about poor quality inputs)
Image Resolution: designed for standard RGB images, ideally 224×224 pixels
Code includes skin feature extraction to normalize for color differences.
4. Training Data 
Dataset: Images with Hb levels. (provided by committees)
Input type: Lip-centered facial images
Labels: Hb values (g/dL), some with ethnicity metadata
Preprocessing: convert images to RGB and resized to 224 x 224 pixels, Handcrafted features from skin segmentation and Lab color statistics, normalization and augmentation (ColorJitter, horizontal flip)
Split: 89% training, 11% validation
5. Evaluation Data
Dataset: Separate holdout set of lip images with Hb levels
Metrics: MAE (Mean Absolute Error)
Evaluate on multiple devices and lighting conditions to ensure generalization
6. Ethical Considerations
Models should not replace blood tests.
Predictions may vary by device, lighting, and skin tone.
Ensure informed consent for image collection.
Avoid using it in clinical decisions without professional supervision.
Improvement is required for accuracy and confidence.
7. Caveats and Recommendations 
The model is device and lighting sensitive, so preprocessing and augmentation help generalization.
Use analyze_lighting() to flag poor quality images.
Optional skin feature extraction improves robustness across different devices.
Always accompany predictions with disclaimers about research only use.
