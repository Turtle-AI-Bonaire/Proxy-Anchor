# Proxy Anchor Loss for Turtle Recognition

Turtle AI implementation of **Proxy Anchor Loss for Deep Metric Learning** adapted for sea turtle facial recognition. This repository is a specialized fork of the original [Proxy Anchor Loss](https://arxiv.org/abs/2003.13911) implementation, customized for the Tag-a-Turtle project in collaboration with Bonaire Turtles.

**Proxy-Anchor Loss** provides state-of-the-art performance for turtle face recognition with fast convergence, enabling accurate identification of individual sea turtles from facial photographs.

This repository provides source code for training turtle recognition models and pretrained weights optimized for turtle facial features.

### Training Performance on Turtle Dataset

The adapted implementation shows significant improvements in turtle recognition accuracy compared to feature matching approaches like LightGlue, showing around 74% accuracy at R@5 compared to around 50% for LightGlue.

## Project Context

This implementation is part of the **Tag-a-Turtle** project, which aims to create a non-invasive system for identifying individual sea turtles by their facial features. The system helps researchers track turtles over time without needing physical tags, supporting conservation efforts by Bonaire Turtles organization.

### Key Adaptations for Turtle Recognition

- **Underwater Image Processing**: Enhanced preprocessing for underwater photography conditions
- **Turtle-Specific Augmentations**: Custom data augmentation strategies for turtle facial features
- **Small Dataset Optimization**: Techniques for working with limited turtle image datasets
- **Face Detection Integration**: Seamless integration with turtle face detection models (YOLO)

## Requirements

- Python3
- PyTorch (> 1.0)
- NumPy
- tqdm
- wandb
- OpenCV (for turtle-specific preprocessing)
- [Pytorch-Metric-Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

## Dataset Structure

The turtle dataset should follow this structure:

```
turtle_dataset/
├── train/
│   ├── turtle_001/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── turtle_002/
│   └── ...
└── test/
    ├── turtle_101/
    └── ...
```

### Data Preparation

1. **Turtle Face Detection**: Use YOLO to detect and crop turtle faces from full images
2. **Image Preprocessing**: Apply underwater-specific preprocessing (color correction, contrast enhancement)
3. **Quality Filtering**: Remove blurry or poorly oriented images
4. **Train/Test Split**: Maintain turtle identity separation between training and testing sets

## Training Turtle Recognition Model

### Basic Training Command

```bash
python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model resnet50 \
                --embedding-size 512 \
                --batch-size 64 \
                --lr 1e-4 \
                --dataset combo \
                --warm 2 \
                --bn-freeze 1 \
                --lr-decay-step 10
```

### Turtle-Specific Training Parameters

- **`--turtle-augment`**: Enable turtle-specific data augmentation
- **`--underwater-preprocess`**: Apply underwater image preprocessing
- **`--min-images-per-turtle`**: Minimum images required per turtle identity
- **`--face-detection-model`**: Path to YOLO turtle face detection model

### Recommended Training Configuration

For optimal turtle recognition performance:

```bash
python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model resnet101 \
                --embedding-size 512 \
                --batch-size 170 \
                --lr 1e-4 \
                --dataset combo \
                --warm 2 \
                --bn-freeze 1 \
                --lr-decay-step 10 \
                --epochs 140
```

## Evaluating Turtle Recognition

```bash
python evaluate.py --gpu-id 0 \
                   --batch-size 64 \
                   --model [model with which the checkpoint was trained] \
                   --embedding-size [batch size with which the checkpoint was trained] \
                   --dataset bon \
                   --resume ./models/saved_checkpoint_name.pth 
```

## Integration with Tag-a-Turtle Pipeline

This model integrates seamlessly with the Tag-a-Turtle recognition pipeline:

```python
# Example integration
from turtle_recognition import TurtleProxyAnchor

model = TurtleProxyAnchor(
    model_path='./models/turtle_resnet50_best.pth',
    embedding_size=512
)

# Generate embeddings for turtle faces
turtle_embedding = model.extract_embedding(turtle_face_image)
```

## Turtle-Specific Features

### Data Augmentation
- **Minimal data augmentation to avoid losing crucial detail**
- **Embedding aggregation for improved accuracy**
- **Dataset classes for loading relevant turtle individual datasets**

## Field Deployment

### Google Colab Integration
This model is designed to work within Google Colab for field researchers:

```python
# Load model in Colab
!pip install turtle-recognition-requirements.txt
from turtle_proxy_anchor import load_model

model = load_model('turtle_resnet50_best.pth')
results = model.identify_turtle(query_image, gallery_embeddings)
```

### Performance Optimization
- **Model quantization** for faster inference
- **Batch processing** for multiple turtle images
- **Embedding caching** to avoid recomputation
- **GPU memory optimization** for large datasets

## Contributors

| Contributor | Role | Primary Contributions |
|-------------|------|----------------------|
| Daniil Rayu | ML Engineer | Proxy Anchor adaptation, model training, hyperparameter tuning, dataset adaptation |

## Acknowledgements

This implementation is built upon:

- [Original Proxy Anchor Loss implementation](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020)
- [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
- Sea Turtle Conservation Bonaire organization for providing the turtle dataset

## Related Projects

- [Tag-a-Turtle Main Repository](https://github.com/Turtle-AI-Bonaire/inference_pipeline) - Complete turtle recognition pipeline
- [Turtle Face Detection](https://github.com/Turtle-AI-Bonaire/yolo-turtle-detection) - YOLO-based turtle face detection

## Contact

For questions about the turtle recognition implementation:
- Email: deltafontys@gmail.com
- Project Repository: https://github.com/Turtle-AI-Bonaire/Proxy-Anchor
- Main Pipeline: https://github.com/Turtle-AI-Bonaire/inference_pipeline
