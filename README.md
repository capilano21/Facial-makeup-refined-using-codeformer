# Makeup Application and Face Restoration Pipeline

This repository includes scripts for applying makeup effects to images and then refining them using face restoration techniques. It consists of:

- **`makeup.py`**: Applies makeup effects to an image.
- **`codeformer.py`**: Enhances the makeup-applied image using the CodeFormer model.

## Prerequisites

Make sure you have the following software and libraries installed:

### Python Version

- Python 3.7+ is required.

### Install Dependencies

Install the necessary libraries using `pip`:

```bash
pip install opencv-python torch numpy Pillow face_recognition basicsr facelib torchvision
