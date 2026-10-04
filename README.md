# Generating radiological descriptions based on X-ray images
## Setup
### 1. Clone the repository
```bash
git clone https://github.com/a-mx/xray
cd xray
```
### 2. Create virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate #Windows
source .venv/bin/activate #Linux
```
### 3. Install required packages
```bash
pip install uv
uv pip install -r requirements.txt
```
## Training
```bash
python -m train --model_type cnn_llm
```
## Inference
```bash
python -m utils.inference --model-type cnn_llm --ckpt ./model.pth --image image.png
```