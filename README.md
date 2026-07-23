# Building Segmentation from Aerial Imagery

Semantic segmentation model that identifies buildings in aerial imagery using U-Net with a ResNet34 encoder. Achieved **91% IoU** on validation data.

## Results

| Dataset | Images | Validation IoU |
|---|---|---|
| Dubai Aerial Imagery | 72 | 27% |
| Drone Aerial Imagery | 400 | **91%** |

## Setup

```bash
git clone https://github.com/KalanM-P/building-segmentation.git
cd building-segmentation
pip install -r requirements.txt
```

## Usage

```bash
python src/train_drone.py
python src/visualize_drone_predictions.py
```

## Datasets

- [Dubai Aerial Imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)
- [Drone Segmentation Dataset](https://www.kaggle.com/datasets/santurini/semantic-segmentation-drone-dataset)
