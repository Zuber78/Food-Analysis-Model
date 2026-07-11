# Local YOLO Food Nutrition Analyzer

A full-stack app that uploads a food image, runs a local YOLO detector, and estimates nutrition from a local food nutrition table.

No Gemini or external vision API is used.

## Setup

1. Install Node dependencies:
   ```bash
   npm install
   ```
2. Use the project Python environment:
   ```bash
   .\.venv\Scripts\activate
   ```
3. Copy environment settings:
   ```bash
   copy .env.example .env
   ```
4. Set the Python and model paths in `.env`:
   ```env
   PYTHON_PATH=D:\Food Analysis Model\.venv\Scripts\python.exe
   FOOD_MODEL_PATH=D:\Food Analysis Model\models\indian-food-yolov11.pt
   SECONDARY_FOOD_MODEL_PATH=D:\Food Analysis Model\models\foodseg103-yolov8.pt
   PORT=3002
   ```
5. Start the app:
   ```bash
   npm run dev
   ```
6. Open http://localhost:5173

## Hugging Face Food Models

The project is configured by default for this Indian-food YOLO model:

https://huggingface.co/2meirl/indian-food-recognize-yolov11

Download the Indian-food weights:

```powershell
mkdir models
Invoke-WebRequest -Uri "https://huggingface.co/2meirl/indian-food-recognize-yolov11/resolve/main/best1.pt" -OutFile "models\indian-food-yolov11.pt"
```

The Indian model detects 30 classes including `Dal`, `WhiteRice`, `Biryani`, `Dosa`, `Idli`, `Samosa`, `PalakPaneer`, and `RajmaCurry`. Use this model for Indian meals like rajma chawal.

The broader 104-class FoodSeg103 model is also available:

https://huggingface.co/magnusdtd/yolov8-foodseg103

Download the FoodSeg103 weights:

```powershell
mkdir models
curl.exe -L "https://huggingface.co/magnusdtd/yolov8-foodseg103/resolve/main/yolov8_foodseg103.pt" -o "models\foodseg103-yolov8.pt"
```

FoodSeg103 exposes 104 classes and is useful for general foods like pizza, hamburg, rice, noodles, fruit, vegetables, and desserts. It does not know Indian dish names like `RajmaCurry`, so it can misclassify Indian meals.

The app can use both models together:

- `FOOD_MODEL_PATH`: primary Indian model for rajma, dal, rice, biryani, dosa, and similar meals.
- `SECONDARY_FOOD_MODEL_PATH`: general model used as an override for foods such as burger, pizza, fries, pasta, noodles, bread, and desserts.

If you want only one model, remove `SECONDARY_FOOD_MODEL_PATH` from `.env`.

## Training Your Own Food YOLO Model

YOLO needs object-detection labels, not only class folders. Every image needs a matching `.txt` file with bounding boxes.

Use this folder layout:

```text
datasets/food/
  data.yaml
  images/
    train/
    val/
  labels/
    train/
    val/
```

Each label row must use YOLO format:

```text
class_id x_center y_center width height
```

All coordinates are normalized from `0` to `1`.

Start from the template:

```bash
copy datasets\food\data.yaml.example datasets\food\data.yaml
```

Then train:

```bash
.\.venv\Scripts\python.exe train_food_yolo.py --data datasets/food/data.yaml --epochs 50 --imgsz 640 --batch 8 --device cpu
```

After training, set `.env` to the trained weights:

```env
FOOD_MODEL_PATH=D:\Food Analysis Model\runs\detect\food-yolo\weights\best.pt
```

Restart the dev server.

## Nutrition Mapping

YOLO detects food names. Nutrition is estimated from `food_nutrition.json`.

The model class names in `datasets/food/data.yaml` should match keys in `food_nutrition.json`, for example:

```yaml
names:
  0: dal
  1: roti
  2: rice
  3: burger
```

```json
{
  "dal": {
    "weight_g": 180,
    "calories": 210,
    "protein_g": 12,
    "carbs_g": 28,
    "fat_g": 5,
    "fiber_g": 7,
    "healthy": true
  }
}
```

## Current Limitation

`yolov8n.pt` is only a starter base model. It is not a food-trained model. To detect dal, roti, rice, burger, and other food reliably, train with labeled food images and use the produced `best.pt`.

## API

- `POST /analyze`
- Form-data field: `image`
