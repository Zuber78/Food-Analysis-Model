import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Train a local YOLO food detector.')
    parser.add_argument('--data', default='datasets/food/data.yaml', help='Path to YOLO dataset YAML.')
    parser.add_argument('--base-model', default='yolov8n.pt', help='Base YOLO weights for transfer learning.')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs.')
    parser.add_argument('--imgsz', type=int, default=640, help='Training image size.')
    parser.add_argument('--batch', type=int, default=8, help='Batch size.')
    parser.add_argument('--device', default='cpu', help='Use cpu, 0, 1, etc.')
    parser.add_argument('--project', default='runs/detect', help='Ultralytics output project folder.')
    parser.add_argument('--name', default='food-yolo', help='Training run name.')
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f'Dataset config not found: {data_path}. Create it from datasets/food/data.yaml.example.'
        )

    model = YOLO(args.base_model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )

    best_path = Path(args.project) / args.name / 'weights' / 'best.pt'
    print(f'\nTraining complete. Set FOOD_MODEL_PATH={best_path}')


if __name__ == '__main__':
    main()
