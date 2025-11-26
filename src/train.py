#!/usr/bin/env python3
"""
Simple YOLOv8 training script with MLflow logging.
Usage:
  export MLFLOW_TRACKING_URI=http://0.0.0.0:5000
  python src/train.py --data data/processed/dataset.yaml --img 640 --batch 16 --epochs 2 --weights yolov8n.pt --project models --name smoke
"""
import argparse
import time
import os
from pathlib import Path
import mlflow
from ultralytics import YOLO

def find_best(weights_dir: Path):
    cand = weights_dir / "weights" / "best.pt"
    if cand.exists():
        return cand
    pts = list(weights_dir.rglob("*.pt"))
    return pts[-1] if pts else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset.yaml")
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--project", default="models")
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("vehicle-detection")

    run_name = args.name or f"yolov8-{int(time.time())}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "data": args.data,
            "img": args.img,
            "batch": args.batch,
            "epochs": args.epochs,
            "init_weights": args.weights,
            "device": args.device or "cpu"
        })

        model = YOLO(args.weights)
        print("Starting training...")
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.img,
            batch=args.batch,
            project=str(Path(args.project)),
            name=run_name,
            device=args.device or None,
            exist_ok=True
        )

        weights_dir = Path(args.project) / run_name
        best = find_best(weights_dir)
        if best and best.exists():
            print("Found best:", best)
            mlflow.log_artifact(str(best), artifact_path="model")
        else:
            print("No best.pt found automatically in", weights_dir)

        mlflow.log_artifacts(str(weights_dir), artifact_path="training_run")
        print("Training finished. MLflow run id:", mlflow.active_run().info.run_id)

if __name__ == "__main__":
    main()
