#!/usr/bin/env python3
"""
Simple detect script using Ultralytics YOLOv8.
Usage:
  python src/detect.py --weights models/<run>/weights/best.pt --source data/processed/images/test --out inference --conf 0.25
"""
import argparse
from pathlib import Path
import os
import mlflow
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", default="data/processed/images/test")
    parser.add_argument("--out", default="inference")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save_txt", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    print("Running inference...")
    model.predict(source=str(args.source), save=True, conf=args.conf, project=str(out_dir), name="preds", save_txt=args.save_txt)
    print("Inference outputs saved to:", out_dir)

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    with mlflow.start_run(run_name="inference-"+Path(args.weights).stem):
        mlflow.log_param("weights", args.weights)
        mlflow.log_param("source", args.source)
        mlflow.log_artifacts(str(out_dir), artifact_path="inference")

if __name__ == "__main__":
    main()
