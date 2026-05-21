from ultralytics import YOLO
import os

def main():
    model = YOLO("yolo11n.pt")

    # data.yaml'ın tam yolunu ver
    data_path = os.path.join(os.path.dirname(__file__), "data.yaml")

    model.train(
        data=data_path,
        epochs=10,
        imgsz=640,
        device="cpu"
    )

if __name__ == "__main__":
    main()