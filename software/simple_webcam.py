
import cv2
import argparse
from ultralytics import YOLO

def run_webcam(model, img_size):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, imgsz=img_size)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            labels = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
                if score < 0.4:
                    continue
                color = (0, 255, 0)
                text = f"{model.names[label]}: {score:.2f}" if hasattr(model, 'names') else f"{label}: {score:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='10epoch.pt')
    parser.add_argument('--img-size', type=int, nargs=2, default=(320, 320))
    args = parser.parse_args()
    model = YOLO(args.model)
    run_webcam(model, tuple(args.img_size))
