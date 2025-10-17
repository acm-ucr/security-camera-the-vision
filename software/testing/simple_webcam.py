import cv2
import argparse
from ultralytics import YOLO

conversion = ["car", "cat", "dog", "person"]

def score_to_bgr(score: float) -> tuple[int, int, int]:
    """Map score in [0,1] to a BGR color from red (low) to green (high)."""
    s = max(0.0, min(1.0, float(score)))
    g = int(255 * s)
    r = int(255 * (1.0 - s))
    b = 0
    return (b, g, r)


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
                if score < 0.6:
                    continue
                color = score_to_bgr(score)
                text = f"{conversion[int(model.names[label])]}: {score:.2f}" if hasattr(model, 'names') else f"{label}: {score:.2f}"
                # draw box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # text background for readability
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                tx1 = int(x1)
                ty1 = int(y1) - th - 6
                ty1 = max(0, ty1)
                tx2 = tx1 + tw + 6
                ty2 = ty1 + th + 6
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, -1)
                cv2.putText(frame, text, (tx1 + 3, ty2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='cricket.pt')
    parser.add_argument('--img-size', type=int, nargs=2, default=(320, 320))
    args = parser.parse_args()
    model = YOLO(args.model)
    run_webcam(model, tuple(args.img_size))
