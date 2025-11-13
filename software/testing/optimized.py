
import cv2
import argparse
from ultralytics import YOLO
try:
    import torch
except Exception:
    torch = None

# prefer MPS on macOS when available
default_device = 'cpu'
if torch is not None:
    try:
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            default_device = 'mps'
        elif torch.cuda.is_available():
            default_device = 'cuda'
    except Exception:
        # fall back to cpu
        default_device = 'cpu'


def score_to_bgr(score: float) -> tuple[int, int, int]:
    """Map score in [0,1] to a BGR color from red (low) to green (high)."""
    s = max(0.0, min(1.0, float(score)))
    g = int(255 * s)
    r = int(255 * (1.0 - s))
    b = 0
    return (b, g, r)


def run_webcam(model, img_size, webcam_idx=0, skip_frames=0, use_threads=False, queue_size=2, conf=0.001, device='cpu', half=False):
    cap = cv2.VideoCapture(webcam_idx)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # enable OpenCV optimizations
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass

    frame_q = None
    stop_event = None
    if use_threads:
        import threading, queue as _queue
        frame_q = _queue.Queue(maxsize=max(1, queue_size))
        stop_event = threading.Event()

        def _capture_loop():
            while not stop_event.is_set():
                ret, f = cap.read()
                if not ret:
                    stop_event.set()
                    break
                try:
                    if frame_q.full():
                        # drop oldest
                        _ = frame_q.get_nowait()
                except Exception:
                    pass
                frame_q.put(f)

        t = threading.Thread(target=_capture_loop, daemon=True)
        t.start()

    print("Press 'q' to quit.")
    frame_i = 0
    import time
    fps = 0.0
    last_time = time.time()
    try:
        while True:
            # get frame (threaded or direct)
            if use_threads:
                try:
                    frame = frame_q.get(timeout=1.0)
                except Exception:
                    print("No frame from capture thread; exiting")
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            # optional frame skipping
            if skip_frames > 0 and (frame_i % (skip_frames + 1)) != 0:
                frame_i += 1
                continue

            # resize before inference to reduce work
            small = cv2.resize(frame, img_size)

            # inference: pass device/half/imgsz/conf to reduce post-filtering; let Ultralytics handle preprocessing
            try:
                results = model(small, device=device, half=half, imgsz=img_size, conf=conf, verbose=False)
            except Exception:
                # fallback without device/half
                results = model(small, imgsz=img_size, conf=conf)

            # draw results (convert boxes back to original frame scale)
            h_ratio = frame.shape[1] / img_size[0]
            v_ratio = frame.shape[0] / img_size[1]
            for r in results:
                # boxes in XYXY on small image
                try:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    labels = r.boxes.cls.cpu().numpy().astype(int)
                except Exception:
                    continue

                for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
                    if score < conf:
                        continue
                    # map box coords back to original frame scale
                    ox1 = int(x1 * h_ratio)
                    oy1 = int(y1 * v_ratio)
                    ox2 = int(x2 * h_ratio)
                    oy2 = int(y2 * v_ratio)

                    color = score_to_bgr(score)
                    text = f"{model.names[label]}: {score:.2f}" if hasattr(model, 'names') else f"{label}: {score:.2f}"

                    # draw box and label
                    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, 2)
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    tx1 = ox1
                    ty1 = max(0, oy1 - th - 6)
                    tx2 = tx1 + tw + 6
                    ty2 = ty1 + th + 6
                    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, -1)
                    cv2.putText(frame, text, (tx1 + 3, ty2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # compute FPS and overlay (draw before showing)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last_time))
            last_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow('webcam', frame)
            frame_i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if use_threads and stop_event is not None:
            stop_event.set()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='50epoch.pt')
    parser.add_argument('--img-size', type=int, nargs=2, default=(320, 320))
    parser.add_argument('--webcam', type=int, default=0, help='Webcam index')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for drawing')
    parser.add_argument('--use-threads', action='store_true', help='Use threaded capture')
    parser.add_argument('--skip-frames', type=int, default=0, help='Process every (skip_frames+1)-th frame')
    parser.add_argument('--queue-size', type=int, default=2, help='Capture queue size when using threads')
    parser.add_argument('--device', default=default_device, help='Device to run model on (cpu, cuda, or mps)')
    parser.add_argument('--half', action='store_true', help='Use FP16 (only when running on CUDA)')
    parser.add_argument('--num-threads', type=int, default=0, help='Set torch.set_num_threads(N)')
    args = parser.parse_args()
    # set thread count for torch
    if args.num_threads and torch is not None:
        try:
            torch.set_num_threads(int(args.num_threads))
        except Exception:
            pass

    model = YOLO(args.model)
    # report chosen device and half setting
    print(f"Requested device: {args.device}, half={args.half}")
    # adjust half precision availability: only enable half on CUDA
    if args.half and args.device != 'cuda':
        print("FP16 (--half) requested but only supported on CUDA; ignoring --half for this run.")
        args.half = False

    # try moving model to device (Ultralytics model supports .to())
    if args.device and torch is not None:
        try:
            # only move to cuda/mps if available
            if args.device == 'cuda' and torch.cuda.is_available():
                model.to('cuda')
            elif args.device == 'mps' and getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                try:
                    model.to('mps')
                except Exception:
                    # some ultralytics model internals may not support .to('mps') directly; fallback
                    pass
        except Exception:
            pass

    print(f"Model running on device: {args.device}, half={args.half}")

    run_webcam(model, tuple(args.img_size), webcam_idx=args.webcam, skip_frames=args.skip_frames, use_threads=args.use_threads, queue_size=args.queue_size, conf=args.conf, device=args.device, half=args.half)
