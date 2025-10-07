#!/usr/bin/env python3
"""
Webcam object-detection demo using a PyTorch .pt model.

Assumptions & behaviour
- The model file is `10epoch.pt` in the same directory as this script.
- The script will try several common output formats and convert them to
  a canonical list of detections: (x1, y1, x2, y2, score, label).
- Press 'q' to quit the webcam window.

If your model uses a different signature, update `run_inference()` to
match its inputs/outputs.
"""

import numpy as np


def preprocess_frame(frame: np.ndarray, img_size: tuple[int, int] = (640, 640)) -> tuple[object, float, tuple[int, int]]:
    """Resize and normalize an OpenCV BGR frame for a generic PyTorch model.

    Returns: tensor (1,C,H,W), scale factor applied (for mapping boxes back), and original shape
    """
    import torch as _torch

    orig_h, orig_w = frame.shape[:2]
    target_w, target_h = img_size

    # letterbox resize (preserve aspect ratio)
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    # create canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    # BGR -> RGB, transpose to CHW, convert to float, normalize to [0,1]
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = _torch.from_numpy(img).unsqueeze(0)
    return tensor, scale, (orig_w, orig_h)


def postprocess_outputs(outputs, scale: float, orig_shape: tuple[int, int], conf_thres: float = 0.4, preprocessed: bool = True):
    """Convert several plausible model outputs to a list of detections.

    Returns list of (x1,y1,x2,y2,score,label) in original image coordinates.
    """
    prev = 0
    frame_i = 0
    # Optionally set torch threads

    # 1) torchvision detection models -> list[dict] or dict with 'boxes','scores','labels'
    if isinstance(outputs, dict):
        #!/usr/bin/env python3
        """Webcam object-detection demo.

        This script tries to be robust across model formats (Ultralytics YOLO objects,
        TorchScript, or standard torch.nn.Module checkpoints). It draws bounding boxes
        and the class name + confidence above each box.

        Usage (example):
          python3 main.py --model 10epoch.pt --img-size 320 320 --skip-frames 1 --use-threads

        If your model requires a custom input signature you'll need to adapt
        `load_model()` or the inference path in `run_webcam()`.
        """

        from typing import Tuple, List, Optional, Dict
        import argparse
        import time
        import threading
        import queue
        import sys

        import numpy as np
        import cv2

        try:
            import torch
        except Exception:
            torch = None


        def load_model(path: str, device: str = "cpu"):
            """Try loading the model in a few common ways.

            Returns (model, is_ultralytics_flag).
            """
            # 1) ultralytics.YOLO wrapper (preferred if available)
            try:
                import ultralytics  # type: ignore

                try:
                    model = ultralytics.YOLO(path)
                    return model, True
                except Exception:
                    # sometimes ultralytics cannot load a raw torch archive; fallthrough
                    pass
            except Exception:
                pass

            # 2) TorchScript
            if torch is not None:
                try:
                    model = torch.jit.load(path, map_location=device)
                    model.eval()
                    return model, False
                except Exception:
                    pass

            # 3) torch.load (state_dict or pickled module)
            if torch is not None:
                try:
                    obj = torch.load(path, map_location=device)
                    # If it's a state_dict, user must provide a model architecture; we can't
                    # reconstruct arbitrary architectures. Give a helpful error.
                    if isinstance(obj, dict) and any(k.startswith("state_dict") or k.endswith("state_dict") for k in obj.keys()):
                        raise RuntimeError("The file appears to be a state_dict. This demo cannot instantiate arbitrary model architectures. Provide a scripted/torch.jit model or an Ultralytics export.")
                    if hasattr(obj, "eval") and hasattr(obj, "parameters"):
                        obj.eval()
                        return obj, False
                except Exception as e:
                    raise RuntimeError(f"Failed to load model via ultralytics/torch.jit/torch.load: {e}\nIf this is a weights-only checkpoint, either export a scripted model or install the library that produced it (for example, ultralytics).")

            raise RuntimeError("No supported model loader available (need torch or ultralytics)")


        def letterbox(frame: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
            """Resize image with unchanged aspect ratio using padding (letterbox).

            Returns resized image and scale factor applied to original image.
            """
            h, w = frame.shape[:2]
            tw, th = target_size
            scale = min(tw / w, th / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (nw, nh))
            canvas = np.zeros((th, tw, 3), dtype=np.uint8)
            canvas[:nh, :nw] = resized
            return canvas, scale


        def preprocess_frame_for_tensor(frame: np.ndarray, img_size: Tuple[int, int]) -> Tuple[object, float, Tuple[int, int]]:
            """Prepare frame for tensor-based models (BGR -> RGB, CHW, normalized).
            Returns tensor (1,C,H,W), scale, (orig_w, orig_h)
            """
            orig_h, orig_w = frame.shape[:2]
            canvas, scale = letterbox(frame, img_size)
            # BGR to RGB
            img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            if torch is None:
                #!/usr/bin/env python3
                """Webcam object-detection demo.

                This script is a self-contained demo that loads a .pt model (Ultralytics or
                TorchScript/torch.load) and runs inference on a webcam feed. It draws
                bounding boxes with the class name and confidence above each box.

                Usage example:
                  python3 main.py --model 10epoch.pt --img-size 320 320 --skip-frames 1 --use-threads
                """

                from typing import Tuple, List, Optional, Dict
                import argparse
                import time
                import threading
                import queue
                import sys

                import numpy as np
                import cv2

                try:
                    import torch
                except Exception:
                    torch = None


                def load_model(path: str, device: str = "cpu"):
                    """Try loading the model in a few common ways.

                    Returns (model, is_ultralytics_flag).
                    """
                    # 1) ultralytics.YOLO wrapper (preferred if available)
                    try:
                        import ultralytics  # type: ignore

                        try:
                            model = ultralytics.YOLO(path)
                            return model, True
                        except Exception:
                            # sometimes ultralytics cannot load a raw torch archive; fallthrough
                            pass
                    except Exception:
                        pass

                    # 2) TorchScript
                    if torch is not None:
                        try:
                            model = torch.jit.load(path, map_location=device)
                            model.eval()
                            return model, False
                        except Exception:
                            pass

                    # 3) torch.load (state_dict or pickled module)
                    if torch is not None:
                        try:
                            obj = torch.load(path, map_location=device)
                            # If it's a state_dict, user must provide a model architecture; we can't
                            # reconstruct arbitrary architectures. Give a helpful error.
                            if isinstance(obj, dict) and any(k.startswith("state_dict") or k.endswith("state_dict") for k in obj.keys()):
                                raise RuntimeError(
                                    "The file appears to be a state_dict. This demo cannot instantiate arbitrary model architectures."
                                )
                            if hasattr(obj, "eval") and hasattr(obj, "parameters"):
                                obj.eval()
                                return obj, False
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to load model via ultralytics/torch.jit/torch.load: {e}\nIf this is a weights-only checkpoint, either export a scripted model or install the library that produced it (for example, ultralytics)."
                            )

                    raise RuntimeError("No supported model loader available (need torch or ultralytics)")


                def letterbox(frame: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
                    """Resize image with unchanged aspect ratio using padding (letterbox).

                    Returns resized image and scale factor applied to original image.
                    """
                    h, w = frame.shape[:2]
                    tw, th = target_size
                    scale = min(tw / w, th / h)
                    nw, nh = int(w * scale), int(h * scale)
                    resized = cv2.resize(frame, (nw, nh))
                    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
                    canvas[:nh, :nw] = resized
                    return canvas, scale


                def preprocess_frame_for_tensor(frame: np.ndarray, img_size: Tuple[int, int]) -> Tuple['torch.Tensor', float, Tuple[int, int]]:
                    """Prepare frame for tensor-based models (BGR -> RGB, CHW, normalized).
                    Returns tensor (1,C,H,W), scale, (orig_w, orig_h)
                    """
                    orig_h, orig_w = frame.shape[:2]
                    canvas, scale = letterbox(frame, img_size)
                    # BGR to RGB
                    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    img = np.transpose(img, (2, 0, 1))
                    if torch is None:
                        raise RuntimeError("Torch is required for tensor-based inference paths")
                    tensor = torch.from_numpy(img).unsqueeze(0)
                    return tensor, scale, (orig_w, orig_h)


                def postprocess_outputs(outputs, scale: float, orig_shape: Tuple[int, int], conf_thres: float = 0.4, preprocessed: bool = True) -> List[Tuple[int, int, int, int, float, int]]:
                    """Normalize several plausible model outputs into list of detections.

                    Each detection is (x1,y1,x2,y2,score,label) in original image coordinates.
                    """
                    dets = []

                    # Case A: Ultralytics Results (model(frame) or model.predict) -> results list
                    try:
                        first = outputs[0]
                        if hasattr(first, "boxes"):
                            boxes = first.boxes
                            try:
                                xyxy = boxes.xyxy.cpu().numpy()
                                confs = boxes.conf.cpu().numpy()
                                clss = boxes.cls.cpu().numpy().astype(int)
                            except Exception:
                                xyxy = np.array(boxes.xyxy.tolist())
                                confs = np.array(boxes.conf.tolist())
                                clss = np.array(boxes.cls.tolist()).astype(int)

                            for (x1, y1, x2, y2), conf, c in zip(xyxy, confs, clss):
                                if conf < conf_thres:
                                    continue
                                dets.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(c)))
                            return dets
                    except Exception:
                        pass
                    except Exception:
                        pass

                    # Case B: torchvision-like dict {'boxes':Tensor, 'scores':Tensor, 'labels':Tensor}
                    try:
                        if isinstance(outputs, dict) and 'boxes' in outputs:
                            boxes = outputs['boxes'].cpu().numpy()
                            scores = outputs['scores'].cpu().numpy()
                            labels = outputs['labels'].cpu().numpy().astype(int)
                            for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                                if s < conf_thres:
                                    continue
                                if preprocessed:
                                    x1 = int(x1 / scale)
                                    y1 = int(y1 / scale)
                                    x2 = int(x2 / scale)
                                    y2 = int(y2 / scale)
                                dets.append((x1, y1, x2, y2, float(s), int(l)))
                            return dets
                    except Exception:
                        pass

                    # Case C: raw tensor/ndarray Nx6 or Nx7 -> [x1,y1,x2,y2,score,label]
                    try:
                        arr = outputs
                        if hasattr(arr, 'cpu'):
                            arr = arr.cpu().numpy()
                        arr = np.asarray(arr)
                        if arr.ndim == 2 and arr.shape[1] >= 5:
                            for row in arr:
                                x1, y1, x2, y2 = map(float, row[:4])
                                score = float(row[4])
                                label = int(row[5]) if row.shape[0] > 5 else 0
                                if score < conf_thres:
                                    continue
                                if preprocessed:
                                    x1 = int(x1 / scale)
                                    y1 = int(y1 / scale)
                                    x2 = int(x2 / scale)
                                    y2 = int(y2 / scale)
                                dets.append((int(x1), int(y1), int(x2), int(y2), score, label))
                            return dets
                    except Exception:
                        pass

                    return dets


                def draw_detections(frame: np.ndarray, detections: List[Tuple[int, int, int, int, float, int]], label_map: Optional[Dict[int, str]] = None):
                    """Draw boxes and class name+confidence on the frame in-place."""
                    for (x1, y1, x2, y2, score, label) in detections:
                        cls_name = str(label_map[label]) if (label_map and label in label_map) else str(label)
                        text = f"{cls_name} {score:.2f}"
                        # choose color by label
                        color = tuple(int(c) for c in np.array([(label * 37) % 255, (label * 61) % 255, (label * 97) % 255]))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        # text background
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = 0.6
                        thickness = 1
                        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
                        # put background rectangle above the box (preferably)
                        bx1 = x1
                        by2 = max(y1 - 4, h + 4)
                        bx2 = x1 + w + 6
                        by1 = by2 - h - 6
                        # clamp
                        bx1 = max(0, bx1)
                        by1 = max(0, by1)
                        bx2 = min(frame.shape[1], bx2)
                        by2 = min(frame.shape[0], by2)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
                        cv2.putText(frame, text, (bx1 + 3, by2 - 4), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


                def run_webcam(model, device: str = "cpu", webcam_idx: int = 0, img_size: Tuple[int, int] = (416, 416), conf: float = 0.4, force_cpu: bool = False, half_arg: bool = False, skip_frames: int = 0, debug: bool = False, use_threads: bool = False, num_threads: int = 0):
                    cap = cv2.VideoCapture(webcam_idx)
                    if not cap.isOpened():
                        raise RuntimeError(f"Could not open webcam index {webcam_idx}")

                    print("Starting webcam. Press 'q' to quit.")
                    # restrict CPU threads if requested
                    try:
                        if num_threads and torch is not None:
                            torch.set_num_threads(num_threads)
                            print(f"Set torch num threads to {num_threads}")
                    except Exception:
                        pass

                    is_ultralytics = False
                    try:
                        import ultralytics  # type: ignore
                        if model.__class__.__module__.startswith(ultralytics.__name__):
                            is_ultralytics = True
                    except Exception:
                        is_ultralytics = False

                    # try to extract label names if available
                    label_map = None
                    try:
                        if is_ultralytics:
                            names = None
                            if hasattr(model, 'names'):
                                names = model.names
                            elif hasattr(model, 'model') and hasattr(model.model, 'names'):
                                names = model.model.names
                            if names is not None:
                                if isinstance(names, dict):
                                    label_map = {int(k): v for k, v in names.items()}
                                else:
                                    label_map = {i: v for i, v in enumerate(names)}
                    except Exception:
                        label_map = None

                    frame_i = 0
                    q = None
                    stop_event = threading.Event()

                    if use_threads:
                        q = queue.Queue(maxsize=1)

                        def capture_thread():
                            while not stop_event.is_set():
                                ret, f = cap.read()
                                if not ret:
                                    break
                                try:
                                    if q.full():
                                        _ = q.get_nowait()
                                except Exception:
                                    pass
                                q.put(f)

                        t = threading.Thread(target=capture_thread, daemon=True)
                        t.start()

                   
                        while True:
                            if use_threads:
                                try:
                                    frame = q.get(timeout=1.0)
                                except Exception:
                                    print("No frame from capture thread; exiting")
                                    break
                            else:
                                ret, frame = cap.read()
                                if not ret:
                                    break

                            # simple frame skipping
                            if skip_frames > 0 and (frame_i % (skip_frames + 1)) != 0:
                                frame_i += 1
                                continue

                            t0 = time.time()

                            # Ultralytics path: let the library handle preprocessing
                            if is_ultralytics:
                                try:
                                    outputs = model(frame)
                                except Exception:
                                    outputs = model.predict(frame)
                                dets = postprocess_outputs(outputs, scale=1.0, orig_shape=(frame.shape[1], frame.shape[0]), conf_thres=conf, preprocessed=False)
                            else:
                                # tensor-based inference
                                tensor, scale, orig_shape = preprocess_frame_for_tensor(frame, img_size)
                                if torch is not None:
                                    # pick model device/dtype
                                    try:
                                        first = next(model.parameters())
                                        model_device = first.device
                                    except Exception:
                                        model_device = torch.device(device)

                                    if force_cpu:
                                        model_device = torch.device('cpu')

                                    tensor = tensor.to(model_device)
                                    if half_arg and model_device.type == 'cuda':
                                        tensor = tensor.half()
                                    else:
                                        tensor = tensor.float()

                                    # run inference
                                    try:
                                        if model_device.type == 'cuda':
                                            autocast = getattr(torch.cuda.amp, 'autocast', None)
                                            if autocast is None:
                                                with torch.no_grad():
                                                    outputs = model(tensor)
                                            else:
                                                with torch.no_grad(), autocast(enabled=half_arg):
                                                    outputs = model(tensor)
                                        else:
                                            with torch.no_grad():
                                                outputs = model(tensor)
                                    except Exception as e:
                                        raise RuntimeError(f"Model inference failed: {e}")

                                    dets = postprocess_outputs(outputs, scale=scale, orig_shape=orig_shape, conf_thres=conf, preprocessed=True)
                                else:
                                    dets = []

                            frame_i += 1

                            if debug:
                                print("Detections:", dets)

                            #!/usr/bin/env python3
                            """Webcam object-detection demo.

                            This script is a self-contained demo that loads a .pt model (Ultralytics or
                            TorchScript/torch.load) and runs inference on a webcam feed. It draws
                            bounding boxes with the class name and confidence above each box.

                            Usage example:
                              python3 main.py --model 10epoch.pt --img-size 320 320 --skip-frames 1 --use-threads
                            """

                            from typing import Tuple, List, Optional, Dict
                            import argparse
                            import time
                            import threading
                            import queue
                            import sys

                            import numpy as np
                            import cv2

                            try:
                                import torch
                            except Exception:
                                torch = None


                            def load_model(path: str, device: str = "cpu"):
                                """Try loading the model in a few common ways.

                                Returns (model, is_ultralytics_flag).
                                """
                                # 1) ultralytics.YOLO wrapper (preferred if available)
                                try:
                                    import ultralytics  # type: ignore

                                    try:
                                        model = ultralytics.YOLO(path)
                                        return model, True
                                    except Exception:
                                        # sometimes ultralytics cannot load a raw torch archive; fallthrough
                                        pass
                                except Exception:
                                    pass

                                # 2) TorchScript
                                if torch is not None:
                                    try:
                                        model = torch.jit.load(path, map_location=device)
                                        model.eval()
                                        return model, False
                                    except Exception:
                                        pass

                                # 3) torch.load (state_dict or pickled module)
                                if torch is not None:
                                    try:
                                        obj = torch.load(path, map_location=device)
                                        # If it's a state_dict, user must provide a model architecture; we can't
                                        # reconstruct arbitrary architectures. Give a helpful error.
                                        if isinstance(obj, dict) and any(k.startswith("state_dict") or k.endswith("state_dict") for k in obj.keys()):
                                            raise RuntimeError(
                                                "The file appears to be a state_dict. This demo cannot instantiate arbitrary model architectures."
                                            )
                                        if hasattr(obj, "eval") and hasattr(obj, "parameters"):
                                            obj.eval()
                                            return obj, False
                                    except Exception as e:
                                        raise RuntimeError(
                                            f"Failed to load model via ultralytics/torch.jit/torch.load: {e}\nIf this is a weights-only checkpoint, either export a scripted model or install the library that produced it (for example, ultralytics)."
                                        )

                                raise RuntimeError("No supported model loader available (need torch or ultralytics)")


                            def letterbox(frame: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
                                """Resize image with unchanged aspect ratio using padding (letterbox).

                                Returns resized image and scale factor applied to original image.
                                """
                                h, w = frame.shape[:2]
                                tw, th = target_size
                                scale = min(tw / w, th / h)
                                nw, nh = int(w * scale), int(h * scale)
                                resized = cv2.resize(frame, (nw, nh))
                                canvas = np.zeros((th, tw, 3), dtype=np.uint8)
                                canvas[:nh, :nw] = resized
                                return canvas, scale


                            def preprocess_frame_for_tensor(frame: np.ndarray, img_size: Tuple[int, int]) -> Tuple[object, float, Tuple[int, int]]:
                                """Prepare frame for tensor-based models (BGR -> RGB, CHW, normalized).
                                Returns tensor (1,C,H,W), scale, (orig_w, orig_h)
                                """
                                orig_h, orig_w = frame.shape[:2]
                                canvas, scale = letterbox(frame, img_size)
                                # BGR to RGB
                                img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                                img = np.transpose(img, (2, 0, 1))
                                if torch is None:
                                    raise RuntimeError("Torch is required for tensor-based inference paths")
                                tensor = torch.from_numpy(img).unsqueeze(0)
                                return tensor, scale, (orig_w, orig_h)


                            def postprocess_outputs(outputs, scale: float, orig_shape: Tuple[int, int], conf_thres: float = 0.4, preprocessed: bool = True) -> List[Tuple[int, int, int, int, float, int]]:
                                """Normalize several plausible model outputs into list of detections.

                                Each detection is (x1,y1,x2,y2,score,label) in original image coordinates.
                                """
                                dets = []

                                # Case A: Ultralytics Results (model(frame) or model.predict) -> results list
                                try:
                                    first = outputs[0]
                                    if hasattr(first, "boxes"):
                                        boxes = first.boxes
                                        try:
                                            xyxy = boxes.xyxy.cpu().numpy()
                                            confs = boxes.conf.cpu().numpy()
                                            clss = boxes.cls.cpu().numpy().astype(int)
                                        except Exception:
                                            xyxy = np.array(boxes.xyxy.tolist())
                                            confs = np.array(boxes.conf.tolist())
                                            clss = np.array(boxes.cls.tolist()).astype(int)

                                        for (x1, y1, x2, y2), conf, c in zip(xyxy, confs, clss):
                                            if conf < conf_thres:
                                                continue
                                            dets.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(c)))
                                        return dets
                                except Exception:
                                    pass

                                # Case B: torchvision-like dict {'boxes':Tensor, 'scores':Tensor, 'labels':Tensor}
                                try:
                                    if isinstance(outputs, dict) and 'boxes' in outputs:
                                        boxes = outputs['boxes'].cpu().numpy()
                                        scores = outputs['scores'].cpu().numpy()
                                        labels = outputs['labels'].cpu().numpy().astype(int)
                                        for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                                            if s < conf_thres:
                                                continue
                                            if preprocessed:
                                                x1 = int(x1 / scale)
                                                y1 = int(y1 / scale)
                                                x2 = int(x2 / scale)
                                                y2 = int(y2 / scale)
                                            dets.append((x1, y1, x2, y2, float(s), int(l)))
                                        return dets
                                except Exception:
                                    pass

                                # Case C: raw tensor/ndarray Nx6 or Nx7 -> [x1,y1,x2,y2,score,label]
                                try:
                                    arr = outputs
                                    if hasattr(arr, 'cpu'):
                                        arr = arr.cpu().numpy()
                                    arr = np.asarray(arr)
                                    if arr.ndim == 2 and arr.shape[1] >= 5:
                                        for row in arr:
                                            x1, y1, x2, y2 = map(float, row[:4])
                                            score = float(row[4])
                                            label = int(row[5]) if row.shape[0] > 5 else 0
                                            if score < conf_thres:
                                                continue
                                            if preprocessed:
                                                x1 = int(x1 / scale)
                                                y1 = int(y1 / scale)
                                                x2 = int(x2 / scale)
                                                y2 = int(y2 / scale)
                                            dets.append((int(x1), int(y1), int(x2), int(y2), score, label))
                                        return dets
                                except Exception:
                                    pass

                                return dets


                            def draw_detections(frame: np.ndarray, detections: List[Tuple[int, int, int, int, float, int]], label_map: Optional[Dict[int, str]] = None):
                                """Draw boxes and class name+confidence on the frame in-place."""
                                for (x1, y1, x2, y2, score, label) in detections:
                                    cls_name = str(label_map[label]) if (label_map and label in label_map) else str(label)
                                    text = f"{cls_name} {score:.2f}"
                                    # choose color by label
                                    color = tuple(int(c) for c in np.array([(label * 37) % 255, (label * 61) % 255, (label * 97) % 255]))
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                    # text background
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    scale = 0.6
                                    thickness = 1
                                    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
                                    # put background rectangle above the box (preferably)
                                    bx1 = x1
                                    by2 = max(y1 - 4, h + 4)
                                    bx2 = x1 + w + 6
                                    by1 = by2 - h - 6
                                    # clamp
                                    bx1 = max(0, bx1)
                                    by1 = max(0, by1)
                                    bx2 = min(frame.shape[1], bx2)
                                    by2 = min(frame.shape[0], by2)
                                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
                                    cv2.putText(frame, text, (bx1 + 3, by2 - 4), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


                            def run_webcam(model, device: str = "cpu", webcam_idx: int = 0, img_size: Tuple[int, int] = (416, 416), conf: float = 0.4, force_cpu: bool = False, half_arg: bool = False, skip_frames: int = 0, debug: bool = False, use_threads: bool = False, num_threads: int = 0):
                                cap = cv2.VideoCapture(webcam_idx)
                                if not cap.isOpened():
                                    raise RuntimeError(f"Could not open webcam index {webcam_idx}")

                                print("Starting webcam. Press 'q' to quit.")
                                # restrict CPU threads if requested
                                try:
                                    if num_threads and torch is not None:
                                        torch.set_num_threads(num_threads)
                                        print(f"Set torch num threads to {num_threads}")
                                except Exception:
                                    pass

                                is_ultralytics = False
                                try:
                                    import ultralytics  # type: ignore
                                    if model.__class__.__module__.startswith(ultralytics.__name__):
                                        is_ultralytics = True
                                except Exception:
                                    is_ultralytics = False

                                # try to extract label names if available
                                label_map = None
                                try:
                                    if is_ultralytics:
                                        names = None
                                        if hasattr(model, 'names'):
                                            names = model.names
                                        elif hasattr(model, 'model') and hasattr(model.model, 'names'):
                                            names = model.model.names
                                        if names is not None:
                                            if isinstance(names, dict):
                                                label_map = {int(k): v for k, v in names.items()}
                                            else:
                                                label_map = {i: v for i, v in enumerate(names)}
                                except Exception:
                                    label_map = None

                                frame_i = 0
                                q = None
                                stop_event = threading.Event()

                                if use_threads:
                                    q = queue.Queue(maxsize=1)

                                    def capture_thread():
                                        while not stop_event.is_set():
                                            ret, f = cap.read()
                                            if not ret:
                                                break
                                            try:
                                                if q.full():
                                                    _ = q.get_nowait()
                                            except Exception:
                                                pass
                                            q.put(f)

                                    t = threading.Thread(target=capture_thread, daemon=True)
                                    t.start()

                                try:
                                    while True:
                                        if use_threads:
                                            try:
                                                frame = q.get(timeout=1.0)
                                            except Exception:
                                                print("No frame from capture thread; exiting")
                                                break
                                        else:
                                            ret, frame = cap.read()
                                            if not ret:
                                                break
            
                                        # simple frame skipping
                                        if skip_frames > 0 and (frame_i % (skip_frames + 1)) != 0:
                                            frame_i += 1
                                            continue
            
                                        t0 = time.time()
            
                                        # Ultralytics path: let the library handle preprocessing
                                        if is_ultralytics:
                                            try:
                                                outputs = model(frame)
                                            except Exception:
                                                outputs = model.predict(frame)
                                            dets = postprocess_outputs(outputs, scale=1.0, orig_shape=(frame.shape[1], frame.shape[0]), conf_thres=conf, preprocessed=False)
                                        else:
                                            # tensor-based inference
                                            tensor, scale, orig_shape = preprocess_frame_for_tensor(frame, img_size)
                                            if torch is not None:
                                                # pick model device/dtype
                                                try:
                                                    first = next(model.parameters())
                                                    model_device = first.device
                                                except Exception:
                                                    model_device = torch.device(device)
            
                                                if force_cpu:
                                                    model_device = torch.device('cpu')
            
                                                tensor = tensor.to(model_device)
                                                if half_arg and model_device.type == 'cuda':
                                                    tensor = tensor.half()
                                                else:
                                                    tensor = tensor.float()
            
                                                # run inference
                                                try:
                                                    if model_device.type == 'cuda':
                                                        autocast = getattr(torch.cuda.amp, 'autocast', None)
                                                        if autocast is None:
                                                            with torch.no_grad():
                                                                outputs = model(tensor)
                                                        else:
                                                            with torch.no_grad(), autocast(enabled=half_arg):
                                                                outputs = model(tensor)
                                                    else:
                                                        with torch.no_grad():
                                                            outputs = model(tensor)
                                                except Exception as e:
                                                    raise RuntimeError(f"Model inference failed: {e}")
            
                                                dets = postprocess_outputs(outputs, scale=scale, orig_shape=orig_shape, conf_thres=conf, preprocessed=True)
                                            else:
                                                dets = []
            
                                        frame_i += 1
            
                                        if debug:
                                            print("Detections:", dets)
            
                                        draw_detections(frame, dets, label_map=label_map)
                                        fps = 1.0 / (time.time() - t0 + 1e-6)
                                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                                        cv2.imshow("webcam", frame)
                                        key = cv2.waitKey(1) & 0xFF
                                        if key == ord('q'):
                                            break
                                except Exception as e:
                                    print(f"Error occurred during webcam loop: {e}")
                                finally:
                                    stop_event.set()
                                    cap.release()
                                    cv2.destroyAllWindows()


                            def parse_args():
                                p = argparse.ArgumentParser(description="Run a .pt model on webcam and draw boxes + class names")
                                p.add_argument('--model', default='10epoch.pt', help='Path to model file')
                                p.add_argument('--device', default='cuda' if torch is not None and torch.cuda.is_available() else 'cpu')
                                p.add_argument('--force-cpu', action='store_true', help='Force CPU even if CUDA is available')
                                p.add_argument('--half', action='store_true', help='Use fp16 when running on CUDA (faster on GPUs that support it)')
                                p.add_argument('--skip-frames', type=int, default=0, help='Process every (skip_frames+1)-th frame')
                                p.add_argument('--webcam', type=int, default=0, help='Webcam index')
                                p.add_argument('--img-size', type=int, nargs=2, default=(416, 416), help='Inference image size (W H)')
                                p.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
                                p.add_argument('--debug', action='store_true', help='Print debug info')
                                p.add_argument('--use-threads', action='store_true', help='Use capture thread to reduce latency')
                                p.add_argument('--num-threads', type=int, default=0, help='Set torch.set_num_threads(N)')
                                return p.parse_args()


                            def main():
                                args = parse_args()
                                model, _ = load_model(args.model, device=args.device)
                                run_webcam(
                                    model,
                                    device=args.device,
                                    webcam_idx=args.webcam,
                                    img_size=tuple(args.img_size),
                                    conf=args.conf,
                                    force_cpu=args.force_cpu,
                                    half_arg=args.half,
                                    skip_frames=args.skip_frames,
                                    debug=args.debug,
                                    use_threads=args.use_threads,
                                    num_threads=args.num_threads,
                                )



if __name__ == '__main__':
    main()
