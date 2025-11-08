import cv2
import argparse
from ultralytics import YOLO
from mqtt_client import initialize_mqtt, send_detection_message, send_batch_detection, is_mqtt_connected
from apns_manager import initialize_apns_manager, send_detection_notification, send_system_status_notification, get_apns_manager
import os
import sys
import time
import signal
import logging
import subprocess
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

try:
    import torch
except Exception:
    torch = None

# Load environment variables from config.env file
config_path = Path(__file__).parent / 'config.env'
load_dotenv(config_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_camera.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

conversion = ["car", "cat", "dog", "person"]

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    logger.info(f"Received signal {sig}, shutting down gracefully...")
    shutdown_flag = True

# Auto-detect best device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
def get_best_device():
    """Auto-detect the best available device for inference"""
    if torch is None:
        return 'cpu'

    try:
        # Prefer MPS on macOS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        # NVIDIA GPU
        elif torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass

    return 'cpu'
def score_to_bgr(score: float) -> tuple[int, int, int]:
    """Map score in [0,1] to a BGR color from red (low) to green (high)."""
    s = max(0.0, min(1.0, float(score)))
    g = int(255 * s)
    r = int(255 * (1.0 - s))
    b = 0
    return (b, g, r)


def create_ffmpeg_process(rtmps_url, width, height, fps=30):
    """
    Create an FFmpeg process to stream video to Cloudflare RTMPS
    
    Args:
        rtmps_url: Full RTMPS URL including stream key
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second (default: 30)
    
    Returns:
        subprocess.Popen object or None if failed
    """
    try:
        # FFmpeg command to encode raw video from stdin and stream to RTMPS
        ffmpeg_cmd = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',  # OpenCV uses BGR format
            '-r', str(fps),
            '-i', '-',  # Read from stdin
            '-c:v', 'libx264',
            '-preset', 'ultrafast',  # Low latency
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-f', 'flv',
            '-flvflags', 'no_duration_filesize',
            rtmps_url
        ]
        
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
        logger.info(f"FFmpeg process started for RTMPS streaming: {rtmps_url[:50]}...")
        return process
    except Exception as e:
        logger.error(f"Failed to start FFmpeg process: {e}")
        return None


def write_frame_to_ffmpeg(process, frame):
    """
    Write a frame to FFmpeg process stdin
    
    Args:
        process: FFmpeg subprocess
        frame: OpenCV frame (numpy array)
    
    Returns:
        True if successful, False otherwise
    """
    if process is None or process.stdin is None:
        return False
    
    try:
        # Write frame data to FFmpeg stdin
        process.stdin.write(frame.tobytes())
        process.stdin.flush()
        return True
    except Exception as e:
        logger.warning(f"Failed to write frame to FFmpeg: {e}")
        return False


def run_webcam(model, img_size, headless=False, max_retries=5, retry_delay=5, read_timeout=30, detection_interval=60):
    """
    Run object detection on RTMP stream with automatic reconnection

    Args:
        model: YOLO model instance
        img_size: Tuple of (width, height) for inference
        headless: If True, don't display video window (for Docker/server)
        max_retries: Maximum connection retry attempts before long wait
        retry_delay: Seconds to wait between retry attempts
        read_timeout: Seconds to wait for frame before considering stream dead
        detection_interval: Run detection every N frames (default: 60)
    """
    global shutdown_flag
    frames = detection_interval

    # Get RTMP stream URL from environment variable
    rtmp_url = os.getenv('RTMP_STREAM_URL')
    if not rtmp_url:
        logger.error("RTMP_STREAM_URL not found in config.env")
        return
    
    # Get Cloudflare RTMPS output URL from environment variable or use provided one
    cloudflare_rtmps_url = os.getenv('CLOUDFLARE_RTMPS_URL', 
                                     'rtmps://live.cloudflare.com:443/live/bab8136eedaef02767c89e9dedbf3b48kb6a07db567bc485d2f4156bb122899cb')
    logger.info(f"Cloudflare RTMPS output URL configured: {cloudflare_rtmps_url[:50]}...")

    # Initialize MQTT connection
    logger.info("Initializing MQTT connection...")
    if initialize_mqtt():
        logger.info("MQTT connection established")
    else:
        logger.warning("MQTT connection failed, continuing without MQTT")

    # Initialize APNs manager for push notifications
    logger.info("Initializing APNs manager for push notifications...")
    apns_manager = initialize_apns_manager()
    logger.info(f"APNs manager initialized. Registered devices: {apns_manager.get_registered_devices_count()}")

    consecutive_failures = 0
    cap = None
    last_frame_time = None
    connection_start_time = None
    frames_processed = 0
    camera_status = None  # Track camera status: None, 'online', 'offline'
    
    # FFmpeg output stream variables
    ffmpeg_process = None
    stream_width = None
    stream_height = None
    stream_fps = 30
    output_stream_failures = 0
    
    # Cache for last annotations to redraw on skipped frames
    last_annotations = []  # List of (boxes, scores, labels, colors, texts)

    try:
        while not shutdown_flag:
            try:
                # Connection attempt
                if cap is None or not cap.isOpened():
                    logger.info(f"Connecting to RTMP stream: {rtmp_url[:50]}...")
                    cap = cv2.VideoCapture(rtmp_url)

                    # Set buffer size to minimum to reduce latency and avoid stale frames
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if not cap.isOpened():
                        consecutive_failures += 1

                        if consecutive_failures >= max_retries:
                            logger.error(f"Failed to connect after {max_retries} attempts. Waiting 60s before retry...")

                            # Send offline notification if status changed
                            if camera_status != 'offline':
                                send_system_status_notification(
                                    'offline',
                                    f'Camera failed to connect after {max_retries} attempts',
                                    'rtmp_stream'
                                )
                                camera_status = 'offline'

                            time.sleep(60)
                            consecutive_failures = 0
                            continue

                        logger.warning(f"Connection failed. Retry {consecutive_failures}/{max_retries} in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue

                    # Successfully connected
                    consecutive_failures = 0
                    connection_start_time = datetime.now()
                    frames_processed = 0
                    logger.info("✅ Connected to stream successfully")
                    
                    # Get stream dimensions for FFmpeg
                    stream_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    stream_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    stream_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                    logger.info(f"Stream dimensions: {stream_width}x{stream_height} @ {stream_fps} fps")
                    
                    # Start FFmpeg process for RTMPS output
                    if ffmpeg_process is None:
                        ffmpeg_process = create_ffmpeg_process(
                            cloudflare_rtmps_url,
                            stream_width,
                            stream_height,
                            stream_fps
                        )
                        if ffmpeg_process is None:
                            logger.warning("Failed to start FFmpeg output stream, continuing without streaming")

                    # Send online notification if status changed
                    if camera_status != 'online':
                        send_system_status_notification(
                            'online',
                            'Camera connected and streaming',
                            'rtmp_stream'
                        )
                        camera_status = 'online'

                # Read frame with timeout detection
                ret, frame = cap.read()
                current_time = time.time()

                if not ret:
                    logger.warning("Stream interrupted or ended. Attempting reconnection...")

                    # Send offline notification if status changed
                    if camera_status != 'offline':
                        send_system_status_notification(
                            'offline',
                            'Stream interrupted or connection lost',
                            'rtmp_stream'
                        )
                        camera_status = 'offline'

                    cap.release()
                    cap = None
                    time.sleep(retry_delay)
                    continue

                # Update last frame time
                last_frame_time = current_time
                frames_processed += 1

                # Log stats every 100 frames
                if frames_processed % 100 == 0:
                    uptime = datetime.now() - connection_start_time
                    logger.info(f"Stream uptime: {uptime}, Frames processed: {frames_processed}")

                # Frame skipping logic - run detection every N frames
                # Always send frames to output stream, but only run detection periodically
                run_detection = (frames == 0)
                if frames > 0:
                    frames -= 1
                else:
                    frames = detection_interval

                # Run inference only when needed
                if run_detection:
                    results = model(frame, imgsz=img_size)
                else:
                    results = []

                # Process detections (only if we ran detection)
                if run_detection:
                    # Clear previous annotations
                    last_annotations = []
                    
                    for r in results:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        scores = r.boxes.conf.cpu().numpy()
                        labels = r.boxes.cls.cpu().numpy().astype(int)

                        # Log all detections (even below threshold) for debugging
                        if len(boxes) > 0:
                            logger.debug(f"Raw detections: {len(boxes)} objects found")

                        # Collect all high-confidence detections for batch MQTT sending
                        detected_objects = []

                        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
                            # Convert class ID to name using conversion array
                            class_name = conversion[label] if 0 <= label < len(conversion) else f"class_{label}"

                            # Log detection even if below threshold
                            if score >= 0.3:  # Lower threshold for logging only
                                logger.info(f"Detected {class_name} with confidence {score:.2f}")

                            if score < 0.6:
                                continue

                            # Add to batch for MQTT
                            detected_objects.append({
                                'class': class_name,
                                'confidence': float(score)
                            })

                            color = score_to_bgr(score)
                            text = f"{class_name}: {score:.2f}"

                            # Cache annotation for redrawing on skipped frames
                            last_annotations.append({
                                'box': (int(x1), int(y1), int(x2), int(y2)),
                                'color': color,
                                'text': text
                            })

                            # Draw bounding box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                            # Draw text background for readability
                            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            tx1 = int(x1)
                            ty1 = int(y1) - th - 6
                            ty1 = max(0, ty1)
                            tx2 = tx1 + tw + 6
                            ty2 = ty1 + th + 6
                            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, -1)
                            cv2.putText(frame, text, (tx1 + 3, ty2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                        # Send all detections in a single MQTT message
                        if detected_objects and is_mqtt_connected():
                            send_batch_detection(detected_objects, "rtmp_stream")
                            logger.info(f"Sent MQTT batch: {len(detected_objects)} objects - {[obj['class'] for obj in detected_objects]}")

                        # Send push notifications to all registered iOS devices
                        if detected_objects:
                            send_detection_notification(detected_objects, "rtmp_stream")
                            logger.info(f"Triggered APNs notification for {len(detected_objects)} detections")
                else:
                    # Redraw cached annotations on frames where we skip detection
                    for ann in last_annotations:
                        x1, y1, x2, y2 = ann['box']
                        color = ann['color']
                        text = ann['text']
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw text background for readability
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        tx1 = x1
                        ty1 = y1 - th - 6
                        ty1 = max(0, ty1)
                        tx2 = tx1 + tw + 6
                        ty2 = ty1 + th + 6
                        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, -1)
                        cv2.putText(frame, text, (tx1 + 3, ty2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Stream annotated frame to Cloudflare RTMPS (always send, even if no detection)
                if ffmpeg_process is not None:
                    success = write_frame_to_ffmpeg(ffmpeg_process, frame)
                    if not success:
                        output_stream_failures += 1
                        if output_stream_failures >= 10:
                            logger.warning("FFmpeg output stream failed multiple times, attempting to restart...")
                            try:
                                if ffmpeg_process.stdin:
                                    ffmpeg_process.stdin.close()
                                ffmpeg_process.terminate()
                                ffmpeg_process.wait(timeout=5)
                            except:
                                pass
                            ffmpeg_process = None
                            output_stream_failures = 0
                    else:
                        output_stream_failures = 0
                else:
                    # Try to restart FFmpeg if we have stream dimensions
                    if stream_width and stream_height and frames_processed % 300 == 0:
                        logger.info("Attempting to restart FFmpeg output stream...")
                        ffmpeg_process = create_ffmpeg_process(
                            cloudflare_rtmps_url,
                            stream_width,
                            stream_height,
                            stream_fps
                        )

                # Display frame only if not in headless mode
                if not headless:
                    cv2.imshow('Security Camera', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User pressed 'q', exiting...")
                        break

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                if cap:
                    cap.release()
                    cap = None
                time.sleep(retry_delay)

    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        if cap:
            cap.release()
        
        # Cleanup FFmpeg process
        if ffmpeg_process is not None:
            logger.info("Closing FFmpeg output stream...")
            try:
                if ffmpeg_process.stdin:
                    ffmpeg_process.stdin.close()
                ffmpeg_process.terminate()
                ffmpeg_process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error closing FFmpeg process: {e}")
                try:
                    ffmpeg_process.kill()
                except:
                    pass
        
        if not headless:
            cv2.destroyAllWindows()

        # Shutdown APNs manager
        apns_manager = get_apns_manager()
        if apns_manager:
            apns_manager.shutdown()

        logger.info("Shutdown complete")

if __name__ == '__main__':
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Security Camera Object Detection System')
    # Get model path from environment variable, fallback to command line arg
    default_model = os.getenv('MODEL_PATH', '50epoch.pt')
    parser.add_argument('--model', default=default_model, help='Path to YOLO model file')
    parser.add_argument('--img-size', type=int, nargs=2, default=(320, 320), help='Inference image size (width height)')
    parser.add_argument('--device', default=None, help='Device to run on (cpu, cuda, mps). Auto-detected if not specified.')
    parser.add_argument('--headless', action='store_true', help='Run without display window (for Docker/server)')
    parser.add_argument('--max-retries', type=int, default=5, help='Max connection retry attempts before long wait')
    parser.add_argument('--retry-delay', type=int, default=5, help='Seconds to wait between retries')
    parser.add_argument('--detection-interval', type=int, default=30, help='Run detection every N frames (default: 60)')
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Security Camera Object Detection System Starting")
    logger.info("="*60)

    # Auto-detect device if not specified
    device = args.device if args.device else get_best_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model from: {args.model}")
    logger.info(f"Headless mode: {args.headless}")
    logger.info(f"Image size: {args.img_size}")

    try:
        model = YOLO(args.model)

        # Move model to the specified device
        if device != 'cpu':
            try:
                model.to(device)
                logger.info(f"Model moved to {device}")
            except Exception as e:
                logger.warning(f"Could not move model to {device}, falling back to CPU: {e}")
                device = 'cpu'

        logger.info("Starting camera stream processing...")
        run_webcam(
            model,
            tuple(args.img_size),
            headless=args.headless,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            detection_interval=args.detection_interval
        )

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("="*60)
        logger.info("Security Camera System Stopped")
        logger.info("="*60)
