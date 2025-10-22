# Security Camera Vision - Docker Setup

A security camera system with YOLO object detection and MQTT integration, containerized with Docker.

## Features

- Real-time object detection using YOLOv8
- RTMP/RTMPS stream support
- MQTT integration for IoT alerts
- Dockerized for easy deployment
- Detects: cars, cats, dogs, and persons

## Prerequisites

- Docker (20.10 or higher)
- Docker Compose (1.29 or higher)
- A valid `config.env` file with your credentials

## Quick Start

### 1. Configure Environment Variables

Copy the example config and fill in your values:

```bash
cp .env.example config.env
```

Edit `config.env`:

```env
# MQTT Broker Configuration
MQTT_BROKER=your_mqtt_broker_address
MQTT_PORT=1883
MQTT_USERNAME=your_username
MQTT_PASSWORD=your_password
MQTT_TOPIC=camera/detections

# RTMP Stream Configuration
RTMP_STREAM_URL=rtmps://your-stream-url
```

### 2. Configure Docker Parameters (Optional)

You can customize which config and model files to use by creating a `.env` file:

```bash
cp .env.docker .env
```

Edit `.env` to customize paths:

```env
# Path to your config file
CONFIG_PATH=./config.env

# Path to your model file
MODEL_PATH=./50epoch.pt
```

**Default values:**
- Config: `./config.env`
- Model: `./50epoch.pt`

### 3. Build and Run with Docker Compose

```bash
# Build the Docker image
docker-compose build

# Run the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### 4. Alternative: Run with Custom Parameters

You can override config and model paths without creating a `.env` file:

```bash
# Use custom model and config
CONFIG_PATH=./my-config.env MODEL_PATH=./my-model.pt docker-compose up -d

# Use different model only
MODEL_PATH=./10epoch.pt docker-compose up -d
```

### 5. Alternative: Build and Run with Docker Commands

```bash
# Build the image
docker build -t security-camera-vision .

# Run with custom config and model
docker run -d \
  --name security-camera \
  --restart unless-stopped \
  -v $(pwd)/config.env:/app/config.env:ro \
  -v $(pwd)/50epoch.pt:/app/models/model.pt:ro \
  -e MODEL_PATH=/app/models/model.pt \
  security-camera-vision

# Run with different model
docker run -d \
  --name security-camera \
  --restart unless-stopped \
  -v $(pwd)/config.env:/app/config.env:ro \
  -v $(pwd)/my-model.pt:/app/models/model.pt:ro \
  -e MODEL_PATH=/app/models/model.pt \
  security-camera-vision

# View logs
docker logs -f security-camera

# Stop and remove
docker stop security-camera
docker rm security-camera
```

## Configuration

### Docker Parameters

You can configure which files to use when starting the container:

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `CONFIG_PATH` | Path to config.env file | `./config.env` | `./config.production.env` |
| `MODEL_PATH` | Path to YOLO model file | `./50epoch.pt` | `./my-model.pt` |

**Usage examples:**

```bash
# Use default paths (config.env and 50epoch.pt)
docker-compose up -d

# Use custom model
MODEL_PATH=./10epoch.pt docker-compose up -d

# Use custom config and model
CONFIG_PATH=./prod.env MODEL_PATH=./best-model.pt docker-compose up -d

# Or set in .env file (copy from .env.docker)
cp .env.docker .env
# Edit .env, then:
docker-compose up -d
```

### Application Environment Variables

All configuration is done via `config.env`:

| Variable | Description | Example |
|----------|-------------|---------|
| `MQTT_BROKER` | MQTT broker hostname/IP | `mqtt.example.com` |
| `MQTT_PORT` | MQTT broker port | `1883` |
| `MQTT_USERNAME` | MQTT username | `user` |
| `MQTT_PASSWORD` | MQTT password | `pass123` |
| `MQTT_TOPIC` | MQTT topic for detections | `camera/detections` |
| `RTMP_STREAM_URL` | RTMP/RTMPS stream URL | `rtmps://...` |

### Model Configuration

The model file is mounted as a volume and can be easily changed without rebuilding the image.

**To use a different model:**

1. **Via environment variable:**
   ```bash
   MODEL_PATH=./your-model.pt docker-compose up -d
   ```

2. **Via .env file:**
   ```bash
   # In .env file
   MODEL_PATH=./your-model.pt
   ```

3. **Via docker command:**
   ```bash
   docker run -v $(pwd)/your-model.pt:/app/models/model.pt:ro ...
   ```

**No rebuild required!** Just change the volume mount and restart the container.

## Deployment Options

### Option 1: Docker Compose (Recommended)

Best for local development and simple deployments.

```bash
docker-compose up -d
```

### Option 2: Docker Swarm

For production clusters:

```bash
docker stack deploy -c docker-compose.yml security-camera
```

### Option 3: Kubernetes

Create a deployment using the Docker image.

## Troubleshooting

### Container won't start

Check logs:
```bash
docker-compose logs
```

Common issues:
- Missing `config.env` file
- Invalid RTMP stream URL
- MQTT connection issues

### Can't connect to MQTT broker

1. Verify broker is accessible from container
2. Check firewall rules
3. Test with: `docker exec -it security-camera-vision ping mqtt.broker.com`

### Stream not opening

1. Verify RTMP URL is correct
2. Check if stream requires authentication
3. Test with: `docker exec -it security-camera-vision python -c "import cv2; cap = cv2.VideoCapture('your-url'); print(cap.isOpened())"`

## Resource Usage

Default limits (configurable in docker-compose.yml):
- CPU: 2 cores max, 1 core reserved
- Memory: 4GB max, 2GB reserved

Adjust based on your hardware:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
```

## Monitoring

View real-time logs:
```bash
docker-compose logs -f security-camera
```

Check resource usage:
```bash
docker stats security-camera-vision
```

## Security Notes

- `config.env` contains sensitive credentials - never commit to git
- The file is mounted read-only in the container
- Use secrets management in production (Docker Secrets, Kubernetes Secrets)

## Development

### Local Development without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python server.py
```

### Rebuild after changes

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## MQTT Message Format

Detection messages are published in JSON format:

```json
{
  "objects": [
    {"class": "person", "confidence": 0.85},
    {"class": "car", "confidence": 0.92}
  ],
  "timestamp": "2025-10-22T10:30:45.123456"
}
```

## License

[Your License Here]

## Support

For issues and questions, please open an issue on GitHub.
