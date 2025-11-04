#!/usr/bin/env python3
"""
APNs Manager Module
Handles device token registration via MQTT and sends push notifications
"""

import paho.mqtt.client as mqtt
import json
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from threading import Lock
from simple_apns import APNSClient, Payload

# Load environment variables
config_path = Path(__file__).parent / "config.env"
load_dotenv(config_path)

logger = logging.getLogger(__name__)


class APNsManager:
    """
    Manages device tokens and sends APNs push notifications
    """

    def __init__(self):
        """Initialize APNs manager with configuration"""
        self.device_tokens = {}  # {client_id: device_token}
        self.tokens_lock = Lock()

        # APNs Configuration
        self.apns_key_path = os.getenv("APNS_KEY_PATH")
        self.apns_key_id = os.getenv("APNS_KEY_ID")
        self.apns_team_id = os.getenv("APNS_TEAM_ID")
        self.apns_bundle_id = os.getenv("APNS_BUNDLE_ID")
        self.apns_environment = os.getenv("APNS_ENVIRONMENT", "development")

        # Notification cooldown
        self.cooldown_seconds = int(os.getenv("NOTIFICATION_COOLDOWN", "60"))
        self.last_notification_time = {}  # {detection_type: timestamp}
        self.cooldown_lock = Lock()

        # MQTT Configuration for device registration
        self.mqtt_broker = os.getenv("MQTT_BROKER")
        self.mqtt_port = int(os.getenv("MQTT_PORT", 1883))
        self.mqtt_username = os.getenv("MQTT_USERNAME")
        self.mqtt_password = os.getenv("MQTT_PASSWORD")
        self.mqtt_topic = "app/device/register/#"

        # APNs Client
        self.apns_client = None
        self._initialize_apns_client()

        # MQTT Client for device registration
        self.mqtt_client = None
        self._initialize_mqtt_subscriber()

    def _initialize_apns_client(self):
        """Initialize APNs client with token-based authentication"""
        try:
            if not all(
                [
                    self.apns_key_path,
                    self.apns_key_id,
                    self.apns_team_id,
                    self.apns_bundle_id,
                ]
            ):
                logger.warning(
                    "APNs configuration incomplete. Push notifications will be disabled."
                )
                logger.warning(
                    "Required: APNS_KEY_PATH, APNS_KEY_ID, APNS_TEAM_ID, APNS_BUNDLE_ID"
                )
                return

            # Check if key file exists
            key_path = Path(self.apns_key_path)
            if not key_path.exists():
                logger.error(f"APNs key file not found: {self.apns_key_path}")
                return

            # Determine server (sandbox or production)
            use_sandbox = self.apns_environment.lower() == "development"

            # Initialize APNs client with simple_apns
            self.apns_client = APNSClient(
                team_id=self.apns_team_id,
                auth_key_id=self.apns_key_id,
                auth_key_path=str(key_path),
                bundle_id=self.apns_bundle_id,
                use_sandbox=use_sandbox,
            )

            logger.info(f"✅ APNs client initialized successfully")
            logger.info(f"   Environment: {self.apns_environment}")
            logger.info(f"   Bundle ID: {self.apns_bundle_id}")
            logger.info(f"   Key ID: {self.apns_key_id}")
            logger.info(f"   Team ID: {self.apns_team_id}")
            logger.info(f"   Cooldown: {self.cooldown_seconds}s")

        except Exception as e:
            logger.error(f"Failed to initialize APNs client: {e}", exc_info=True)
            self.apns_client = None

    def _initialize_mqtt_subscriber(self):
        """Initialize MQTT subscriber for device token registration"""
        try:
            if not all(
                [
                    self.mqtt_broker,
                    self.mqtt_port,
                    self.mqtt_username,
                    self.mqtt_password,
                ]
            ):
                logger.warning(
                    "MQTT configuration incomplete. Device registration will be disabled."
                )
                return

            # Create MQTT client for device registration
            self.mqtt_client = mqtt.Client(client_id="python-apns-manager")
            self.mqtt_client.username_pw_set(self.mqtt_username, self.mqtt_password)

            # Set callbacks
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_device_registration

            # Connect to broker
            logger.info(
                f"Connecting to MQTT broker for device registration: {self.mqtt_broker}:{self.mqtt_port}"
            )
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)

            # Start background loop
            self.mqtt_client.loop_start()

        except Exception as e:
            logger.error(f"Failed to initialize MQTT subscriber: {e}", exc_info=True)
            self.mqtt_client = None

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback when MQTT connection is established"""
        if rc == 0:
            logger.info(f"✅ Connected to MQTT broker for device registration")
            # Subscribe to device registration topic with QoS 1
            client.subscribe(self.mqtt_topic, qos=1)
            logger.info(f"📡 Subscribed to: {self.mqtt_topic}")
        else:
            logger.error(f"❌ MQTT connection failed with code {rc}")

    def _on_device_registration(self, client, userdata, msg):
        """Callback when device registration message is received"""
        try:
            # Extract client_id from topic
            # Example: "app/device/register/iOS-Vision-ABC123" → "iOS-Vision-ABC123"
            topic_parts = msg.topic.split("/")
            client_id = topic_parts[-1] if len(topic_parts) >= 3 else "unknown"

            # Parse JSON payload
            payload = msg.payload.decode("utf-8")
            data = json.loads(payload)

            device_token = data.get("device_token")
            client_id_from_payload = data.get("client_id")
            platform = data.get("platform", "ios")

            if not device_token:
                logger.warning(
                    f"Received registration without device_token from {client_id}"
                )
                return

            # Store device token in memory
            with self.tokens_lock:
                self.device_tokens[client_id] = device_token

            logger.info(f"📥 Device registered:")
            logger.info(f"   Client ID: {client_id}")
            logger.info(f"   Platform: {platform}")
            logger.info(f"   Token: {device_token[:20]}...")
            logger.info(f"   Retained: {msg.retain}")
            logger.info(f"   Total devices: {len(self.device_tokens)}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in device registration: {e}")
        except Exception as e:
            logger.error(f"Error processing device registration: {e}", exc_info=True)

    def _should_send_notification(self, detection_type):
        """
        Check if notification should be sent based on cooldown period
        """
        with self.cooldown_lock:
            current_time = time.time()
            last_time = self.last_notification_time.get(detection_type, 0)

            if current_time - last_time >= self.cooldown_seconds:
                self.last_notification_time[detection_type] = current_time
                return True
            else:
                remaining = int(self.cooldown_seconds - (current_time - last_time))
                logger.debug(
                    f"Notification cooldown active for '{detection_type}'. {remaining}s remaining."
                )
                return False

    def send_push_notification(self, detected_objects, source="camera"):
        """
        Send push notification to all registered devices

        Args:
            detected_objects: List of detected objects [{'class': 'person', 'confidence': 0.95}, ...]
            source: Source identifier (default: "camera")
        """
        if not self.apns_client:
            logger.debug("APNs client not initialized. Skipping push notification.")
            return

        if not detected_objects:
            return

        # Create detection summary
        detection_summary = ", ".join([obj["class"] for obj in detected_objects])

        # Use sorted unique classes as cooldown key to handle order variations
        # Example: ["person", "dog"] and ["dog", "person"] both become "dog, person"
        unique_classes = sorted(set([obj["class"] for obj in detected_objects]))
        detection_type = ", ".join(unique_classes)

        # Check cooldown
        if not self._should_send_notification(detection_type):
            return

        # Get all device tokens
        with self.tokens_lock:
            tokens = list(self.device_tokens.values())

        if not tokens:
            logger.debug("No device tokens registered. Skipping push notification.")
            return

        # Create notification payload
        title = "🎥 Detection Alert"
        body = f"Detected: {detection_summary}"

        # Custom data
        custom_data = {
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "detections": detected_objects,
        }

        # Create APNs payload
        payload = Payload()
        payload.set_alert(title=title, body=body)
        payload.set_sound("default")
        payload.set_badge(1)
        # Add custom data fields
        for key, value in custom_data.items():
            payload.add_custom_data(key, value)

        # Send to all devices
        success_count = 0
        failed_count = 0

        for device_token in tokens:
            try:
                self.apns_client.send_notification(
                    device_token=device_token,
                    payload=payload,
                )
                success_count += 1
                logger.debug(f"✅ Push sent to {device_token[:20]}...")

            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to send push to {device_token[:20]}...: {e}")

        logger.info(
            f"📲 Push notifications sent: {success_count} success, {failed_count} failed"
        )
        logger.info(f"   Message: {body}")

    def send_system_notification(self, status, message, source="camera"):
        """
        Send system status notification (camera online/offline)

        Args:
            status: 'online' or 'offline'
            message: Detailed message
            source: Source identifier (default: "camera")
        """
        if not self.apns_client:
            logger.debug("APNs client not initialized. Skipping system notification.")
            return

        # Get all device tokens
        with self.tokens_lock:
            tokens = list(self.device_tokens.values())

        if not tokens:
            logger.debug("No device tokens registered. Skipping system notification.")
            return

        # Create notification based on status
        if status == "offline":
            title = "⚠️ Camera Offline"
            emoji = "🔴"
        elif status == "online":
            title = "✅ Camera Online"
            emoji = "🟢"
        else:
            title = "📡 Camera Status"
            emoji = "ℹ️"

        body = f"{emoji} {message}"

        # Custom data
        custom_data = {
            "source": source,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "type": "system_status",
        }

        # Create APNs payload
        payload = Payload()
        payload.set_alert(title=title, body=body)
        payload.set_sound("default")
        payload.set_badge(1)
        # Add custom data fields
        for key, value in custom_data.items():
            payload.add_custom_data(key, value)

        # Send to all devices
        success_count = 0
        failed_count = 0

        for device_token in tokens:
            try:
                self.apns_client.send_notification(
                    device_token=device_token,
                    payload=payload,
                )
                success_count += 1
                logger.debug(f"✅ System notification sent to {device_token[:20]}...")

            except Exception as e:
                failed_count += 1
                logger.warning(
                    f"Failed to send system notification to {device_token[:20]}...: {e}"
                )

        logger.info(
            f"📲 System notifications sent: {success_count} success, {failed_count} failed"
        )
        logger.info(f"   Status: {status.upper()}")
        logger.info(f"   Message: {body}")

    def get_registered_devices_count(self):
        """Get count of registered devices"""
        with self.tokens_lock:
            return len(self.device_tokens)

    def shutdown(self):
        """Cleanup resources"""
        logger.info("Shutting down APNs manager...")
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        logger.info("APNs manager stopped")


# Global instance
_apns_manager = None


def initialize_apns_manager():
    """Initialize global APNs manager instance"""
    global _apns_manager
    if _apns_manager is None:
        _apns_manager = APNsManager()
    return _apns_manager


def get_apns_manager():
    """Get global APNs manager instance"""
    global _apns_manager
    if _apns_manager is None:
        _apns_manager = initialize_apns_manager()
    return _apns_manager


def send_detection_notification(detected_objects, source="camera"):
    """
    Convenience function to send detection notification

    Args:
        detected_objects: List of detected objects [{'class': 'person', 'confidence': 0.95}, ...]
        source: Source identifier (default: "camera")
    """
    manager = get_apns_manager()
    if manager:
        manager.send_push_notification(detected_objects, source)


def send_system_status_notification(status, message, source="camera"):
    """
    Convenience function to send system status notification

    Args:
        status: 'online' or 'offline'
        message: Detailed message
        source: Source identifier (default: "camera")
    """
    manager = get_apns_manager()
    if manager:
        manager.send_system_notification(status, message, source)
