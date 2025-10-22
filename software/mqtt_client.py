"""
MQTT Client Module for Security Camera
Handles MQTT connection and message publishing for object detection alerts
"""
import paho.mqtt.client as mqtt
import json
from datetime import datetime
import threading
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from config.env file in the same directory as this script
config_path = Path(__file__).parent / 'config.env'
load_dotenv(config_path)

# MQTT Configuration from environment variables
MQTT_BROKER = os.getenv('MQTT_BROKER')
MQTT_PORT = int(os.getenv('MQTT_PORT'))
MQTT_USERNAME = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
MQTT_TOPIC = os.getenv('MQTT_TOPIC')

class MQTTClient:
    """MQTT Client for sending detection messages"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.connection_lock = threading.Lock()
        
    def setup_connection(self):
        """Setup MQTT client and establish connection"""
        try:
            self.client = mqtt.Client()
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            
            # Set up callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_publish = self._on_publish
            
            # Connect to broker
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
            
            # Wait a moment for connection to establish
            time.sleep(1)
            return self.connected
            
        except Exception as e:
            print(f"❌ Error setting up MQTT connection: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for when client connects to broker"""
        if rc == 0:
            self.connected = True
            print("✅ Connected to MQTT broker")
        else:
            self.connected = False
            print(f"❌ Failed to connect to MQTT broker. Return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for when client disconnects from broker"""
        self.connected = False
        print("Disconnected from MQTT broker")
    
    def _on_publish(self, client, userdata, mid):
        """Callback for when message is published"""
        print(f"📤 Message published successfully (mid: {mid})")
    
    def send_detection(self, classes, confidences, source="rtmp_stream"):
        """Send object detection message to MQTT broker"""
        if not self.connected or self.client is None:
            print("❌ MQTT client not connected. Cannot send message.")
            return False
        
        objects = [{'class': classes[i], 'confidence': confidences[i]} for i in range(len(classes))]
        # Create detection message
        message = {
            "objects": objects,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Publish message
            result = self.client.publish(MQTT_TOPIC, json.dumps(message))
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"📤 Sent detection: {objects}")
                return True
            else:
                print(f"❌ Failed to publish message. Return code: {result.rc}")
                return False
                
        except Exception as e:
            print(f"❌ Error sending MQTT message: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            print("Disconnected from MQTT broker")

# Global MQTT client instance
mqtt_client = MQTTClient()

def initialize_mqtt():
    """Initialize MQTT connection"""
    return mqtt_client.setup_connection()

def send_detection_message(class_name, confidence, source="rtmp_stream"):
    """Send detection message using global MQTT client"""
    return mqtt_client.send_detection(class_name, confidence, source)

def disconnect_mqtt():
    """Disconnect from MQTT broker"""
    mqtt_client.disconnect()

def is_mqtt_connected():
    """Check if MQTT client is connected"""
    return mqtt_client.connected
