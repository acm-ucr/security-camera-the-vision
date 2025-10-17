#!/usr/bin/env python3
"""
Test script to verify MQTT connection and message sending
"""
from mqtt_client import initialize_mqtt, send_detection_message, is_mqtt_connected, disconnect_mqtt
import time

def test_mqtt_connection():
    """Test MQTT connection and message sending"""
    print("🧪 Testing MQTT connection...")
    
    # Initialize connection
    if initialize_mqtt():
        print("✅ MQTT connection successful")
        
        # Test sending a detection message
        print("📤 Sending test detection message...")
        success = send_detection_message(['person', 'cat', 'dog'], [0.85, 0.7, 0.6])
        
        if success:
            print("✅ Test message sent successfully")
        else:
            print("❌ Failed to send test message")
        
        # Wait a moment for message to be sent
        time.sleep(2)
        
        # Disconnect
        disconnect_mqtt()
        print("✅ MQTT test completed")
        
    else:
        print("❌ MQTT connection failed")

if __name__ == "__main__":
    test_mqtt_connection()
