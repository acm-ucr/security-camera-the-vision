import paho.mqtt.client as mqtt
import json
import os
import time
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from simple_apns import APNSClient, Payload

# Load environment variables
config_path = Path(__file__).parent / 'config.env'
load_dotenv(config_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Store collected device tokens
device_tokens = {}

def on_connect(client, userdata, flags, rc):
    """MQTT connection callback"""
    if rc == 0:
        logger.info("✅ Connected to MQTT broker")
        # Subscribe to device registration topic
        client.subscribe("app/device/register/#", qos=1)
        logger.info("📡 Subscribed to: app/device/register/#")
        logger.info("⏳ Waiting 3 seconds to collect device tokens...")
    else:
        logger.error(f"❌ Connection failed with code {rc}")

def on_message(client, userdata, msg):
    """MQTT message callback"""
    try:
        # Extract client_id from topic
        topic_parts = msg.topic.split('/')
        client_id = topic_parts[-1] if len(topic_parts) >= 3 else "unknown"

        # Parse JSON payload
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)

        device_token = data.get('device_token')
        platform = data.get('platform', 'ios')

        if device_token:
            device_tokens[client_id] = {
                'token': device_token,
                'platform': platform,
                'retained': msg.retain
            }

            logger.info(f"📥 Device found:")
            logger.info(f"   Client ID: {client_id}")
            logger.info(f"   Platform: {platform}")
            logger.info(f"   Token: {device_token[:20]}...")
            logger.info(f"   Retained: {msg.retain}")

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Error processing message: {e}")

def collect_device_tokens():
    """Connect to MQTT and collect device tokens"""
    logger.info("="*60)
    logger.info("Step 1: Collecting Device Tokens from MQTT")
    logger.info("="*60)

    # MQTT Configuration
    mqtt_broker = os.getenv('MQTT_BROKER')
    mqtt_port = int(os.getenv('MQTT_PORT', 1883))
    mqtt_username = os.getenv('MQTT_USERNAME')
    mqtt_password = os.getenv('MQTT_PASSWORD')

    if not all([mqtt_broker, mqtt_port, mqtt_username, mqtt_password]):
        logger.error("❌ MQTT configuration incomplete in config.env")
        return False

    logger.info(f"🔌 Connecting to MQTT broker: {mqtt_broker}:{mqtt_port}")

    # Create MQTT client
    client = mqtt.Client(client_id="apns-test-script")
    client.username_pw_set(mqtt_username, mqtt_password)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(mqtt_broker, mqtt_port, keepalive=60)

        # Run loop for 3 seconds to collect retained messages
        client.loop_start()
        time.sleep(3)
        client.loop_stop()
        client.disconnect()

        logger.info(f"\n📊 Total devices found: {len(device_tokens)}")
        return True

    except Exception as e:
        logger.error(f"❌ MQTT connection error: {e}")
        return False

def send_test_push_notification():
    """Send test push notification to all collected devices"""
    logger.info("\n" + "="*60)
    logger.info("Step 2: Sending Test Push Notifications")
    logger.info("="*60)

    if not device_tokens:
        logger.warning("⚠️  No device tokens found. Please ensure:")
        logger.warning("   1. iOS app is running and registered")
        logger.warning("   2. Device sent retained message to MQTT")
        logger.warning("   3. MQTT broker persistence is enabled")
        return

    # APNs Configuration
    apns_key_path = os.getenv('APNS_KEY_PATH')
    apns_key_id = os.getenv('APNS_KEY_ID')
    apns_team_id = os.getenv('APNS_TEAM_ID')
    apns_bundle_id = os.getenv('APNS_BUNDLE_ID')
    apns_environment = os.getenv('APNS_ENVIRONMENT', 'development')

    if not all([apns_key_path, apns_key_id, apns_team_id, apns_bundle_id]):
        logger.error("❌ APNs configuration incomplete in config.env")
        logger.error("Required: APNS_KEY_PATH, APNS_KEY_ID, APNS_TEAM_ID, APNS_BUNDLE_ID")
        return

    # Check if key file exists
    key_path = Path(apns_key_path)
    if not key_path.exists():
        logger.error(f"❌ APNs key file not found: {apns_key_path}")
        logger.error(f"   Current working directory: {Path.cwd()}")
        logger.error(f"   Looking for: {key_path.absolute()}")
        return

    logger.info(f"📱 APNs Configuration:")
    logger.info(f"   Key Path: {apns_key_path}")
    logger.info(f"   Key ID: {apns_key_id}")
    logger.info(f"   Team ID: {apns_team_id}")
    logger.info(f"   Bundle ID: {apns_bundle_id}")
    logger.info(f"   Environment: {apns_environment}")

    # Initialize APNs client
    try:
        use_sandbox = (apns_environment.lower() == 'development')

        apns_client = APNSClient(
            team_id=apns_team_id,
            auth_key_id=apns_key_id,
            auth_key_path=str(key_path),
            bundle_id=apns_bundle_id,
            use_sandbox=use_sandbox
        )

        logger.info(f"✅ APNs client initialized")
        logger.info(f"   Server: {'api.sandbox.push.apple.com' if use_sandbox else 'api.push.apple.com'}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize APNs client: {e}", exc_info=True)
        return

    # Create test notification payload
    payload = Payload()
    payload.set_alert(title='🧪 Test Notification', body='This is a test push from APNs test script')
    payload.set_sound('default')
    payload.set_badge(1)
    payload.add_custom_data('test', True)
    payload.add_custom_data('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
    payload.add_custom_data('source', 'test_apns.py')

    logger.info(f"\n📲 Sending test notifications to {len(device_tokens)} device(s)...\n")

    # Send to all devices
    success_count = 0
    failed_count = 0

    for client_id, device_info in device_tokens.items():
        device_token = device_info['token']

        logger.info(f"📤 Sending to {client_id}...")
        logger.info(f"   Token: {device_token[:20]}...")

        try:
            apns_client.send_notification(
                device_token=device_token,
                payload=payload
            )
            success_count += 1
            logger.info(f"   ✅ Success!")

        except Exception as e:
            failed_count += 1
            logger.error(f"   ❌ Failed: {e}")
            logger.error(f"   Possible reasons:")
            logger.error(f"      - Invalid device token")
            logger.error(f"      - Environment mismatch (dev token sent to prod, or vice versa)")
            logger.error(f"      - Bundle ID mismatch")
            logger.error(f"      - Network connectivity issue")

        logger.info("")  # Empty line for readability

    # Summary
    logger.info("="*60)
    logger.info("📊 Test Summary")
    logger.info("="*60)
    logger.info(f"Total devices: {len(device_tokens)}")
    logger.info(f"✅ Success: {success_count}")
    logger.info(f"❌ Failed: {failed_count}")

    if success_count > 0:
        logger.info("\n🎉 Check your iOS device for the test notification!")

    if failed_count > 0:
        logger.info("\n⚠️  Troubleshooting tips:")
        logger.info("   1. Verify APNS_ENVIRONMENT matches your iOS app build")
        logger.info("      - development: Xcode debug builds")
        logger.info("      - production: TestFlight or App Store builds")
        logger.info("   2. Verify APNS_BUNDLE_ID matches your iOS app")
        logger.info("   3. Ensure device token is from the same app/environment")
        logger.info("   4. Check network connectivity to APNs servers")

def main():
    """Main test function"""
    logger.info("🧪 APNs Push Notification Test Script")
    logger.info("="*60)

    # Step 1: Collect device tokens from MQTT
    if not collect_device_tokens():
        logger.error("\n❌ Failed to collect device tokens. Exiting.")
        return

    if not device_tokens:
        logger.warning("\n⚠️  No device tokens found.")
        logger.warning("\nPlease ensure:")
        logger.warning("   1. iOS app has registered device token via MQTT")
        logger.warning("   2. MQTT message was sent with retained=true")
        logger.warning("   3. MQTT broker has persistence enabled")
        logger.warning("\nExiting.")
        return

    # Step 2: Send test push notification
    send_test_push_notification()

    logger.info("\n✅ Test script completed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}", exc_info=True)
