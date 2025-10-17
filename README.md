# Security Camera Team 3


MQTT Message Format:

objects = [{'class': classes[i], 'confidence': confidences[i]} for i in range(len(classes))]
        
message = {
    "objects": objects,
    "timestamp": datetime.now().isoformat()
}