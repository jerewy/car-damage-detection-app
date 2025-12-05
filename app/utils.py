from ultralytics import YOLO
from groq import Groq
from .config import MODEL_PATH, GROQ_API_KEY
import cv2
import json
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model once
logger.info(f"Loading model from {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def run_inference(image_path, output_path):
    """
    Runs YOLO inference on the image.
    Saves the annotated image to output_path.
    Returns a dictionary with detections and metadata.
    """
    start_time = time.time()
    
    # Run inference
    results = model(image_path, imgsz=640, conf=0.4)
    
    end_time = time.time()
    processing_time_ms = (end_time - start_time) * 1000
    
    result = results[0]
    
    # Save annotated image
    annotated_frame = result.plot()
    cv2.imwrite(str(output_path), annotated_frame)
    
    # Extract detections for LLM and UI
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = result.names[cls_id]
        confidence = float(box.conf[0])
        
        # Simple location logic based on box center
        x_center = box.xywh[0][0]
        img_width = result.orig_shape[1]
        if x_center < img_width / 3:
            location = "Left Side"
        elif x_center > 2 * img_width / 3:
            location = "Right Side"
        else:
            location = "Center"

        detections.append({
            "class": class_name,
            "confidence": confidence,
            "location": location,
            "severity": "Medium" 
        })
        
    return {
        "detections": detections,
        "processing_time": round(processing_time_ms, 2),
        "count": len(detections),
        "model_info": "YOLO11s"
    }

def generate_insurance_report(detection_data, make, model_name, year):
    """
    Generates a structured JSON report using Groq API based on detections.
    """
    if not GROQ_API_KEY:
        return {"error": "Groq API Key not configured"}
        
    client = Groq(api_key=GROQ_API_KEY)
    
    vehicle_info = f"{year} {make} {model_name}"
    detections = detection_data.get("detections", [])
    det_str = json.dumps(detections, indent=2)
    
    # Fallback if no detections
    if not detections:
        return {
            "summary": f"No specific damage patterns were detected by the automated system on the {vehicle_info}.",
            "severity_level": "Low",
            "severity_reasoning": "No visible dents, scratches, or tears identified by AI.",
            "repair_urgency": "None",
            "urgency_details": "Vehicle appears to be in good condition based on visual inspection.",
            "cost_range": "$0",
            "cost_details": "No repairs needed.",
            "recommended_action": "Routine maintenance.",
            "immediate_actions": ["Verify visual inspection manually.", "Wash vehicle to ensure no hidden marks."],
            "secondary_actions": ["Keep regular service schedule."]
        }

    prompt = f"""
    You are an expert car insurance adjuster using an AI Damage Assessment System.
    
    Vehicle: {vehicle_info}
    Detected Damages: {det_str}
    
    Generate a professional damage assessment report in valid JSON format ONLY. 
    Do not include any markdown formatting like ```json ... ```.
    
    The JSON must have these exact keys:
    {{
        "summary": "Concise summary of findings (e.g., 'Our system identified 2 dents...')",
        "severity_level": "Low, Medium, or High",
        "severity_reasoning": "Brief explanation of severity",
        "repair_urgency": "Non-urgent, Urgent, or Critical",
        "urgency_details": "Why is it urgent? (e.g., 'Does not affect safety')",
        "cost_range": "Estimated cost (e.g., '$300 - $800 USD')",
        "cost_details": "Breakdown or method (e.g., 'Paintless Dent Removal')",
        "recommended_action": "Main recommendation (e.g., 'Professional inspection')",
        "immediate_actions": ["Action 1", "Action 2", "Action 3"],
        "secondary_actions": ["Action 1", "Action 2"]
    }}
    """
        
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"} # Enforce JSON mode
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return {
            "summary": "Error generating detailed report.",
            "severity_level": "Unknown",
            "severity_reasoning": str(e),
            "repair_urgency": "Unknown",
            "urgency_details": "Contact support.",
            "cost_range": "Unknown",
            "cost_details": "Unknown",
            "recommended_action": "Manual Inspection",
            "immediate_actions": ["Contact support"],
            "secondary_actions": []
        }