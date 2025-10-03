import os
import json
import requests
import base64

# --- CONFIGURATION ---
INPUT_DIR = "invoices"
OUTPUT_DIR = "output"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
# Use the open-source Vision Language Model
OLLAMA_MODEL = "llava" 

# --- HELPER FUNCTIONS ---

def encode_image_to_base64(image_path):
    """Encodes an image file into a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return None

def get_json_from_llava(image_base64, image_filename):
    """Sends an image and a prompt to the LLaVA model for parsing."""
    print("  - Sending image to LLaVA for parsing...")
    
    prompt = """
    You are an expert data extraction model specializing in hotel invoices.
    Analyze the provided invoice image and extract the following information into a single, valid JSON object:
    - hotel_name
    - invoice_number
    - invoice_date
    - guest_name
    - company_name
    - total_amount

    If a field is not present in the image, set its value to null. Return only the JSON object.
    """
    
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "images": [image_base64] # The key difference: sending the image
        }
        
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=300) # Increased timeout for VLM
        response.raise_for_status()
        response_text = response.json()['response']
        print("  - ✅ LLaVA parsing successful.")
        return json.loads(response_text)
        
    except requests.exceptions.Timeout:
        print(f"  - ❌ LLaVA parsing failed: The request timed out. The model may be too slow on your hardware.")
        return None
    except Exception as e:
        print(f"  - ❌ LLaVA parsing failed: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # We will process one of your sample images
    # Change this to 'invoice_57_page-0001.jpg' to test the other one
    image_to_process = "invoice_28_page-0001.jpg"
    
    print(f"\n--- Processing: {image_to_process} ---")
    image_path = os.path.join(INPUT_DIR, image_to_process)
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        exit()
        
    # 1. Encode the image
    b64_image = encode_image_to_base64(image_path)
    
    if b64_image:
        # 2. Send the image to LLaVA for structuring
        structured_json = get_json_from_llava(b64_image, image_to_process)
        
        # 3. Save the result
        if structured_json:
            output_filename = os.path.splitext(image_to_process)[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_path, 'w') as f:
                json.dump(structured_json, f, indent=2)
            print(f"  - ✅ Successfully saved output to {output_path}")
            
            print("\n--- Final Extracted JSON ---")
            print(json.dumps(structured_json, indent=2))