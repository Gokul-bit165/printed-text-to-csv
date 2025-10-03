import os
import json
import cv2
import requests
from paddleocr import PaddleOCR

# --- CONFIGURATION ---
INPUT_DIR = "invoices"
OUTPUT_DIR = "output"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b" 

# --- INITIALIZE OCR MODEL ---
print("Initializing PaddleOCR model...")
# Ensure you have the GPU version of paddlepaddle installed for this to use the GPU
try:
    ocr_reader = PaddleOCR(lang='en', use_textline_orientation=True)
    print("✅ Model initialized.")
except Exception as e:
    print(f"❌ Failed to initialize PaddleOCR. Error: {e}")
    exit()

# --- HELPER FUNCTIONS ---
def extract_text_from_image(image_path):
    """Extracts all text from an image using PaddleOCR."""
    print("  - Performing OCR on the entire image...")
    try:
        result = ocr_reader.ocr(image_path, cls=True)
        if result and result[0] is not None:
            texts = [line[1][0] for line in result[0]]
            print("  - ✅ OCR successful.")
            return "\n".join(texts)
        return ""
    except Exception as e:
        print(f"  - ❌ OCR failed: {e}")
        return ""

def get_json_from_local_llm(full_text):
    """Sends the full OCR text to a local LLM for structuring."""
    print("  - Sending text to local LLM for parsing...")
    prompt = f"""
    You are an expert data extraction model. From the provided OCR text of a document, extract the key information into a clean JSON object.
    
    Identify fields such as 'invoice_number', 'invoice_date', 'buyer_name', 'seller_name', 'total_amount', and any listed products.

    **OCR Text:**
    {full_text}

    Return only the valid JSON object.
    """
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False}
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120) # Added timeout
        response.raise_for_status()
        response_text = response.json()['response']
        print("  - ✅ LLM parsing successful.")
        return json.loads(response_text)
    except Exception as e:
        print(f"  - ❌ LLM parsing failed: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Process all images in the input directory
    for image_file in os.listdir(INPUT_DIR):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"\n--- Processing: {image_file} ---")
            image_path = os.path.join(INPUT_DIR, image_file)

            # 1. Extract all text from the image
            raw_text = extract_text_from_image(image_path)
            
            if raw_text:
                # 2. Send the raw text to the local LLM for structuring
                structured_json = get_json_from_local_llm(raw_text)
                
                # 3. Save the result
                if structured_json:
                    output_filename = os.path.splitext(image_file)[0] + ".json"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    with open(output_path, 'w') as f:
                        json.dump(structured_json, f, indent=2)
                    print(f"  - ✅ Successfully saved output to {output_path}")