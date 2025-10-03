# main_parser.py
import os
import json
import cv2
import requests
import pandas as pd
from paddleocr import PaddleOCR
from pydantic import ValidationError
from schemas import Invoice, ProductItem
from typing import Optional # <-- THIS LINE IS ADDED

# --- CONFIGURATION ---
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("FATAL ERROR: config.json not found.")
    exit()

OLLAMA_MODEL = config.get('ollama_model')
OLLAMA_ENDPOINT = config.get('ollama_endpoint')
TEMPLATES = config.get('templates')

class OCRProcessor:
    """Handles image pre-processing and OCR."""
    def __init__(self):
        self.ocr_reader = PaddleOCR(lang='en', use_textline_orientation=True)

    def get_full_text(self, image_path: str) -> str:
        # A more robust way to get full text
        try:
            result = self.ocr_reader.ocr(image_path, cls=True)
            if result and result[0]:
                return " ".join([line[1][0] for line in result[0]])
        except Exception:
            return ""
        return ""


    def extract_regions(self, image_path: str, region_definitions: dict) -> dict:
        """Extracts text from predefined regions of an image."""
        image = cv2.imread(image_path)
        if image is None: return {}
        h, w = image.shape[:2]
        
        ocr_data = {}
        for name, (y1, y2, x1, x2) in region_definitions.items():
            region_img = image[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]
            text = self._ocr_on_region(region_img)
            ocr_data[name] = text
        return ocr_data

    def _ocr_on_region(self, region_img):
        try:
            result = self.ocr_reader.ocr(region_img, cls=True)
            return "\n".join([line[1][0] for line in result[0]]) if result and result[0] else ""
        except Exception:
            return ""

class LLMParser:
    """Handles parsing and structuring text using a local LLM."""
    def __init__(self, model: str, endpoint: str):
        self.model = model
        self.endpoint = endpoint
        self.schema_json_str = json.dumps(Invoice.model_json_schema(), indent=2)

    def parse(self, ocr_data: dict) -> Optional[dict]:
        """Sends OCR data to the LLM and validates the output."""
        full_text = ocr_data.get('full_text', '')
        prompt = self._create_prompt(full_text)
        
        try:
            payload = {"model": self.model, "prompt": prompt, "format": "json", "stream": False}
            response = requests.post(self.endpoint, json=payload, timeout=180)
            response.raise_for_status()
            response_text = response.json()['response']

            validated_invoice = Invoice.model_validate_json(response_text)
            print("  - ✅ LLM parsing and validation successful.")
            return validated_invoice.model_dump()
        except (ValidationError, requests.RequestException, Exception) as e:
            print(f"  - ❌ Parsing failed: {e}")
            return None

    def _create_prompt(self, ocr_text: str) -> str:
        return f"""
        You are an expert data extraction model. Analyze the noisy OCR text from an invoice and convert it into a clean JSON object that strictly conforms to the provided Pydantic schema.

        **Instructions:**
        - Extract all data accurately. Pay close attention to line items, quantities, rates, and amounts.
        - Convert all dates to YYYY-MM-DD format.
        - If a field is not found, use null.
        - Your output MUST be a single, valid JSON object that matches the schema.

        **Pydantic Schema:**
        ```json
        {self.schema_json_str}
        ```

        **Noisy OCR Text:**
        {ocr_text}

        Return only the JSON object.
        """

def flatten_data(invoice_data: dict) -> list:
    """Flattens the invoice JSON for CSV conversion."""
    base_info = {k: v for k, v in invoice_data.items() if k != 'line_items'}
    
    if not invoice_data.get('line_items'):
        return [base_info]
        
    flat_list = []
    for item in invoice_data['line_items']:
        row = base_info.copy()
        row.update(item)
        flat_list.append(row)
    return flat_list

def main():
    """Main execution function."""
    print("Initializing Open-Source Invoice Parser...")
    ocr_processor = OCRProcessor()
    llm_parser = LLMParser(model=OLLAMA_MODEL, endpoint=OLLAMA_ENDPOINT)
    all_invoice_data = []

    for image_file in os.listdir("invoices"):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"\n--- Processing: {image_file} ---")
            image_path = os.path.join("invoices", image_file)
            
            # Simple template detection
            full_text_sample = ocr_processor.get_full_text(image_path)
            template_name = None
            if "BAGGA" in full_text_sample:
                template_name = "hotel_bagga"
            elif "BEATLE" in full_text_sample:
                template_name = "the_beatle"

            if not template_name:
                print(f"  - ⚠️  Could not identify a template for {image_file}. Using full text.")
                regional_ocr_data = {'full_text': full_text_sample}
            else:
                print(f"  - Detected template: {template_name}")
                regional_ocr_data = ocr_processor.extract_regions(image_path, TEMPLATES[template_name])
            
            invoice_json = llm_parser.parse(regional_ocr_data)
            
            if invoice_json:
                all_invoice_data.extend(flatten_data(invoice_json))
    
    if all_invoice_data:
        df = pd.DataFrame(all_invoice_data)
        df.to_csv("output/results.csv", index=False, encoding='utf-8')
        print("\n✅ All invoices processed. Results saved to output/results.csv")

if __name__ == "__main__":
    main()