import os
import json
import requests
import gradio as gr
from paddleocr import PaddleOCR
import tempfile
from PIL import Image
import pandas as pd
import re
import datetime # For unique filenames

# --- CONFIGURATION (Ensure these match your environment) ---
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b" # Or "llama3" depending on your Ollama setup

# --- INITIALIZE OCR MODEL ---
print("Initializing PaddleOCR model...")
try:
    # use_gpu=False if you don't have GPU setup for PaddlePaddle
    ocr_reader = PaddleOCR(lang='en', use_textline_orientation=True, show_log=False, use_gpu=False)
    print("✅ PaddleOCR model initialized.")
except Exception as e:
    print(f"❌ Failed to initialize PaddleOCR. Error: {e}. Please check your PaddleOCR installation and GPU settings.")
    ocr_reader = None # Set to None so the app can still run, but OCR functions will fail gracefully

# --- HELPER FUNCTIONS ---
def extract_text_from_image(image_path):
    """Extracts all text from an image using PaddleOCR."""
    if ocr_reader is None:
        return "OCR model not initialized.", "OCR model not initialized."

    print(f"  - Performing OCR on {image_path}...")
    try:
        result = ocr_reader.ocr(image_path, cls=True)
        if result and result[0] is not None:
            texts = [line[1][0] for line in result[0]]
            print("  - ✅ OCR successful.")
            return "\n".join(texts), None # Return text and no error
        return "", "No text found by OCR."
    except Exception as e:
        print(f"  - ❌ OCR failed: {e}")
        return "", f"OCR failed: {e}"

def get_json_from_local_llm(full_text, image_filename="unknown"):
    """Sends the full OCR text to a local LLM for structuring."""
    if not full_text or "OCR failed" in full_text:
        return {"error": "No text extracted or OCR failed."}, "No text to process."

    print(f"  - Sending text from {image_filename} to local LLM for parsing...")
    prompt = f"""
    You are an expert data extraction model. From the provided OCR text of a document, extract the key information into a clean JSON object.
    
    Identify fields such as 'invoice_number', 'invoice_date' (format YYYY-MM-DD if possible), 'buyer_name', 'seller_name', 'total_amount', 'currency'.
    Also, extract 'items' as a list of objects, where each object has 'description', 'quantity', 'unit_price', 'line_total'.
    If an item field is missing or cannot be confidently extracted, use null.
    If a top-level field is not found, it can be omitted or set to null.

    **OCR Text:**
    {full_text}

    Return only the valid JSON object.
    """
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False}
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        response_text = response.json()['response']
        
        # Robustly parse JSON from LLM response
        try:
            parsed_json = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(1))
            else:
                parsed_json = json.loads(response_text) # Try parsing directly as a last resort
        
        print("  - ✅ LLM parsing successful.")
        return parsed_json, None
    except requests.exceptions.ConnectionError:
        print(f"  - ❌ LLM connection failed. Is Ollama running at {OLLAMA_ENDPOINT}?")
        return {"error": f"Could not connect to Ollama. Is it running at {OLLAMA_ENDPOINT}?"}, "LLM connection failed."
    except requests.exceptions.Timeout:
        print("  - ❌ LLM request timed out.")
        return {"error": "LLM request timed out. The model might be slow or the prompt too long."}, "LLM request timed out."
    except Exception as e:
        print(f"  - ❌ LLM parsing failed: {e}. Raw response: {response_text}")
        return {"error": f"LLM parsing failed: {e}. Raw response: {response_text}"}, f"LLM parsing failed: {e}"

def flatten_json_for_csv(json_data, image_filename):
    """
    Flattens a single structured JSON object into a list of dictionaries,
    one for each invoice line item, duplicating common invoice fields.
    Adds the original filename for traceability.
    """
    flattened_records = []
    
    # Extract top-level invoice details
    invoice_details = {
        'original_filename': image_filename,
        'invoice_number': json_data.get('invoice_number'),
        'invoice_date': json_data.get('invoice_date'),
        'buyer_name': json_data.get('buyer_name'),
        'seller_name': json_data.get('seller_name'),
        'total_amount': json_data.get('total_amount'),
        'currency': json_data.get('currency'),
    }

    items = json_data.get('items', [])
    if not items:
        # If no items, still add the invoice details as one record
        flattened_records.append({**invoice_details, 'item_description': None, 'item_quantity': None, 'item_unit_price': None, 'item_line_total': None})
    else:
        for item in items:
            record = {**invoice_details} # Copy invoice details for each item
            record['item_description'] = item.get('description')
            record['item_quantity'] = item.get('quantity')
            record['item_unit_price'] = item.get('unit_price')
            record['item_line_total'] = item.get('line_total')
            flattened_records.append(record)
            
    return flattened_records

def process_invoices(image_files):
    """Main function to process multiple uploaded invoice images."""
    if not image_files:
        return "", "{}", pd.DataFrame(), None, "Please upload one or more images."
    
    all_raw_texts = []
    all_structured_jsons = []
    all_flattened_data = []
    
    status_messages = []

    # Create a temporary directory for CSV output
    temp_dir = tempfile.mkdtemp()
    
    for i, image_file in enumerate(image_files):
        filename = os.path.basename(image_file.name)
        status_messages.append(f"Processing {filename} ({i+1}/{len(image_files)})...")
        print(f"\n--- Processing: {filename} ---")

        # 1. Extract all text from the image
        raw_text, ocr_error = extract_text_from_image(image_file.name)
        all_raw_texts.append(f"--- {filename} ---\n{raw_text or ocr_error}\n\n")

        if ocr_error:
            status_messages.append(f"  - ❌ OCR for {filename} failed: {ocr_error}")
            structured_data = {"filename": filename, "error": ocr_error}
            all_structured_jsons.append(structured_data)
            all_flattened_data.extend(flatten_json_for_csv(structured_data, filename)) # Still add to CSV for traceability
            continue
        
        # 2. Send the raw text to the local LLM for structuring
        structured_data, llm_error = get_json_from_local_llm(raw_text, filename)
        all_structured_jsons.append(structured_data)
        
        if llm_error:
            status_messages.append(f"  - ❌ LLM for {filename} failed: {llm_error}")
            # Add error to structured data if it's not already there
            if 'error' not in structured_data:
                 structured_data['error'] = llm_error
            all_flattened_data.extend(flatten_json_for_csv(structured_data, filename))
            continue

        status_messages.append(f"  - ✅ Successfully processed {filename}")
        all_flattened_data.extend(flatten_json_for_csv(structured_data, filename))
            
    combined_raw_text = "".join(all_raw_texts)
    combined_json_output = json.dumps(all_structured_jsons, indent=2)

    # 3. Create a single DataFrame and CSV for all processed invoices
    if all_flattened_data:
        df = pd.DataFrame(all_flattened_data)
        
        # Define a consistent order of columns for better readability and EDA
        # Ensure all columns exist, fill missing with None
        desired_columns = [
            'original_filename', 'invoice_number', 'invoice_date', 
            'seller_name', 'buyer_name', 'total_amount', 'currency',
            'item_description', 'item_quantity', 'item_unit_price', 'item_line_total',
            'error' # Include error column if it was added
        ]
        
        # Add any columns from df that are not in desired_columns (e.g., if LLM extracts new fields)
        final_columns = []
        for col in desired_columns:
            if col in df.columns and col not in final_columns:
                final_columns.append(col)
        for col in df.columns:
            if col not in final_columns:
                final_columns.append(col)
        
        df = df.reindex(columns=final_columns)

        # Generate a unique filename for the CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(temp_dir, f"invoices_summary_{timestamp}.csv")
        df.to_csv(csv_filename, index=False)
        
        print(f"\n--- Generated CSV: {csv_filename} ---")
        
        return (
            combined_raw_text, 
            combined_json_output, 
            df, 
            csv_filename, 
            "\n".join(status_messages) + "\n\nAll invoices processed. Download the combined CSV below."
        )
    else:
        return (
            combined_raw_text, 
            combined_json_output, 
            pd.DataFrame({"Status": ["No data processed or all failed"]}), 
            None, 
            "\n".join(status_messages) + "\n\nNo valid data was extracted to create a CSV."
        )


# --- GRADIO INTERFACE ---
print("\n🚀 Starting Gradio application...")

if ocr_reader is None:
    gr.Warning("PaddleOCR model failed to initialize. OCR functionality will not work.")

with gr.Blocks(title="Invoice Processor with LLM & CSV Export") as demo:
    gr.Markdown(
        """
        # 🧾 Multi-Invoice Processor with Local LLM (Ollama + PaddleOCR)
        Upload one or more invoice images (PNG, JPG) and let PaddleOCR extract the raw text,
        then a local LLM (via Ollama) will structure the data.
        
        The extracted data from all invoices will be combined into a single view and
        can be downloaded as a CSV file for further analysis (EDA).
        
        **Note:** Ensure Ollama is running and the specified model is downloaded (`ollama run llama3:8b`).
        """
    )

    with gr.Row():
        # Allow multiple file uploads
        image_input = gr.File(
            type="filepath", 
            label="Upload Invoice Images (Multiple allowed)", 
            file_types=["image"], 
            file_count="multiple" # Key change here
        )
        with gr.Column():
            process_btn = gr.Button("Process Invoices", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False, lines=5)
    
    gr.Markdown("---")

    with gr.Tabs():
        with gr.TabItem("Raw OCR Texts (Combined)"):
            raw_text_output = gr.Textbox(label="Extracted Text (from PaddleOCR)", lines=10, interactive=False)
        with gr.TabItem("Structured JSON Outputs (Combined)"):
            json_output_display = gr.Code(label="Structured Data (from LLM)", language="json", interactive=False, lines=15)
        with gr.TabItem("Combined Data (CSV Preview)"):
            csv_dataframe_display = gr.DataFrame(label="Combined Invoice Data (for EDA)", row_count=(5, 'dynamic'), col_count=(5, 'dynamic'))
            csv_download_btn = gr.File(label="Download Combined CSV", file_count="single", type="filepath", interactive=False)
            
    # Connect the button click to the processing function
    process_btn.click(
        fn=process_invoices,
        inputs=image_input,
        outputs=[raw_text_output, json_output_display, csv_dataframe_display, csv_download_btn, status_output]
    )

demo.launch()
print("✅ Gradio app launched. Open your browser to the URL provided above.")