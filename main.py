# Install necessary packages first:
# pip install fastapi uvicorn python-multipart google-generativeai pillow python-dotenv

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import google.generativeai as genai
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing.")
genai.configure(api_key=api_key)

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

app = FastAPI()

@app.post("/ocr-translate")
async def ocr_and_translate(file: UploadFile = File(...)):
    try:
        # Step 1: Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Step 2: OCR extraction
        ocr_response = model.generate_content(
            [image, "Extract **all visible text** from this image **exactly as it appears**, without summarizing or interpreting. Do not translate."],
            generation_config={"temperature": 0.2}
        )
        english_text = ocr_response.text.strip()

        # Step 3: Translation
        translation_response = model.generate_content(
            [f"Translate this English text to Arabic:\n\n{english_text}"],
            generation_config={"temperature": 0.3}
        )
        translated_text = translation_response.text.strip()

        return JSONResponse(content={
            "english_text": english_text,
            "translated_text": translated_text
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

