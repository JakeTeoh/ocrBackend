import pytesseract
import cv2
import numpy as np
import requests
import subprocess
import tempfile
import os
import re
import json


def preprocess_image(imageUrl):
    response = requests.get(imageUrl)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    return denoised


def basicOcr(image):
    text = pytesseract.image_to_string(image)

    return text


def pdfToPng(pdf):
    # Create a temporary directory to store the converted file
    gs_path = r'C:\Program Files\gs\gs10.02.1\bin\gswin64.exe'
    response = requests.get(pdf)
    response.raise_for_status()

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "temp.pdf")
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        png_path = os.path.join(temp_dir, "output.png")

        command = [gs_path,
                   "-dNOPAUSE",
                   "-sDEVICE=png16m",
                   "-r600",
                   f"-sOutputFile={png_path}",
                   pdf_path,
                   "-dBATCH"]

        subprocess.run(command)
        image = cv2.imread(temp_dir)

        return image


def simpleOcr(request):
    if request.method == 'POST':
        json_data = json.loads(request.body.decode('utf-8'))
        imageUrl = json_data.get('imageUrl')

        processedImage = preprocess_image(imageUrl)
        text = basicOcr(processedImage)

        response_data = {
            'text': text,
            'status': 200
        }

        return response_data
    else:
        response_data = {
            "error": "Invalid request method.",
            "status": 405
        }

        return response_data


def extract_account_number(text):
    accountNumberPattern = r"\b\d{12}\b"
    account_number = re.findall(accountNumberPattern, text)
    return account_number


def extract_name(text):
    namePattern = r"\b[A-Z][A-Z]*(?: [A-Z][A-Z]*)*(?: (?:BIN|BINTI|A\/P\.?|A\/L) [A-Z][A-Z]*)?\b"
    name = re.findall(namePattern, text)
    return name
