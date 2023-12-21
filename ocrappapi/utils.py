import base64
import io
import pytesseract
import cv2
import numpy as np
import requests
import subprocess
import tempfile
import os
import re
import json
import csv

from PIL import Image


def basicOcr(image, psm):
    custom_config = f'--oem 3 --psm {psm}'
    text = pytesseract.image_to_string(image, config=custom_config)

    return text


def ocrToDataframe(image, psm):
    custom_config = f'--oem 3 --psm {psm}'
    df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME, config=custom_config)

    return df


def ocrToData(image, psm):
    custom_config = f'--oem 3 --psm {psm}'
    data = pytesseract.image_to_data(image, config=custom_config)

    return data


def validate_psm(psm_value):
    try:
        psm = int(psm_value)
        if 0 <= psm <= 13:
            return psm
    except (ValueError, TypeError):
        pass
    return 11


def extract_text_by_block(block_num, dataframe):
    block_data = dataframe[(dataframe['block_num'] == block_num) & (dataframe['conf'] != -1)]

    return ' '.join(block_data['text'].astype(str))


def extract_text_by_regex(pattern, df):
    text_data = ' '.join(df['text'].dropna())
    matches = re.findall(pattern, text_data)
    return ' '.join(matches)


def preprocess_image(gray, preProcessingConfig):
    if not preProcessingConfig:
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        # If preProcessingConfig is None or empty, return the original image
        return gray

    if preProcessingConfig.get("denoised", True):
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    kernelSize = preProcessingConfig.get("kernelSize", 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

    if preProcessingConfig.get("morphologyEx", False):
        iterations = preProcessingConfig.get("morphologyEx_iterations", 1)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)

    if preProcessingConfig.get("dilated", False):
        iterations = preProcessingConfig.get("dilated_iterations", 1)
        gray = cv2.dilate(gray, kernel, iterations=iterations)

    if preProcessingConfig.get("eroded", False):
        iterations = preProcessingConfig.get("eroded_iterations", 1)
        gray = cv2.erode(gray, kernel, iterations=iterations)

    if preProcessingConfig.get("threshold", False):
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    if preProcessingConfig.get("equalizeHist", False):
        gray = cv2.equalizeHist(gray)

    return gray


def ocr_by_template(request):
    json_data = json.loads(request.body.decode('utf-8'))
    imageUrl = json_data.get('imageUrl')
    config = json_data.get('config')
    preProcessingConfig = json_data.get('preProcessingConfig', {})
    psm = validate_psm(json_data.get('psm', 11))

    response_data = {
        'text': [],
        'status': 200
    }

    response = requests.get(imageUrl)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed_image = preprocess_image(gray, preProcessingConfig)

    df = ocrToDataframe(processed_image, psm)
    for item in config:
        if item.get("extractMethod") == "block":
            text = extract_text_by_block(item.get("value"), df)
        elif item.get("extractMethod") == "regex":
            text = extract_text_by_regex(item.get("value"), df)
        field = item.get("fieldName")
        response_data['text'].append({
            'Field': field,
            'Text': text
        })

    return response_data


def simpleOcr(request):
    json_data = json.loads(request.body.decode('utf-8'))
    imageUrl = json_data.get('imageUrl')
    preProcessingConfig = json_data.get('preProcessingConfig', {})
    psm = validate_psm(json_data.get('psm', 11))

    response = requests.get(imageUrl)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed_image = preprocess_image(gray, preProcessingConfig)
    text = basicOcr(processed_image, psm)

    response_data = {
        'text': text,
        'status': 200
    }

    return response_data


def new_template(request):
    json_data = json.loads(request.body.decode('utf-8'))
    imageData = base64.b64decode(json_data.get('image'))
    preProcessingConfig = json_data.get('preProcessingConfig', {})
    psm = validate_psm(json_data.get('psm', 11))

    np_arr = np.frombuffer(imageData, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed_image = preprocess_image(gray, preProcessingConfig)

    data = ocrToData(processed_image, psm)

    data_stream = io.StringIO(data)
    csv_reader = csv.DictReader(data_stream, delimiter='\t')

    structured_data = [row for row in csv_reader]

    response_data = {
        "data": structured_data,
        "status": 200
    }

    return response_data
