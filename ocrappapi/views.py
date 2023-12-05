import json
from io import BytesIO
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import pytesseract
import numpy as np
import cv2
import base64
import ocrappapi.utils as utils


@csrf_exempt
def image_preprocessing(request):
    if request.method == 'POST':
        json_data = json.loads(request.body.decode('utf-8'))
        image_data = json_data.get('image')
        if json_data.get('platform') == "web":
            image_data = image_data.split(",")[1]

        image_data = base64.b64decode(image_data)
        image_io = BytesIO(image_data)

        image = Image.open(image_io)

        openedImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        kernel_size = (int(json_data.get('kernelSizeA', 1)), int(json_data.get('kernelSizeB', 1)))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        if json_data.get('enableGrayScale'):
            gray_image = cv2.cvtColor(openedImage, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = cv2.cvtColor(openedImage, cv2.COLOR_BGR2RGB)

        if json_data.get('enableThreshold'):
            _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            threshold_image = gray_image

        if json_data.get('enableMorphological'):
            cleaned_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel,
                                             iterations=int(json_data.get('morphologicalIterations', 1)))
        else:
            cleaned_image = threshold_image

        if json_data.get('enableDilated'):
            dilated = cv2.dilate(cleaned_image, kernel, iterations=int(json_data.get('dilatedIterations', 1)))
        else:
            dilated = cleaned_image

        if json_data.get('enableErosion'):
            eroded = cv2.erode(dilated, kernel, iterations=int(json_data.get('erosionIterations', 1)))
        else:
            eroded = dilated

        if json_data.get('enableDenoising'):
            denoised_image = cv2.fastNlMeansDenoising(eroded, h=int(json_data.get('enoisingH', 10)))
        else:
            denoised_image = eroded

        enhanced = cv2.equalizeHist(denoised_image)

        _, processed_image_encoded = cv2.imencode('.png', enhanced)
        processed_image_base64 = base64.b64encode(processed_image_encoded).decode('utf-8')

        response_data = {
            'processed_image': processed_image_base64,
            'success': True
        }

        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=405)


@csrf_exempt
def ocrRequest(request):
    if request.method == 'POST':
        json_data = json.loads(request.body.decode('utf-8'))
        ocrMethod = json_data.get('ocrMethod')
        if ocrMethod == 'directToString':
            return JsonResponse(utils.simpleOcr(request))

    else:
        response_data = {
            "error": "Invalid request method.",
            "status": 405
        }
        return JsonResponse(response_data)


