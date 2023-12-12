import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import ocrappapi.utils as utils


@csrf_exempt
def ocrRequest(request):
    if request.method == 'POST':
        json_data = json.loads(request.body.decode('utf-8'))
        ocrMethod = json_data.get('ocrMethod')
        if ocrMethod == 'directToString':
            return JsonResponse(utils.simpleOcr(request))
        elif ocrMethod == 'template':
            return JsonResponse(utils.ocr_by_template(request))
        else:
            response_data = {
                "error": "Invalid OCR method.",
                "status": 405
            }
            return JsonResponse(response_data)

    else:
        response_data = {
            "error": "Invalid request method.",
            "status": 405
        }
        return JsonResponse(response_data)


@csrf_exempt
def newTemplate(request):
    if request.method == 'POST':
        return JsonResponse(utils.new_template(request))
    else:
        response_data = {
            "error": "Invalid request method.",
            "status": 405
        }
        return JsonResponse(response_data)

