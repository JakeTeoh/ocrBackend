from django.db import models

class ScannedImage(models.Model):
    image = models.ImageField(upload_to='scanned_images/')
    extracted_text = models.TextField(null=True, blank=True)

    class Meta:
        app_label = 'ocrapp'

