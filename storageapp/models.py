from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import models

class Upload(models.Model):
    owner = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    file = models.FileField(upload_to="uploads/")  # goes to S3 via DEFAULT_FILE_STORAGE
    created_at = models.DateTimeField(auto_now_add=True)
    predicted = models.CharField(max_length=4, blank=True)  # e.g., "7"
    prob = models.FloatField(null=True, blank=True)
    model_used = models.CharField(max_length=20, blank=True)

    def filename(self):
        return self.file.name.split("/")[-1]

    def __str__(self):
        return f"{self.owner} • {self.filename()}"
