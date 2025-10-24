from django.urls import path
from . import views

urlpatterns = [
    path("upload/", views.upload_view, name="upload"),
    path("my-files/", views.list_view, name="list"),
    path("delete/<int:pk>/", views.delete_view, name="delete"),

    # two classify endpoints
    path("classify/<int:pk>/cnn/", views.classify_cnn, name="classify_cnn"),
    path("classify/<int:pk>/fc/",  views.classify_fc,  name="classify_fc"),
]
