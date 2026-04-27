from django.urls import path
from . import views

urlpatterns = [
    path('', views.nutrition_interface, name='nutrition_interface'),
]