from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('pergunta/', views.rag_consultas, name='rag_consultas'),
]