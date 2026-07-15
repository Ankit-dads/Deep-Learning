from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict_view, name='predict'),
    path('bulk/', views.bulk_predict_view, name='bulk_predict'),
    path('history/', views.history_view, name='history'),
    path('history/delete/<int:pk>/', views.delete_prediction, name='delete_prediction'),
    path('history/clear/', views.clear_history, name='clear_history'),
    path('stats/', views.stats_view, name='stats'),
    path('export/csv/', views.export_csv, name='export_csv'),
    path('api/predict/', views.api_predict, name='api_predict'),
]
