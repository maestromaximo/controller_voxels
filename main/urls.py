from django.urls import path
from . import views

urlpatterns = [
    path('', views.DashboardView.as_view(), name='dashboard'),
    path('create/', views.CreateSimulationView.as_view(), name='create_simulation'),
    path('result/<int:pk>/', views.SimulationResultView.as_view(), name='simulation_result'),
    path('optimize/start/', views.OptimizationStartView.as_view(), name='optimize_start'),
    path('optimize/status/', views.OptimizationStatusView.as_view(), name='optimize_status'),
]

