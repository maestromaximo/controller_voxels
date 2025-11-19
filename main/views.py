from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, DetailView, CreateView
from django.urls import reverse_lazy
from .models import SimulationConfig, SimulationRun
from .forms import SimulationConfigForm
from .services import run_simulation_service
import json

class DashboardView(ListView):
    model = SimulationRun
    template_name = 'main/dashboard.html'
    context_object_name = 'runs'
    ordering = ['-timestamp']

class CreateSimulationView(CreateView):
    model = SimulationConfig
    form_class = SimulationConfigForm
    template_name = 'main/config_form.html'
    
    def form_valid(self, form):
        # Save config
        self.object = form.save()
        
        # Create a Run instance
        run = SimulationRun.objects.create(config=self.object, status='pending')
        
        # Trigger simulation (synchronous for now)
        try:
            run_simulation_service(run.id)
        except Exception as e:
            run.status = 'failed'
            run.save()
            print(f"Simulation failed: {e}")
            
        return redirect('simulation_result', pk=run.id)

class SimulationResultView(DetailView):
    model = SimulationRun
    template_name = 'main/result.html'
    context_object_name = 'run'
