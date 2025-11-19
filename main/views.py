from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, DetailView, CreateView
from django.views.generic.edit import FormMixin
from django.urls import reverse_lazy
from .models import SimulationConfig, SimulationRun
from .forms import SimulationConfigForm, SimulationRecomputeForm
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

class SimulationResultView(FormMixin, DetailView):
    model = SimulationRun
    template_name = 'main/result.html'
    context_object_name = 'run'
    form_class = SimulationRecomputeForm

    def get_success_url(self):
        return self.request.path

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs.setdefault('initial', self.get_initial())
        return kwargs

    def get_initial(self):
        run = self.get_object()
        return {
            'q_temp_weight': run.config.q_temp_weight,
            'q_energy_weight': run.config.q_energy_weight,
            'r_weight': run.config.r_weight,
            'duration': run.config.duration,
            'dt': run.config.dt,
        }

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if 'recompute_form' not in context:
            context['recompute_form'] = self.get_form()
        return context

    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        form = self.get_form()
        if form.is_valid():
            config = self.object.config
            config.q_temp_weight = form.cleaned_data['q_temp_weight']
            config.q_energy_weight = form.cleaned_data['q_energy_weight']
            config.r_weight = form.cleaned_data['r_weight']
            config.duration = form.cleaned_data['duration']
            config.dt = form.cleaned_data['dt']
            config.save()

            self.object.status = 'pending'
            self.object.save()

            try:
                run_simulation_service(self.object.id)
            except Exception as e:
                self.object.status = 'failed'
                self.object.save()
                print(f"Simulation failed: {e}")

            return redirect('simulation_result', pk=self.object.id)
        else:
            return self.get(request, *args, **kwargs)
