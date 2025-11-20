from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, DetailView, CreateView
from django.views.generic.edit import FormMixin
from django.views import View
from django.http import JsonResponse
from django.urls import reverse_lazy
from django.utils import timezone
import threading
from decimal import Decimal
from .models import SimulationConfig, SimulationRun, OptimizationJob
from .forms import SimulationConfigForm, SimulationRecomputeForm
from .services import run_simulation_service, run_optuna_job, OPTIMIZATION_TIME_LIMIT
from src.model import ThermalModel2D
import json

def _to_serializable(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value

class DashboardView(ListView):
    model = SimulationRun
    template_name = 'main/dashboard.html'
    context_object_name = 'runs'
    ordering = ['-timestamp']

class CreateSimulationView(CreateView):
    model = SimulationConfig
    form_class = SimulationConfigForm
    template_name = 'main/config_form.html'
    
    def _apply_bryson_to_data(self, data):
        temp_max = float(data.get('bryson_temp_max') or 5.0)
        heater_capacity = float(data.get('heater_heat_capacity') or 1.6)
        power_max = float(data.get('p_max') or 5.0)
        energy_max = heater_capacity * temp_max
        q_t, q_e, r = ThermalModel2D.bryson_weights(temp_max, energy_max, power_max)
        data['q_temp_weight'] = str(q_t)
        data['q_energy_weight'] = str(q_e)
        data['r_weight'] = str(r)
        return data

    def post(self, request, *args, **kwargs):
        if 'apply_bryson' in request.POST:
            data = request.POST.copy()
            data = self._apply_bryson_to_data(data)
            form = self.form_class(data)
            self.object = None
            return self.render_to_response(self.get_context_data(form=form))
        return super().post(request, *args, **kwargs)

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


class OptimizationStartView(View):
    def post(self, request, *args, **kwargs):
        form = SimulationConfigForm(request.POST)
        if not form.is_valid():
            return JsonResponse({'error': form.errors}, status=400)

        snapshot = {key: _to_serializable(value) for key, value in form.cleaned_data.items()}
        job = OptimizationJob.objects.create(
            status='pending',
            config_snapshot=snapshot,
        )

        thread = threading.Thread(target=run_optuna_job, args=(job.id,), daemon=True)
        thread.start()

        return JsonResponse({'job_id': job.id})


class OptimizationStatusView(View):
    def get(self, request, *args, **kwargs):
        job_id = request.GET.get('job_id')
        if not job_id:
            return JsonResponse({'error': 'job_id parameter is required'}, status=400)

        job = get_object_or_404(OptimizationJob, pk=job_id)
        percent = 0
        elapsed = 0.0
        if job.status in ('completed', 'failed'):
            percent = 100
        elif job.started_at:
            elapsed = (timezone.now() - job.started_at).total_seconds()
            percent = max(0, min(100, int((elapsed / OPTIMIZATION_TIME_LIMIT) * 100)))

        best_params = None
        if job.best_q_temp is not None and job.best_q_energy is not None and job.best_r is not None:
            best_params = {
                'q_temp_weight': job.best_q_temp,
                'q_energy_weight': job.best_q_energy,
                'r_weight': job.best_r,
                'score': job.best_score,
            }

        return JsonResponse({
            'status': job.status,
            'progress_log': job.progress_log,
            'best_params': best_params,
            'percent': percent,
            'elapsed': elapsed,
            'best_score': job.best_score,
        })

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
        if 'generate_animation' in request.POST:
            self.object.status = 'pending'
            self.object.save(update_fields=['status'])
            try:
                run_simulation_service(self.object.id, generate_animation=True)
            except Exception as e:
                self.object.status = 'failed'
                self.object.save(update_fields=['status'])
                print(f"Animation generation failed: {e}")
            return redirect('simulation_result', pk=self.object.id)
        if 'apply_bryson' in request.POST:
            config = self.object.config
            temp_max = config.bryson_temp_max
            energy_max = config.heater_heat_capacity * temp_max
            power_max = config.p_max
            q_t, q_e, r = ThermalModel2D.bryson_weights(temp_max, energy_max, power_max)
            form = self.form_class(initial={
                'q_temp_weight': q_t,
                'q_energy_weight': q_e,
                'r_weight': r,
                'duration': config.duration,
                'dt': config.dt,
            })
            return self.render_to_response(self.get_context_data(recompute_form=form))

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
