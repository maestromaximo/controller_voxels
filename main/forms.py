from django import forms
from .models import SimulationConfig

class SimulationConfigForm(forms.ModelForm):
    class Meta:
        model = SimulationConfig
        fields = '__all__'
        widgets = {
            'target_temp_matrix': forms.HiddenInput(),
        }

class SimulationRecomputeForm(forms.ModelForm):
    class Meta:
        model = SimulationConfig
        fields = ['q_temp_weight', 'q_energy_weight', 'r_weight', 'duration', 'dt']

