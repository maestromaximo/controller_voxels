from django import forms
from .models import SimulationConfig

class SimulationConfigForm(forms.ModelForm):
    class Meta:
        model = SimulationConfig
        fields = '__all__'
        widgets = {
            'target_temp_matrix': forms.HiddenInput(),
        }

