from django.db import models
import json

class SimulationConfig(models.Model):
    COUPLING_CHOICES = [
        ('nearest', 'Nearest Neighbor'),
        ('distance', 'Distance Based (1/r^2)'),
    ]

    name = models.CharField(max_length=100, default="New Simulation")
    created_at = models.DateTimeField(auto_now_add=True)

    # Dimensions
    nx = models.IntegerField(default=5)
    ny = models.IntegerField(default=5)
    voxel_size = models.FloatField(default=0.06, help_text="Side length of a voxel (m)")

    # Physical Parameters
    density = models.FloatField(default=1.184, help_text="Density (kg/m^3)")
    specific_heat = models.FloatField(default=1005.0, help_text="Specific Heat (J/kg*K)")
    thermal_conductivity = models.FloatField(default=0.026, help_text="Thermal Conductivity (W/m*K)")
    heater_heat_capacity = models.FloatField(default=1.6, help_text="Heater Heat Capacity (J/K)")
    
    # Derived/Advanced Parameters (Optional overrides)
    coupling_mode = models.CharField(max_length=20, choices=COUPLING_CHOICES, default='nearest')
    
    # Control Parameters
    q_temp_weight = models.FloatField(default=100.0, help_text="Weight for Temperature Error")
    q_energy_weight = models.FloatField(default=0.1, help_text="Weight for Energy Error")
    r_weight = models.FloatField(default=0.001, help_text="Weight for Control Input")
    p_max = models.FloatField(default=5.0, help_text="Max Power (W)")

    # Simulation Settings
    duration = models.FloatField(default=1500.0, help_text="Simulation Duration (s)")
    dt = models.FloatField(default=0.05, help_text="Time Step (s)")
    
    # Target Temperature Matrix (stored as JSON list of lists)
    target_temp_matrix = models.JSONField(default=list)
    
    # Initial Temperature (Ambient)
    temp_ambient = models.FloatField(default=20.0, help_text="Ambient Temperature (C)")

    def __str__(self):
        return f"{self.name} ({self.nx}x{self.ny})"

class SimulationRun(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    config = models.ForeignKey(SimulationConfig, on_delete=models.CASCADE, related_name='runs')
    timestamp = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Results
    plot_temp_path = models.CharField(max_length=255, blank=True, null=True)
    plot_input_path = models.CharField(max_length=255, blank=True, null=True)
    plot_heatmap_path = models.CharField(max_length=255, blank=True, null=True)
    
    rms_error = models.FloatField(null=True, blank=True)
    settling_time = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return f"Run {self.id} - {self.status}"
