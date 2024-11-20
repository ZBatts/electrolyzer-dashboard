import numpy as np
import pandas as pd
import os

class SyntheticDataGenerator:
    def __init__(self, 
                 total_hours=80000, 
                 interval_hours=250, 
                 num_points=1000, 
                 initial_voltage=1.8, 
                 degradation_rate=0.000005, 
                 current_effect_growth_rate=0.000000001, 
                 temperature_effect_growth_rate=0.000000005, 
                 output_folder='data'):
        
        self.total_hours = total_hours
        self.interval_hours = interval_hours
        self.num_points = num_points
        self.num_intervals = total_hours // interval_hours
        self.initial_voltage = initial_voltage
        self.degradation_rate = degradation_rate
        self.current_effect_growth_rate = current_effect_growth_rate
        self.temperature_effect_growth_rate = temperature_effect_growth_rate
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_folder = output_folder
        self.file_path = os.path.join(self.file_dir, output_folder, 'synthetic_data.csv')
        self.data = pd.DataFrame()

    def generate_time_series(self):
        """Generates time series for operating hours, current, and temperature."""
        np.random.seed(42)
        operating_hours = np.linspace(0, self.total_hours, self.num_points)
        current = np.random.uniform(0, 1000, self.num_points)
        temperature = np.random.normal(60, 5, self.num_points)
        return operating_hours, current, temperature

    def calculate_voltage(self, operating_hours, current, temperature):
        """Calculates voltage using time-varying coefficients and simulates effects."""
        voltage = []
        current_effects = []
        temperature_effects = []

        for i in range(self.num_points):
            current_effect = 0.0001 + self.current_effect_growth_rate * operating_hours[i]
            temperature_effect = 0.005 + self.temperature_effect_growth_rate * operating_hours[i]
            voltage_value = (
                self.initial_voltage +
                (self.degradation_rate * operating_hours[i]) +
                current_effect * current[i] +
                temperature_effect * (temperature[i] - 60) +
                np.random.normal(0, 0.01)
            )
            voltage.append(voltage_value)
            current_effects.append(current_effect)
            temperature_effects.append(temperature_effect)
        
        return voltage, current_effects, temperature_effects

    def generate_data(self):
        """Generates the full synthetic dataset and stores it in a DataFrame."""
        operating_hours, current, temperature = self.generate_time_series()
        voltage, current_effects, temperature_effects = self.calculate_voltage(operating_hours, current, temperature)

        self.data = pd.DataFrame({
            'OperatingHours': operating_hours,
            'Current': current,
            'Temperature': temperature,
            'Voltage': voltage,
            'CurrentEffect': current_effects,
            'TemperatureEffect': temperature_effects
        })

    def save_data(self):
        """Saves the generated data to a CSV file."""
        os.makedirs(self.output_folder, exist_ok=True)
        self.data.to_csv(self.file_path, header=True, index=False)
        print(f"Data saved to {self.file_path}")

    def run(self):
        """Runs the data generation and saves the data to a CSV file."""
        self.generate_data()
        self.save_data()

generator = SyntheticDataGenerator(
    total_hours=80000, 
    initial_voltage=1.8, 
    degradation_rate=0.00001, 
    current_effect_growth_rate=0.000000002, 
    temperature_effect_growth_rate=0.000000006
)
generator.run()
