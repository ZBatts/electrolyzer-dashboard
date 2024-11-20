import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tlr import TransferLinearRegression

class TLRModel:
    def __init__(self, data_path, interval_hours=250, total_hours=80000, lambda_=10.0, output_path=None):
        self.data_path = data_path
        self.interval_hours = interval_hours
        self.total_hours = total_hours
        self.num_intervals = total_hours // interval_hours
        self.lambda_ = lambda_
        self.output_path = output_path or os.path.join(os.path.dirname(data_path), 'tlr_fit_data.csv')
        self.data = pd.read_csv(data_path)
        self.fitting_results_TLR = pd.DataFrame()
        self.data['PredictedVoltage'] = np.nan

    def _prepare_data_interval(self, interval):
        """Extracts data for the specified interval."""
        start_hour = interval * self.interval_hours
        end_hour = (interval + 1) * self.interval_hours
        return self.data[(self.data['OperatingHours'] >= start_hour) & (self.data['OperatingHours'] < end_hour)]

    def _train_base_model(self, X, y):
        """Trains a linear regression model for initial coefficients."""
        base_model = LinearRegression().fit(X, y)
        return base_model.coef_

    def _fit_tlr_model(self, X, y, base_coefficients):
        """Fits the Transfer Linear Regression model for the interval."""
        tlr_model = TransferLinearRegression(lambda_=self.lambda_)
        tlr_model.fit(X, y, base_coefficients)
        return tlr_model

    def fit_intervals(self):
        """Fits the TLR model across intervals and collects metrics."""
        for interval in range(self.num_intervals):
            data_this_interval = self._prepare_data_interval(interval)
            if data_this_interval.empty:
                continue
            
            X = data_this_interval[['OperatingHours', 'Current', 'Temperature']]
            y = data_this_interval['Voltage']
            base_coefficients = self._train_base_model(X, y)
            tlr_model = self._fit_tlr_model(X, y, base_coefficients)
            
            # Store coefficients
            self.fitting_results_TLR.loc[interval, "OperatingHours_coef"] = tlr_model.coef_[0]
            self.fitting_results_TLR.loc[interval, "Current_coef"] = tlr_model.coef_[1]
            self.fitting_results_TLR.loc[interval, "Temperature_coef"] = tlr_model.coef_[2]
            self.fitting_results_TLR.loc[interval, "Intercept"] = tlr_model.coef_[-1]

            # Predictions and metrics
            y_pred = tlr_model.predict(X)
            self.data.loc[data_this_interval.index, 'PredictedVoltage'] = y_pred
            self.fitting_results_TLR.loc[interval, "MSE"] = mean_squared_error(y, y_pred)
            self.fitting_results_TLR.loc[interval, "R2"] = r2_score(y, y_pred)

    def calculate_overall_metrics(self):
        """Calculates overall performance metrics."""
        valid_indices = self.data['Voltage'].notna() & self.data['PredictedVoltage'].notna()
        overall_mse = mean_squared_error(self.data.loc[valid_indices, 'Voltage'], self.data.loc[valid_indices, 'PredictedVoltage'])
        overall_r2 = r2_score(self.data.loc[valid_indices, 'Voltage'], self.data.loc[valid_indices, 'PredictedVoltage'])
        print(f"Overall MSE: {overall_mse}")
        print(f"Overall R-squared: {overall_r2}")

    def merge_results(self):
        """Merges the fitting results with the main data DataFrame."""
        self.data['index'] = self.data.index
        self.fitting_results_TLR['index'] = self.fitting_results_TLR.index
        self.data = self.data.merge(self.fitting_results_TLR, on='index', how='outer').drop(columns=['index'])

    def save_results(self):
        """Saves the final merged DataFrame to a CSV file."""
        self.data.to_csv(self.output_path, header=True, index=False)
        print(f"Results saved to {self.output_path}")

    def run(self):
        """Runs the complete process of fitting the model and saving results."""
        self.fit_intervals()
        self.calculate_overall_metrics()
        self.merge_results()
        self.save_results()

model = TLRModel(data_path="data/synthetic_data.csv")
model.run()


# print(data['Voltage'])
# print(data['PredictedVoltage'])
# print(fitting_results_TLR.shape())

# # Plot the coefficients over time
# plt.figure(figsize=(14, 6))
# plt.plot(fitting_results_TLR.index * interval_hours, fitting_results_TLR["OperatingHours_coef"], label='Operating Hours Coef', marker='o')
# plt.plot(fitting_results_TLR.index * interval_hours, fitting_results_TLR["Current_coef"], label='Current Coef', marker='o')
# plt.plot(fitting_results_TLR.index * interval_hours, fitting_results_TLR["Temperature_coef"], label='Temperature Coef', marker='o')
# plt.plot(fitting_results_TLR.index * interval_hours, fitting_results_TLR["Intercept"], label='Intercept', marker='o')
# plt.xlabel('Operating Hours')
# plt.ylabel('Coefficient Value')
# plt.title('Coefficients Over Time')
# plt.legend()
# plt.grid()
# plt.show()

# # Plot actual vs predicted voltage over time
# plt.figure(figsize=(14, 6))
# plt.plot(data['OperatingHours'], data['Voltage'], label='Actual Voltage', color='blue')
# plt.plot(data['OperatingHours'], data['PredictedVoltage'], label='Predicted Voltage (TLR)', color='red', linestyle='--')
# plt.xlabel('Operating Hours')
# plt.ylabel('Voltage (V)')
# plt.title('Actual vs Predicted Voltage Over Time')
# plt.legend()
# plt.grid()
# plt.show()

# # Plot MSE over intervals
# plt.figure(figsize=(10, 5))
# plt.plot(fitting_results_TLR.index * interval_hours, fitting_results_TLR['MSE'], marker='o', color='purple')
# plt.xlabel('Operating Hours')
# plt.ylabel('Mean Squared Error')
# plt.title('MSE Over Time Intervals')
# plt.grid()
# plt.show()

# # Plot R-squared over intervals
# plt.figure(figsize=(10, 5))
# plt.plot(fitting_results_TLR.index * interval_hours, fitting_results_TLR['R2'], marker='o', color='green')
# plt.xlabel('Operating Hours')
# plt.ylabel('R-squared')
# plt.title('R-squared Over Time Intervals')
# plt.grid()
# plt.show()
