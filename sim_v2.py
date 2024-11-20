import os
import numpy as np
import pandas as pd
import time

from cell_model import MassTransportOverpotential, OhmicOverpotential, elx_tcprops, MemConductivity, ActivationOverpotential, ELX_pem_cell, h2o_tcprop, h2_tcprop, o2_tcprop, thermochemical_properties
from tlr import TransferLinearRegression

start_time = time.time()
# Construct the relative path to the CSV file
relative_path = os.path.join(
    os.path.dirname(__file__), "data", "elx_CM_dataset_5.csv"
)

elx_cm_data = pd.read_csv(
    relative_path)


### Initialize Base Coefficients

elx_cell = ELX_pem_cell(
    T=363.15,
    P_c=1,
    P_a=1,
    i=1,
    i_lim=5,
    t=127,
    sigma_constants=[0.005139, -2.89556, 0.016, 1.625, 0.1875, 0.00326, 1286],
    j_0_OER=0.000000001,
    j_0_HER=0.00075,
    Ea_OER=90000,
    Ea_HER=30000,
    T_ref_OER=303.15,
    T_ref_HER=303.15,
    alpha_c=0.5,
    alpha_a=2,
    r_anode = 25,
    r_cathode= 25)

initial_i_lim = elx_cell.i_lim
initial_j_0_OER = elx_cell.j_0_OER
initial_j_0_HER = elx_cell.j_0_HER
initial_r_anode = elx_cell.r_anode
initial_r_cathode = elx_cell.r_cathode
initial_t = elx_cell.t
initial_sigma_const1 = elx_cell.sigma_constants[0]
initial_sigma_const2 = elx_cell.sigma_constants[1]
initial_sigma_const3 = elx_cell.sigma_constants[2]
initial_sigma_const4 = elx_cell.sigma_constants[3]
initial_sigma_const5 = elx_cell.sigma_constants[4]
initial_sigma_const6 = elx_cell.sigma_constants[5]
initial_sigma_const7 = elx_cell.sigma_constants[6]


base_coefficients = np.array([
    initial_r_anode, initial_r_cathode, 
    initial_t
])

# Initialize penalty matrix
penalty_matrix = np.diag([5, 5, 5, 10, 10, 20, 1, 1, 1, 1, 1, 1, 1])

# Initialize TLR with regularization parameter
lambda_ = 10
tlr = TransferLinearRegression(lambda_=lambda_, penalty_matrix=penalty_matrix)

# Track the previous membrane conductivity at 303K
previous_sigma_303K = MemConductivity(T=elx_cell.T, sigma_constants=[0.005139, -2.89556, 0.016, 1.625, 0.1875, 0.00326, 1286]).sigma_at_reference_temp(303.15)

noise = np.random.normal(0,1, len(elx_cm_data["Current Density [A/cm2]"]))

operating_hour = elx_cm_data["Hour"]
current_density = elx_cm_data["Current Density [A/cm2]"]
temperature = elx_cm_data["Temperature [C]"] + noise
pressure_anode = elx_cm_data["Anode Pressure [barg]"] + (0.1*noise)
pressure_cathode = elx_cm_data["Cathode Pressure [barg]"] + (0.1*noise)

measured_voltage = elx_cm_data["Voltage [V]"]

# Check correlation between features
features_df = pd.DataFrame({
    "Operating Hour": operating_hour,
    "Current Density": current_density,
    "Temperature": temperature,
    "Pressure Anode": pressure_anode,
    "Pressure Cathode": pressure_cathode
})
print(features_df.corr())

# Create a DataFrame to store results
columns = [
    "Interval", "Measured Current Density [A/cm2]", "Measured Voltage [V]",
    "Predicted Voltage [V]", 
    "Initial r_anode", "Initial r_cathode", "Initial t",
    "Updated r_anode", "Updated r_cathode", "Updated t",
]

results_df = pd.DataFrame(columns=columns)


# Main loop through intervals
for interval in range(0, len(elx_cm_data.index)):
    # Get current features (e.g., current density, temperature, etc.)
    features = np.array([current_density, temperature, operating_hour]).T  # Shape (n_samples, 4)

    # Fit the TLR model using the synthetic coefficients and base coefficients with intercept
    tlr.fit(base_coefficients, measured_voltage, base_coefficients, previous_sigma_303K)

    # Apply constraints on the updated synthetic coefficients (if needed)
    tlr.apply_constraints(tlr.coef_ - base_coefficients, base_coefficients, previous_sigma_303K)

    # Update the base coefficients based on the TLR model output
    base_coefficients = tlr.coef_

    # Separate intercept (if necessary) when updating other parameters
    base_coefficients = base_coefficients[:-1] 

    # Update previous conductivity at 303K
    previous_sigma_303K = MemConductivity(T=temperature[interval], sigma_constants=base_coefficients[6:13]).sigma_at_reference_temp(303.15)
    print(previous_sigma_303K)

    # Update the parameters from the fitted model
    updated_r_anode, updated_r_cathode, \
    updated_t,\
    = tlr.coef_[:13]

    # Update the membrane conductivity calculation with the new sigma constants
    sigma_mem = MemConductivity(T=temperature[interval], sigma_constants=[
        initial_sigma_const1, initial_sigma_const2, initial_sigma_const3,
        initial_sigma_const4, initial_sigma_const5, initial_sigma_const6, initial_sigma_const7
    ]).sigma()

    # Recalculate the cell voltage using updated parameters
    V_cell = elx_tcprops(h2o_tcprop(temperature[interval]), 
                         h2_tcprop(temperature[interval]), 
                         o2_tcprop(temperature[interval]), 
                         temperature[interval], 
                         pressure_cathode[interval], 
                         pressure_anode[interval]
                         ).V_rev() 
    + ActivationOverpotential(T=temperature[interval],
                            i=current_density[interval], 
                            j_0=initial_j_0_OER,
                            Ea=elx_cell.Ea_OER, 
                            T_ref=elx_cell.T_ref_OER, 
                            alpha=elx_cell.alpha_a
                            ).V_act() 
    + ActivationOverpotential(T=temperature[interval],
                            i=current_density[interval], 
                            j_0=initial_j_0_HER,
                            Ea=elx_cell.Ea_HER, 
                            T_ref=elx_cell.T_ref_HER, 
                            alpha=elx_cell.alpha_a
                            ).V_act() 
    + OhmicOverpotential(i=current_density[interval]).V_ohm_mem(t=updated_t, sigma=sigma_mem)
    + OhmicOverpotential(i=current_density[interval]).V_ohm_electrode(updated_r_anode)
    + OhmicOverpotential(i=current_density[interval]).V_ohm_electrode(updated_r_cathode)
    + MassTransportOverpotential(i=current_density[interval], i_lim=initial_i_lim, T=temperature[interval]).V_diff()

    # # Save the updated base coefficients for the next interval
    # base_coefficients = tlr.coef_

    # # Update previous conductivity at 303 K
    # previous_sigma_303K = MemConductivity(T=None, sigma_constants=base_coefficients[6:13]).sigma_at_reference_temp(303.15)

    # Append results to the DataFrame
    results_df = results_df.append({
        "Interval": interval,
        "Measured Current Density [A/cm2]": current_density[interval],
        "Measured Voltage [V]": measured_voltage[interval],
        "Predicted Voltage [V]": V_cell,
        "Initial i_lim": initial_i_lim,
        "Initial j_0_OER": initial_j_0_OER,
        "Initial j_0_HER": initial_j_0_HER,
        "Initial r_anode": initial_r_anode,
        "Initial r_cathode": initial_r_cathode,
        "Initial t": initial_t,
        "Updated i_lim": initial_i_lim,
        "Updated j_0_OER": initial_j_0_OER,
        "Updated j_0_HER": initial_j_0_HER,
        "Updated r_anode": updated_r_anode,
        "Updated r_cathode": updated_r_cathode,
        "Updated t": updated_t,
        "Sigma Constant 1": initial_sigma_const1,
        "Sigma Constant 2": initial_sigma_const2,
        "Sigma Constant 3": initial_sigma_const3,
        "Sigma Constant 4": initial_sigma_const4,
        "Sigma Constant 5": initial_sigma_const5,
        "Sigma Constant 6": initial_sigma_const6,
        "Sigma Constant 7": initial_sigma_const7
    }, ignore_index=True)

    print(results_df)

# Save the DataFrame to a CSV file
results_df.to_csv("tlr_results.csv", index=False)
