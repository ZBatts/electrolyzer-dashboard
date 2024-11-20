import numpy as np
from dataclasses import dataclass

### Defining ELX PEM Cell Object

class ELX_pem_cell:
    def __init__(self, T, P_c, P_a, i, i_lim, t, sigma_constants, j_0_OER, j_0_HER, T_ref_OER, T_ref_HER, Ea_OER, Ea_HER, alpha_c, alpha_a, r_anode, r_cathode):
        self.T = T                                # Electrolysis reaction temperature array in Kelvin [K]
        self.P_c =P_c                             # Cathode pressure in [bar]
        self.P_a = P_a                            # Anode pressure in [bar]
        self.i = i                                # Current Density array [A/cm^2]
        self.i_lim = i_lim                        # Maximum Current Density [A/cm2]
        self.t = t                                # Membrane thickness [micro-meters]
        self.sigma_constants = sigma_constants    # Sigma Constants
        self.j_0_OER = j_0_OER                    # OER Reference exchange current density [A/cm^2]
        self.j_0_HER = j_0_HER                    # HER Reference exchange current density [A/cm^2]
        self.T_ref_OER = T_ref_OER                # OER Reference temperature [K]
        self.T_ref_HER = T_ref_HER                # HER Reference Temperature [K]
        self.Ea_OER = Ea_OER                      # OER Activation Energy [J/mol]
        self.Ea_HER = Ea_HER                      # HER Activation Energy [J/mol]
        self.alpha_c = alpha_c                    # Cathode charge transfer coefficient
        self.alpha_a = alpha_a                    # Anode charge transfer coefficient 
        self.r_anode = r_anode                    # Anode Electrode Resistivity [milli-ohms/cm2]
        self.r_cathode = r_cathode                # Cathode Electrode Resistivity [milli-ohms/cm2]

### Thermodynamic Definition and Properties

@dataclass(frozen=True)
class thermochemical_properties:
    T: float

    def enthalpy(self):
        t = self.T / 1000
        return self.A * t + self.B * t**2 / 2 + self.C * t**3 / 3 + self.D * t**4 / 4 - self.E / t + self.F

    def entropy(self):
        t = self.T / 1000
        return self.A * np.log(t) + self.B * t + self.C * t**2 / 2 + self.D * t**3 / 3 - self.E / (2 * t**2) + self.G


@dataclass(frozen=True)
class h2_tcprop(thermochemical_properties):
    A: float = 33.066178
    B: float = -11.363417
    C: float = 11.432816
    D: float = -2.772874
    E: float = -0.158558
    F: float = -9.9807
    G: float = 172.707974
    H: float = 0.0


@dataclass(frozen=True)
class h2o_tcprop(thermochemical_properties):
    A: float = -203.6060
    B: float = 1523.290
    C: float = -3196.413
    D: float = 2474.455
    E: float = 3.855326
    F: float = -256.5478
    G: float = -488.7163
    H: float = -283.8304


@dataclass(frozen=True)
class o2_tcprop(thermochemical_properties):
    A: float = 31.2234
    B: float = -20.23531
    C: float = 57.86644
    D: float = -36.50624
    E: float = -0.007374
    F: float = -8.903471
    G: float = 246.7945
    H: float = 0.0


@dataclass
class elx_tcprops:
    h2_tcprop: h2_tcprop
    h2o_tcprop: h2o_tcprop
    o2_tcprop: o2_tcprop
    T: np.ndarray             # Array of temperatures
    P_c: float
    P_a: float
    F: float = 96485.3329
    z: float = 2.0
    R: float = 8.1344626185324

    def P_h2o(self, T):
        # Calculate vapor pressure for each temperature value
        P_h2o_vals = np.where(T < 373.15,
                              (10**(8.07131 - (1730.63 / (233.426 + (T - 273.15))))) / 100000,
                              (10**(8.14019 - (1810.94 / (244.485 + (T - 273.15))))) / 10000)
        return P_h2o_vals

    def dH_rxn_elx(self):
        # Enthalpy of reaction across all temperatures
        return np.array([self.h2_tcprop.enthalpy() + 0.5 * self.o2_tcprop.enthalpy() - self.h2o_tcprop.enthalpy()])

    def dS_rxn_elx(self):
        # Entropy of reaction across all temperatures
        return np.array([self.h2_tcprop.entropy() + 0.5 * self.o2_tcprop.entropy() - self.h2o_tcprop.entropy()])

    def dG_rxn_elx(self, T):
        # Gibbs free energy across all temperatures
        return self.dH_rxn_elx() - (self.dS_rxn_elx() / 1000.0) * T

    def V_rev(self):
        # Reversible voltage across all temperatures
        return (self.dG_rxn_elx(self.T) * 1000.0) / (self.z * self.F) + \
               ((self.R * self.T) / (self.z * self.F)) * np.log(((self.P_a + 1.0) - self.P_h2o(self.T))**0.5 *
                                                                ((self.P_c + 1.0) - self.P_h2o(self.T)) / (self.P_a + 1.0))


class ActivationOverpotential:
    def __init__(self, T, i, j_0, Ea, T_ref, alpha):
        self.T = T                 # Cell Temperature [K]
        self.i = i                 # Current Density [A/cm2]
        self.j_0 = j_0             # Reference Exchange Current Density [A/cm2]
        self.Ea = Ea               # Activation Energy 
        self.T_ref = T_ref         # Reference Temperature 
        self.alpha = alpha         #
        self.F = 96485.3329        # Faraday's Constant
        self.R = 8.1344626185324   # Ideal Gas Constant
        self.epsilon = 1e-6        # Small value to avoid inf / singular matrix

    def j(self):
        return self.j_0 * np.exp((self.Ea / self.R) * ((1 / self.T_ref) - (1 / self.T)))

    def V_act(self):
        return (self.R * self.T) / (self.alpha * self.F) * np.log((self.i + self.epsilon)/ (self.j() + self.epsilon))


class OhmicOverpotential:
    def __init__(self, i):
        self.i = i                                             # Current Density [A/cm2]
        self.epsilon = 1e-6                                    # Small value to avoid divide by zero

    def V_ohm_mem(self, t, sigma): 
        return self.i * ((t * 0.0001) / (sigma + 1e-6))        # Sigma - conductivity , t - Thickness [micro-meters]
    
    def V_ohm_electrode(self, R):
        return self.i * (0.001 * R)                            # R - resitivity [milli-Ohms / cm2]



class MemConductivity:
    def __init__(self, T, sigma_constants):
        self.T = T                                    # Temperature array [K]
        self.sigma_constants = sigma_constants        # Array of 7 constants for the sigma equation

    def sigma(self):
        # Extract constants
        A, B, C, D, E, F, G = self.sigma_constants
        
        # Use the extracted constants in the sigma formula
        return ((A * ((B + (C * self.T)) + D) / E) - F) * np.exp(G * ((1 / 303.15) - (1 / self.T)))

    def sigma_at_reference_temp(self, ref_T=313.15):
        # Compute the conductivity at the reference temperature (303.15 K by default)
        A, B, C, D, E, F, G = self.sigma_constants
        return ((A * ((B + (C * ref_T)) + D) / E) - F) * np.exp(G * ((1 / 303.15) - (1 / ref_T)))

# class MemConductivity:
#     def __init__(self, T):
#         self.T = T                                 # Temperture [K]

#     def sigma(self):
#         return ((0.005139 * ((-2.89556 + (0.016 * self.T)) + 1.625) / 0.1875) - 0.00326) * np.exp(1286 * ((1 / 303.15) - (1 / self.T)))
    
class MassTransportOverpotential:
    def __init__(self, i, i_lim, T):
        self.i = i                                  # Current Density [A/cm2]
        self.i_lim = i_lim                          # Maximum Current Density of Electrolyzer [A/cm2]
        self.T = T                                  # Temperature [K]
        self.F = 96485.3329                         # Faraday's Constant
        self.R = 8.1344626185324                    # Ideal Gas Constant
    
    def V_diff(self):
        # Apply condition element-wise using np.where()
        V_diff_values = np.where(self.i < 1.6, 0, -1 * ((self.R * self.T) / (4 * self.F)) * np.log(1 - (self.i/self.i_lim)))
        return V_diff_values

# Define PEM cell conditions and material properties (arrays for T and i)
elx_pem_cell = ELX_pem_cell(T=np.array([363.15,363.15, 363.15, 363.15,363.15,313.15,318.15,323.15,328.15,333.15,338.15,343.15,348.15,353.15]),
                            P_c=np.array([1,1,1,1,1,40,40,40,40,40,40,40,40,40]),
                            P_a=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1]),
                            #i=np.array([0.01, 0.05, 0.075, .1, .2, .3, .4, .5, 1, 1.5, 2, 2.5, 3, 3]),
                            i=np.array([1, 1, 1.5, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
                            i_lim = 6,
                            t=140.0,
                            sigma_constants= [0.005139, -2.89556, 0.016, 1.625, 0.1875, 0.00326, 1286],
                            j_0_OER=9.4e-10,
                            j_0_HER=0.00075,
                            Ea_OER=90000,
                            Ea_HER=30000,
                            T_ref_OER=303.15,
                            T_ref_HER=303.15,
                            alpha_c=0.5,
                            alpha_a=2,
                            r_anode = 100,
                            r_cathode= 100)

# Thermodynamics of Electrolysis 
h2obj = h2o_tcprop(elx_pem_cell.T)
h2oobj = h2_tcprop(elx_pem_cell.T)
o2obj = o2_tcprop(elx_pem_cell.T)

elx_rxn_tcprops = elx_tcprops(h2oobj, h2obj, o2obj, elx_pem_cell.T, elx_pem_cell.P_c, elx_pem_cell.P_a)

# Open Circuit Voltage
OCV = elx_rxn_tcprops.V_rev()

# Activation Overpotentials
V_OER = ActivationOverpotential(T=elx_pem_cell.T, i=elx_pem_cell.i, j_0=elx_pem_cell.j_0_OER,
                                Ea=elx_pem_cell.Ea_OER, T_ref=elx_pem_cell.T_ref_OER, alpha=elx_pem_cell.alpha_a)

V_HER = ActivationOverpotential(T=elx_pem_cell.T, i=elx_pem_cell.i, j_0=elx_pem_cell.j_0_HER,
                                Ea=elx_pem_cell.Ea_HER, T_ref=elx_pem_cell.T_ref_HER, alpha=elx_pem_cell.alpha_c)

# Ohmic Overpotential Membrane
sigma_mem = MemConductivity(T=elx_pem_cell.T, sigma_constants=elx_pem_cell.sigma_constants).sigma()
V_ohm_membrane = OhmicOverpotential(i=elx_pem_cell.i)

# Ohmic Overpotential Anode 
V_ohm_anode = OhmicOverpotential(i=elx_pem_cell.i).V_ohm_electrode(R=elx_pem_cell.r_anode)

# Ohmic Overpotential Cathode
V_ohm_cathode = OhmicOverpotential(i=elx_pem_cell.i).V_ohm_electrode(R=elx_pem_cell.r_cathode)

# Mass Transport Overpotential
V_mt = MassTransportOverpotential(i=elx_pem_cell.i, i_lim=elx_pem_cell.i_lim, T=elx_pem_cell.T).V_diff()

# Final cell voltage calculation for all conditions
V_cell = OCV + V_OER.V_act() + V_HER.V_act() + V_ohm_membrane.V_ohm_mem(t=elx_pem_cell.t, sigma=sigma_mem) + V_ohm_anode + V_ohm_cathode + V_mt

# print("Open Circuit Voltage (V):", OCV)
# print("Activation Overpotential OER (V):", V_OER.V_act())
# print("Activation Overpotential HER (V):", V_HER.V_act())
# print("Membrane Ohmic Overpotential (V):", V_ohm_membrane.V_ohm_mem(t=elx_pem_cell.t, sigma=sigma_mem))
# print("Anode Ohmic Overpotential (V):", V_ohm_anode)
# print("Cathode Ohmic Overpotential (V):", V_ohm_cathode)
# print("Mass Transport Overpotential (V):", V_mt)
# print("Cell Voltage (V):", V_cell)
# print("Cell Exchange Current Density:", V_OER.j())
