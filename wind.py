import numpy as np
from scipy.ndimage import gaussian_filter
import traceback

class Wind:
    def __init__(self, sim):
        """Initialize wind module with reference to main simulation"""
        self.sim = sim
        
        # Optimization: Arrays for wind calculations
        self._u_geo = None
        self._v_geo = None
        self._u_new = None
        self._v_new = None

    def initialize(self):
        """Initialize wind fields with small random perturbations."""
        # Initialize u and v components
        self.sim.u = np.zeros((self.sim.map_height, self.sim.map_width))
        self.sim.v = np.zeros((self.sim.map_height, self.sim.map_width))
        
        # Now initialize circulation patterns
        self.initialize_global_circulation()

    def initialize_global_circulation(self):
        """Initialize wind patterns with realistic global circulation and wind belts"""
        # Negative u_component: Wind is blowing TOWARDS the east
        # Positive u_component: Wind is blowing TOWARDS the west
        # Negative v_component: Wind is blowing TOWARDS the south
        # Positive v_component: Wind is blowing TOWARDS the north
        try:
            # Create latitude array (90 at top to -90 at bottom)
            latitudes = np.linspace(90, -90, self.sim.map_height)
            
            # Initialize wind components
            self.sim.u = np.zeros((self.sim.map_height, self.sim.map_width))
            self.sim.v = np.zeros((self.sim.map_height, self.sim.map_width))
            
            # For each row (latitude band)
            for y in range(self.sim.map_height):
                lat = latitudes[y]
                lat_rad = np.deg2rad(lat)
                
                # Base speed scaling with latitude
                speed_scale = np.cos(lat_rad)
                
                # Northern Hemisphere
                if lat > 60:  # Polar cell (60°N to 90°N)
                    angle_factor = (lat - 60) / 30  # 1 at 90°N, 0 at 60°N
                    u_component = -30.0 * speed_scale * angle_factor        # Increased from 15.0
                    v_component = -24.0 * speed_scale * (1 - angle_factor) # Increased from -12.0
                    
                elif lat > 30:  # Ferrel cell (30°N to 60°N)
                    angle_factor = (lat - 30.01) / 30.01  # 1 at 60°N, 0 at 30°N
                    u_component = 50.0 * speed_scale * (1 - angle_factor) # Increased from -25.0
                    v_component = 24.0 * speed_scale * angle_factor        # Increased from 12.0
                    
                elif lat > 0:  # Hadley cell (0° to 30°N)
                    angle_factor = lat / 30  # 1 at 30°N, 0 at equator
                    u_component = -30.0 * speed_scale * (1 - angle_factor)  # Increased from 15.0
                    v_component = -24.0 * speed_scale * angle_factor       # Increased from -12.0
                    
                # Southern Hemisphere (mirror of northern patterns)
                elif lat > -30:  # Hadley cell (0° to 30°S)
                    angle_factor = -lat / 30  # 1 at 30°S, 0 at equator
                    u_component = -30.0 * speed_scale * (1 - angle_factor)  # Increased from 15.0
                    v_component = 24.0 * speed_scale * angle_factor        # Increased from 12.0
                    
                elif lat > -60:  # Ferrel cell (30°S to 60°S)
                    angle_factor = (-lat - 30) / 30  # 1 at 60°S, 0 at 30°S
                    u_component = 50.0 * speed_scale * (1 - angle_factor) # Increased from -25.0
                    v_component = -24.0 * speed_scale * angle_factor       # Increased from -12.0
                    
                else:  # Polar cell (60°S to 90°S)
                    angle_factor = (-lat - 60) / 30  # 1 at 90°S, 0 at 60°S
                    u_component = -30.0 * speed_scale * angle_factor        # Increased from 15.0
                    v_component = 24.0 * speed_scale * (1 - angle_factor)  # Increased from 12.0
                
                # Apply components to the wind field
                self.sim.u[y, :] = u_component
                self.sim.v[y, :] = v_component

        except Exception as e:
            print(f"Error in global circulation initialization: {e}")
            traceback.print_exc()

    def update(self):
        """Update wind speed and direction"""
        try:
            # Initialize arrays if they don't exist
            if not hasattr(self, '_u_geo') or self._u_geo is None:
                self._u_geo = np.zeros_like(self.sim.u, dtype=np.float32)
                self._v_geo = np.zeros_like(self.sim.v, dtype=np.float32)
            
            if not hasattr(self, '_u_new') or self._u_new is None:
                self._u_new = np.zeros_like(self.sim.u, dtype=np.float32)
                self._v_new = np.zeros_like(self.sim.v, dtype=np.float32)
                
            # --- PRESSURE GRADIENT FORCE ---
            # Use numpy's gradient function directly with mode='wrap' for proper handling
            dp_dy, dp_dx = np.gradient(self.sim.pressure, 
                                      self.sim.grid_spacing_y, self.sim.grid_spacing_x,
                                      edge_order=2)  # Higher order accuracy
            
            # --- CORIOLIS PARAMETER ---
            # Calculate Coriolis parameter (vectorized)
            latitudes_rad = np.deg2rad(self.sim.latitude)
            f = 2 * self.sim.Omega * np.sin(latitudes_rad)
            
            # --- GEOSTROPHIC WIND ---
            rho = 1.2  # air density (kg/m³)
            f_mask = np.abs(f) > 1e-10
            
            # Pre-allocate output arrays (single precision for performance)
            if hasattr(self, '_u_geo') and self._u_geo.shape == self.sim.u.shape:
                u_geo = self._u_geo
                v_geo = self._v_geo
                u_geo.fill(0)
                v_geo.fill(0)
            else:
                u_geo = np.zeros_like(self.sim.u, dtype=np.float32)
                v_geo = np.zeros_like(self.sim.v, dtype=np.float32)
                self._u_geo = u_geo
                self._v_geo = v_geo
            
            # Calculate only where f is not near zero (vectorized operation)
            rho_f = rho * f[f_mask]
            u_geo[f_mask] = -1.0 / rho_f * dp_dy[f_mask]
            v_geo[f_mask] = 1.0 / rho_f * dp_dx[f_mask]
            
            # --- THERMAL WIND ---
            # Calculate temperature gradients
            dT_dy, dT_dx = np.gradient(self.sim.temperature_celsius, 
                                      self.sim.grid_spacing_y, self.sim.grid_spacing_x)
            
            # Add thermal wind component (vectorized)
            thermal_factor = 0.003
            u_thermal = thermal_factor * dT_dy
            v_thermal = -thermal_factor * dT_dx
            
            # --- CLOUD-INDUCED WIND EFFECTS ---
            if hasattr(self.sim, 'cloud_cover'):
                # Calculate cloud gradients
                dC_dy, dC_dx = np.gradient(self.sim.cloud_cover, 
                                         self.sim.grid_spacing_y, self.sim.grid_spacing_x)
                
                # Cloud-induced flows (vectorized)
                cloud_factor = 0.05
                u_thermal -= cloud_factor * dC_dx  # Flow toward higher cloud concentration
                v_thermal -= cloud_factor * dC_dy
            
            # --- NEAR-EQUATOR HANDLING ---
            # Single calculation for equator band
            equator_band = np.abs(self.sim.latitude) < 5.0
            if np.any(equator_band):
                gradient_factor = 5e-7
                # Use where to modify only equator band values
                u_geo = np.where(equator_band, -gradient_factor * dp_dx, u_geo)
                v_geo = np.where(equator_band, -gradient_factor * dp_dy, v_geo)
            
            # --- SURFACE FRICTION EFFECTS ---
            is_land = self.sim.elevation > 0
            friction_factor = np.where(is_land, 0.4, 0.2)  # Higher friction over land
            
            # Compute final wind (vectorized with in-place operations)
            # Reduced inertia factor for more dynamic response
            inertia = 0.75
            
            # Total wind components (reuse arrays where possible)
            if not hasattr(self, '_u_new') or self._u_new.shape != self.sim.u.shape:
                self._u_new = np.zeros_like(self.sim.u, dtype=np.float32)
                self._v_new = np.zeros_like(self.sim.v, dtype=np.float32)
            
            # Calculate new winds with vectorized operations
            np.multiply(inertia, self.sim.u, out=self._u_new)
            np.multiply(inertia, self.sim.v, out=self._v_new)
            
            np.add(self._u_new, (1 - inertia) * u_geo, out=self._u_new)
            np.add(self._v_new, (1 - inertia) * v_geo, out=self._v_new)
            
            np.add(self._u_new, u_thermal, out=self._u_new)
            np.add(self._v_new, v_thermal, out=self._v_new)
            
            # Apply friction (vectorized)
            np.multiply(self._u_new, 1 - friction_factor, out=self._u_new)
            np.multiply(self._v_new, 1 - friction_factor, out=self._v_new)
            
            # Apply wind speed limits (vectorized)
            wind_speed = np.sqrt(self._u_new**2 + self._v_new**2)
            max_wind = 50.0  # m/s
            scaling = np.where(wind_speed > max_wind, max_wind / wind_speed, 1.0)
            
            # Scale wind components
            np.multiply(self._u_new, scaling, out=self._u_new)
            np.multiply(self._v_new, scaling, out=self._v_new)
            
            # Update wind fields
            self.sim.u, self._u_new = self._u_new, self.sim.u  # Swap references to avoid allocation
            self.sim.v, self._v_new = self._v_new, self.sim.v
            
        except Exception as e:
            print(f"Error updating wind: {e}")
            traceback.print_exc()

    def calculate_direction(self, u, v):
        """Calculate wind direction in degrees (meteorological convention)"""
        return (270 - np.rad2deg(np.arctan2(v, u))) % 360 