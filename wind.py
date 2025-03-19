import numpy as np
from scipy.ndimage import gaussian_filter, zoom
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
        
        # Adaptive resolution settings
        self._adaptive_mode = False
        self._downsampling_factor = 2  # Default reduction factor
        self._downsampled_fields = None
        
        # Tiered Wind Calculation settings (Solution 1)
        self._update_counter = 0
        self._geostrophic_update_freq = 1    # Update every step (fast component)
        self._thermal_update_freq = 2        # Update every 2 steps (medium component)
        self._cloud_update_freq = 3          # Update every 3 steps (slow component)
        self._friction_update_freq = 5       # Update every 5 steps (very slow component)
        
        # Cached values for components updated less frequently
        self._thermal_wind_cache = None      # Cache for thermal wind
        self._cloud_wind_cache = None        # Cache for cloud-induced wind
        self._friction_factors = None        # Cache for friction calculation
        self._temp_gradients = None          # Cache for temperature gradients
        self._cloud_gradients = None         # Cache for cloud gradients

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
        """Update wind speed and direction using tiered calculation approach"""
        try:
            # Check if high-speed mode is active
            high_speed = getattr(self.sim, 'high_speed_mode', False)
            
            # Update adaptive mode status
            if high_speed and high_speed != self._adaptive_mode:
                self._adaptive_mode = True
                # Clear cached downsampled fields when mode changes
                self._downsampled_fields = None
                
            elif not high_speed and self._adaptive_mode:
                self._adaptive_mode = False
                self._downsampled_fields = None
                
            # Use adaptive resolution in high-speed mode
            if self._adaptive_mode:
                self._update_adaptive()
                return
                
            # Regular full-resolution update continues below
            
            # Increment update counter
            self._update_counter += 1
            
            # Initialize arrays if they don't exist
            if not hasattr(self, '_u_geo') or self._u_geo is None:
                self._u_geo = np.zeros_like(self.sim.u, dtype=np.float32)
                self._v_geo = np.zeros_like(self.sim.v, dtype=np.float32)
            
            if not hasattr(self, '_u_new') or self._u_new is None:
                self._u_new = np.zeros_like(self.sim.u, dtype=np.float32)
                self._v_new = np.zeros_like(self.sim.v, dtype=np.float32)
            
            # --- TIER 1: PRESSURE GRADIENT AND GEOSTROPHIC WIND (UPDATED EVERY STEP) ---
            # This is the most important component for determining large-scale wind patterns
            
            # Calculate pressure gradient
            dp_dy, dp_dx = np.gradient(self.sim.pressure, 
                                      self.sim.grid_spacing_y, self.sim.grid_spacing_x,
                                      edge_order=2)  # Higher order accuracy
            
            # Calculate Coriolis parameter (vectorized)
            latitudes_rad = np.deg2rad(self.sim.latitude)
            f = 2 * self.sim.Omega * np.sin(latitudes_rad)
            
            # Calculate geostrophic wind
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
            
            # Near-equator handling (always calculate this as part of Tier 1)
            equator_band = np.abs(self.sim.latitude) < 5.0
            if np.any(equator_band):
                gradient_factor = 5e-7
                # Use where to modify only equator band values
                u_geo = np.where(equator_band, -gradient_factor * dp_dx, u_geo)
                v_geo = np.where(equator_band, -gradient_factor * dp_dy, v_geo)
            
            # --- TIER 2: THERMAL WIND (UPDATED LESS FREQUENTLY) ---
            # Temperature-driven wind components change more slowly
            
            should_update_thermal = (self._update_counter % self._thermal_update_freq == 0) or (self._thermal_wind_cache is None)
            
            if should_update_thermal:
                # Calculate temperature gradients
                dT_dy, dT_dx = np.gradient(self.sim.temperature_celsius, 
                                          self.sim.grid_spacing_y, self.sim.grid_spacing_x)
                
                # Store for potential reuse
                self._temp_gradients = (dT_dy, dT_dx)
                
                # Calculate thermal wind component (vectorized)
                thermal_factor = 0.003
                u_thermal = thermal_factor * dT_dy
                v_thermal = -thermal_factor * dT_dx
                
                # Cache the thermal wind components
                self._thermal_wind_cache = (u_thermal, v_thermal)
            else:
                # Use cached thermal wind components
                u_thermal, v_thermal = self._thermal_wind_cache
            
            # --- TIER 3: CLOUD-INDUCED EFFECTS (UPDATED EVEN LESS FREQUENTLY) ---
            # Cloud patterns change more slowly than pressure or temperature
            
            u_cloud = 0
            v_cloud = 0
            
            should_update_cloud = (self._update_counter % self._cloud_update_freq == 0) or (self._cloud_wind_cache is None)
            
            if hasattr(self.sim, 'cloud_cover') and should_update_cloud:
                # Calculate cloud gradients
                dC_dy, dC_dx = np.gradient(self.sim.cloud_cover, 
                                         self.sim.grid_spacing_y, self.sim.grid_spacing_x)
                
                # Store for potential reuse
                self._cloud_gradients = (dC_dy, dC_dx)
                
                # Cloud-induced flows (vectorized)
                cloud_factor = 0.05
                u_cloud = -cloud_factor * dC_dx  # Flow toward higher cloud concentration
                v_cloud = -cloud_factor * dC_dy
                
                # Cache the cloud-induced wind components
                self._cloud_wind_cache = (u_cloud, v_cloud)
            elif hasattr(self.sim, 'cloud_cover') and self._cloud_wind_cache is not None:
                # Use cached cloud-induced wind components
                u_cloud, v_cloud = self._cloud_wind_cache
            
            # --- TIER 4: SURFACE FRICTION (UPDATED VERY INFREQUENTLY) ---
            # Surface properties change extremely slowly
            
            should_update_friction = (self._update_counter % self._friction_update_freq == 0) or (self._friction_factors is None)
            
            if should_update_friction:
                # Calculate friction factors based on surface type
                is_land = self.sim.elevation > 0
                friction_factor = np.where(is_land, 0.4, 0.2)  # Higher friction over land
                
                # Cache the friction factors
                self._friction_factors = friction_factor
            else:
                # Use cached friction factors
                friction_factor = self._friction_factors
            
            # --- COMBINING COMPONENTS AND APPLYING INERTIA ---
            
            # Calculate new winds with vectorized operations
            # Use a modified inertia factor based on high-speed mode
            inertia = 0.75
            
            # Initialize new wind arrays (reuse arrays where possible)
            if not hasattr(self, '_u_new') or self._u_new.shape != self.sim.u.shape:
                self._u_new = np.zeros_like(self.sim.u, dtype=np.float32)
                self._v_new = np.zeros_like(self.sim.v, dtype=np.float32)
            
            # Apply inertia from previous wind
            np.multiply(inertia, self.sim.u, out=self._u_new)
            np.multiply(inertia, self.sim.v, out=self._v_new)
            
            # Add geostrophic component (always calculated)
            np.add(self._u_new, (1 - inertia) * u_geo, out=self._u_new)
            np.add(self._v_new, (1 - inertia) * v_geo, out=self._v_new)
            
            # Add thermal component
            np.add(self._u_new, u_thermal, out=self._u_new)
            np.add(self._v_new, v_thermal, out=self._v_new)
            
            # Add cloud component if available
            if hasattr(self.sim, 'cloud_cover'):
                np.add(self._u_new, u_cloud, out=self._u_new)
                np.add(self._v_new, v_cloud, out=self._v_new)
            
            # Apply friction (vectorized)
            np.multiply(self._u_new, 1 - friction_factor, out=self._u_new)
            np.multiply(self._v_new, 1 - friction_factor, out=self._v_new)
            
            # --- LIMITING WIND SPEEDS ---
            
            # Calculate wind speed for limiting
            wind_speed = np.sqrt(self._u_new**2 + self._v_new**2)
            max_wind = 50.0  # m/s
            scaling = np.where(wind_speed > max_wind, max_wind / wind_speed, 1.0)
            
            # Apply speed limits
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

    def _update_adaptive(self):
        """Update wind using lower resolution for performance"""
        try:
            factor = self._downsampling_factor
            
            # Downsample static fields only occasionally
            if self._downsampled_fields is None or self.sim.time_step % 20 == 0:
                # Cache key fields at lower resolution
                h, w = self.sim.map_height, self.sim.map_width
                h_low, w_low = h // factor, w // factor
                
                # Downsample latitudes for Coriolis calculation
                lat_low = zoom(self.sim.latitude, 1/factor, order=1)
                lat_rad_low = np.deg2rad(lat_low)
                f_low = 2 * self.sim.Omega * np.sin(lat_rad_low)
                
                # Downsample elevation for friction calculation
                elevation_low = zoom(self.sim.elevation, 1/factor, order=1)
                is_land_low = elevation_low > 0
                
                # Store downsampled fields
                self._downsampled_fields = {
                    'latitudes': lat_low,
                    'latitudes_rad': lat_rad_low,
                    'f': f_low,
                    'elevation': elevation_low,
                    'is_land': is_land_low,
                    'shape': (h_low, w_low)
                }
            
            # Get cached fields
            ds = self._downsampled_fields
            
            # Always downsample dynamic fields
            # Downsample pressure and temperature for gradient calculations
            p_low = zoom(self.sim.pressure, 1/factor, order=1)
            t_low = zoom(self.sim.temperature_celsius, 1/factor, order=1)
            
            # Downsample wind fields
            u_low = zoom(self.sim.u, 1/factor, order=1)
            v_low = zoom(self.sim.v, 1/factor, order=1)
            
            # Get cloud cover if available
            cloud_low = None
            if hasattr(self.sim, 'cloud_cover') and self.sim.cloud_cover is not None:
                cloud_low = zoom(self.sim.cloud_cover, 1/factor, order=1)
            
            # --- SIMPLIFIED WIND CALCULATIONS ---
            
            # 1. Pressure gradient force
            dp_dy, dp_dx = np.gradient(p_low)
            
            # 2. Calculate geostrophic wind
            rho = 1.2  # air density
            f_mask = np.abs(ds['f']) > 1e-10
            
            # Pre-allocate output arrays
            u_geo = np.zeros_like(u_low)
            v_geo = np.zeros_like(v_low)
            
            # Calculate only where f is not near zero
            if np.any(f_mask):
                rho_f = rho * ds['f'][f_mask]
                u_geo[f_mask] = -1.0 / rho_f * dp_dy[f_mask]
                v_geo[f_mask] = 1.0 / rho_f * dp_dx[f_mask]
            
            # 3. Handle equator region
            equator_band = np.abs(ds['latitudes']) < 5.0
            if np.any(equator_band):
                gradient_factor = 5e-7
                u_geo[equator_band] = -gradient_factor * dp_dx[equator_band]
                v_geo[equator_band] = -gradient_factor * dp_dy[equator_band]
            
            # 4. Thermal wind component
            dT_dy, dT_dx = np.gradient(t_low)
            thermal_factor = 0.003
            u_thermal = thermal_factor * dT_dy
            v_thermal = -thermal_factor * dT_dx
            
            # 5. Cloud-induced wind effects (if available)
            u_cloud = np.zeros_like(u_low)
            v_cloud = np.zeros_like(v_low)
            if cloud_low is not None:
                dC_dy, dC_dx = np.gradient(cloud_low)
                cloud_factor = 0.05
                u_cloud = -cloud_factor * dC_dx
                v_cloud = -cloud_factor * dC_dy
            
            # 6. Friction effect
            friction_factor = np.where(ds['is_land'], 0.4, 0.2)
            
            # 7. Calculate new wind components
            # Use less inertia to compensate for the lower update frequency
            inertia = 0.7  # Reduced from 0.75 for more responsiveness
            
            # Calculate new wind
            u_new = (inertia * u_low +
                    (1 - inertia) * u_geo +
                    u_thermal +
                    u_cloud) * (1 - friction_factor)
            
            v_new = (inertia * v_low +
                    (1 - inertia) * v_geo +
                    v_thermal +
                    v_cloud) * (1 - friction_factor)
            
            # 8. Apply wind speed limits
            wind_speed = np.sqrt(u_new**2 + v_new**2)
            max_wind = 50.0
            scaling = np.where(wind_speed > max_wind, max_wind / wind_speed, 1.0)
            u_new *= scaling
            v_new *= scaling
            
            # 9. Upsample to full resolution
            u_full = zoom(u_new, factor, order=1)
            v_full = zoom(v_new, factor, order=1)
            
            # Ensure proper dimensions
            if u_full.shape != self.sim.u.shape:
                u_full = np.resize(u_full, self.sim.u.shape)
            if v_full.shape != self.sim.v.shape:
                v_full = np.resize(v_full, self.sim.v.shape)
            
            # Update simulation wind fields
            self.sim.u[:] = u_full
            self.sim.v[:] = v_full
            
        except Exception as e:
            print(f"Error in adaptive wind update: {e}")
            traceback.print_exc() 