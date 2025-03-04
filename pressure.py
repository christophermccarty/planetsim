import numpy as np
from scipy.ndimage import gaussian_filter
import time
import traceback

class Pressure:
    def __init__(self, sim):
        """Initialize pressure module with reference to main simulation"""
        self.sim = sim
        self._oscillation_phase = 0
        self._time_varying_factor = 0
        self._temp_avg_counter = 0

    def initialize(self):
        """Initialize pressure field with standard pressure and elevation effects"""
        try:
            # Standard sea-level pressure in hPa converted to Pa
            P0 = 101325.0  # 1013.25 hPa in Pa
            
            print("Initializing pressure...")
            
            # Initialize pressure array with standard pressure
            self.sim.pressure = np.full((self.sim.map_height, self.sim.map_width), P0, dtype=np.float64)
            
            # Apply elevation effects with limits
            max_height = 2000.0  # Limit effective height for pressure conversion
            limited_elevation = np.clip(self.sim.elevation, 0, max_height)
            elevation_factor = np.exp(-limited_elevation / 7400.0)
            elevation_factor = np.clip(elevation_factor, 0.85, 1.0)  # Limit minimum pressure reduction
            
            self.sim.pressure *= elevation_factor
            
            # Precompute latitude pressure patterns and persistent systems
            # This avoids expensive calculations during each update_pressure call
            print("Precomputing global pressure patterns...")
            lat_rad = np.radians(self.sim.latitude)
            
            # Create basic pressure pattern based on latitude
            lat_pressure = np.zeros_like(self.sim.latitude)
            
            # Subtropical high pressure regions (~30° N/S)
            subtropical_high = 800.0 * np.exp(-((np.abs(lat_rad) - np.radians(30)) / np.radians(15))**2)
            
            # Subpolar low pressure regions (~60° N/S)
            subpolar_low = -600.0 * np.exp(-((np.abs(lat_rad) - np.radians(60)) / np.radians(15))**2)
            
            # Polar high pressure
            polar_high = 600.0 * np.exp(-((np.abs(lat_rad) - np.radians(90)) / np.radians(20))**2)
            
            # Equatorial low pressure
            equatorial_low = -400.0 * np.exp(-(lat_rad / np.radians(10))**2)
            
            # Combine all latitude effects and cache
            self._lat_pressure = subtropical_high + subpolar_low + polar_high + equatorial_low
            
            # Create persistent semi-permanent pressure systems
            print("Creating persistent pressure systems...")
            self._persistent_systems = np.zeros_like(self.sim.latitude)
            
            # Example persistent systems (modify coordinates and strengths as needed)
            # North Pacific (Aleutian) Low
            lon_center, lat_center = 180, 55  # Coordinates in degrees
            strength = -400.0  # Negative for low pressure
            self._add_pressure_system(self._persistent_systems, lon_center, lat_center, 20, strength)
            
            # Siberian High
            lon_center, lat_center = 100, 55  # Coordinates in degrees
            strength = 500.0  # Positive for high pressure
            self._add_pressure_system(self._persistent_systems, lon_center, lat_center, 25, strength)
            
            # South Pacific High
            lon_center, lat_center = -120, -30  # Coordinates in degrees
            strength = 350.0  # Positive for high pressure
            self._add_pressure_system(self._persistent_systems, lon_center, lat_center, 20, strength)
            
            # Azores/Bermuda High
            lon_center, lat_center = -40, 35  # Coordinates in degrees
            strength = 450.0  # Positive for high pressure
            self._add_pressure_system(self._persistent_systems, lon_center, lat_center, 18, strength)
            
            # Initialize land-ocean mask for pressure calculations
            print("Creating land-ocean transition zones...")
            land_ocean_mask = np.zeros_like(self.sim.elevation)
            land_ocean_mask[self.sim.elevation > 0] = 1.0  # Land
            # Smooth the mask to create coastal transition zones
            gaussian_filter(land_ocean_mask, sigma=2.0, mode='wrap', output=land_ocean_mask)
            self._land_ocean_factor = land_ocean_mask
            
            # Store normalized version for visualization
            self.sim.pressure_normalized = self.sim.normalize_data(self.sim.pressure)
            print("Pressure initialization complete.")
            
        except Exception as e:
            print(f"Error in pressure initialization: {e}")
            traceback.print_exc()

    def update(self):
        """Update pressure based on temperature gradients, wind patterns and global circulation"""
        try:
            dt = self.sim.time_step_seconds
            P0 = 101325.0  # Standard sea-level pressure
            
            # Basic masks and elevation
            if not hasattr(self, '_elevation_factor'):
                # Pre-compute elevation-based factors once and cache
                is_land = self.sim.elevation > 0
                max_height = 1000.0
                limited_elevation = np.clip(self.sim.elevation, 0, max_height)
                self._elevation_factor = np.exp(-limited_elevation / 7400.0)
                self._is_land = is_land
                
                # Initialize time-varying factors
                self._time_varying_factor = 0.0
                
            # Update time-varying factor - adds periodic forcing to the system
            self._time_varying_factor += 0.1 * dt
            seasonal_factor = np.sin(self._time_varying_factor / 86400 * 2 * np.pi)  # Daily cycle
            
            # Calculate minimum pressure based on elevation (cached)
            min_pressure = P0 * self._elevation_factor
            
            # --- TEMPERATURE-DRIVEN PRESSURE GRADIENTS ---
            # Temperature has a strong effect on pressure (warm = low pressure, cold = high pressure)
            T = self.sim.temperature_celsius
            
            # Calculate pressure changes due to wind convergence/divergence with wrapping
            # Create wrapped arrays for proper edge handling
            u_wrapped = np.hstack((self.sim.u[:, -1:], self.sim.u, self.sim.u[:, :1]))
            v_wrapped = np.vstack((self.sim.v[-1:, :], self.sim.v, self.sim.v[:1, :]))
            
            # Calculate pressure changes due to wind convergence/divergence with wrapping
            du_dx = np.gradient(u_wrapped, axis=1) / self.sim.grid_spacing_x
            dv_dy = np.gradient(v_wrapped, axis=0) / self.sim.grid_spacing_y
            
            # Remove the padding from gradient results
            du_dx = du_dx[:, 1:-1]
            dv_dy = dv_dy[1:-1, :]
            
            wind_divergence = du_dx + dv_dy
            
            # --- ENHANCED WIND-PRESSURE COUPLING ---
            
            # 1. Stronger wind-driven pressure changes to represent real weather dynamics
            # Increased convergence factor to allow winds to have stronger effect on pressure
            convergence_factor = 0.35  # Increased from 0.25
            pressure_change_wind = -self.sim.pressure * wind_divergence * dt * convergence_factor
            
            # 2. Vorticity-based pressure changes - key for cyclone formation
            # Calculate relative vorticity with proper wrapping (curl of wind field)
            # Create properly padded arrays with consistent dimensions
            u_padded = np.pad(self.sim.u, ((1, 1), (1, 1)), mode='wrap')
            v_padded = np.pad(self.sim.v, ((1, 1), (1, 1)), mode='wrap')
            
            # Calculate gradients in x and y directions
            du_dy = (u_padded[2:, 1:-1] - u_padded[:-2, 1:-1]) / (2 * self.sim.grid_spacing_y)
            dv_dx = (v_padded[1:-1, 2:] - v_padded[1:-1, :-2]) / (2 * self.sim.grid_spacing_x)
            
            # Now both arrays have the same shape as the original u and v
            vorticity = dv_dx - du_dy
            
            # High positive vorticity (cyclonic) creates low pressure
            # High negative vorticity (anticyclonic) creates high pressure
            vorticity_factor = 0.3  # Increased from 0.2
            pressure_change_vorticity = -vorticity_factor * vorticity * self.sim.pressure * dt
            
            # 3. Apply baroclinic instability effects (interactions of temperature and pressure gradients)
            # Calculate magnitude of horizontal temperature gradient
            T_wrapped = np.vstack((T[-1:, :], T, T[:1, :]))
            T_wrapped = np.hstack((T_wrapped[:, -1:], T_wrapped, T_wrapped[:, :1]))
            
            # Calculate horizontal temperature gradients
            dT_dx = np.gradient(T_wrapped, axis=1)[1:-1, 1:-1] / self.sim.grid_spacing_x
            dT_dy = np.gradient(T_wrapped, axis=0)[1:-1, 1:-1] / self.sim.grid_spacing_y
            
            # Temperature gradient magnitude
            temp_gradient_mag = np.sqrt(dT_dx**2 + dT_dy**2)
            
            # Simplified baroclinic instability effect - stronger in areas with large temperature gradients
            # and where winds are already strong (reinforcing developing systems)
            wind_mag = np.sqrt(self.sim.u**2 + self.sim.v**2)
            baroclinic_factor = 0.2  # Increased from 0.15
            pressure_change_baroclinic = -baroclinic_factor * temp_gradient_mag * wind_mag * dt
            
            # Modulate the baroclinic effect with latitude - stronger in mid-latitudes where
            # frontogenesis and cyclone development are most common
            lat_rad = np.abs(np.radians(self.sim.latitude))
            midlat_enhancement = np.exp(-((lat_rad - np.radians(45)) / np.radians(25))**2)
            pressure_change_baroclinic *= midlat_enhancement
            
            # 4. Enhanced temperature factor - still important for overall circulation
            temperature_factor = 0.6  # Increased from 0.5
            
            # Cache global average temperature and update less frequently
            if not hasattr(self, '_global_avg_temp') or not hasattr(self, '_temp_avg_counter'):
                self._global_avg_temp = np.mean(T)
                self._temp_avg_counter = 0
            
            self._temp_avg_counter += 1
            if self._temp_avg_counter >= 10:  # Only update every 10 steps
                self._global_avg_temp = np.mean(T)
                self._temp_avg_counter = 0
            
            pressure_change_temp = -temperature_factor * (T - self._global_avg_temp) * dt
            
            # 5. Land-ocean thermal contrast - important for coastal weather
            land_ocean_effect = 0.5 * self._land_ocean_factor * (1 - self._land_ocean_factor) * dt  # Increased from 0.4
            pressure_change_land_ocean = land_ocean_effect * np.abs(np.gradient(T)[0] + np.gradient(T)[1])
            
            # 6. Background latitudinal pressure patterns - REDUCED influence to allow weather patterns
            # to dominate more
            lat_pressure_strength = 0.003  # Reduced from 0.005
            pressure_change_lat = self._lat_pressure * lat_pressure_strength * dt
            
            # 7. Semi-permanent pressure systems - also reduced to allow more dynamism
            persistent_system_strength = 0.01  # Keep at 0.01
            pressure_change_persistent = self._persistent_systems * persistent_system_strength * dt
            
            # 8. Add time-varying oscillations with seasonal component
            if not hasattr(self, '_oscillation_phase'):
                self._oscillation_phase = 0.0
                
            self._oscillation_phase += 0.015  # Increased from 0.01
            if self._oscillation_phase > 2 * np.pi:
                self._oscillation_phase -= 2 * np.pi
                
            oscillation_x = np.sin(2 * np.pi * self.sim.longitude / 360 + self._oscillation_phase)
            oscillation_y = np.sin(2 * np.pi * self.sim.latitude / 180 + self._oscillation_phase)
            # Add seasonal influence
            oscillation = oscillation_x * oscillation_y * (1 + 0.5 * seasonal_factor)
            
            oscillation_factor = 0.4  # Increased from 0.3
            pressure_change_oscillation = oscillation * oscillation_factor * dt
            
            # 9. Weather system persistence (memory effect) to simulate real weather patterns that
            # develop and persist over time rather than rapidly changing
            if not hasattr(self, '_pressure_anomaly'):
                self._pressure_anomaly = np.zeros_like(self.sim.pressure)
            
            # Current anomaly is difference from "climate" (the background pattern)
            climate_pressure = P0 * np.ones_like(self.sim.pressure)
            climate_pressure += self._lat_pressure * 0.1  # Background climate pattern
            current_anomaly = self.sim.pressure - climate_pressure
            
            # Persistence factor - how much of the anomaly persists
            # Reduce persistence to prevent systems from becoming too static
            persistence_factor = 0.9  # Reduced from 0.95
            self._pressure_anomaly = persistence_factor * self._pressure_anomaly + (1 - persistence_factor) * current_anomaly
            
            # Apply the persistent anomaly effect to create weather pattern memory
            weather_persistence_strength = 0.25  # Increased from 0.2
            pressure_change_persistence = weather_persistence_strength * self._pressure_anomaly * dt
            
            # 10. Combine all pressure changes with new weather dynamics
            pressure_change = (pressure_change_wind + 
                              pressure_change_temp + 
                              pressure_change_lat + 
                              pressure_change_vorticity +
                              pressure_change_baroclinic +
                              pressure_change_persistence +
                              pressure_change_land_ocean +
                              pressure_change_persistent +
                              pressure_change_oscillation)
            
            # 11. More localized, weather-like atmospheric variability
            # Use a combination of simulation time and a fixed seed for the random noise
            # This creates variability while ensuring it's not totally chaotic
            seed_value = int((time.time() * 1000 + self.sim.time_step * 17) % 10000)
            np.random.seed(seed_value)
            
            # Create two scales of noise: large-scale and small-scale
            # Large-scale weather systems
            base_noise_large = np.random.normal(0, 1.2, pressure_change.shape)  # Increased variance
            large_scale_noise = np.zeros_like(base_noise_large)
            gaussian_filter(base_noise_large, sigma=5.0, mode='wrap', output=large_scale_noise)
            
            # Smaller-scale disturbances
            base_noise_small = np.random.normal(0, 1.0, pressure_change.shape)
            small_scale_noise = np.zeros_like(base_noise_small)
            gaussian_filter(base_noise_small, sigma=2.0, mode='wrap', output=small_scale_noise)
            
            # Third scale - very localized disturbances
            base_noise_local = np.random.normal(0, 0.8, pressure_change.shape)
            local_noise = np.zeros_like(base_noise_local)
            gaussian_filter(base_noise_local, sigma=0.8, mode='wrap', output=local_noise)
            
            # Combined noise with more emphasis on large-scale patterns
            noise = large_scale_noise * 1.2 + small_scale_noise * 0.7 + local_noise * 0.3
            
            # Scale noise to have more effect in areas with strong temperature gradients
            # This simulates frontal disturbances
            frontal_enhancement = 1.0 + 0.5 * temp_gradient_mag / np.max(temp_gradient_mag + 1e-10)
            noise *= frontal_enhancement
            
            # Apply noise to pressure change with a stronger factor
            pressure_change += noise * 2.0  # Increased noise influence
            
            # 12. Lighter smoothing to preserve weather features
            if not hasattr(self, '_pressure_change_smooth'):
                self._pressure_change_smooth = np.zeros_like(pressure_change)
            
            # Reduce smoothing to preserve more weather detail
            gaussian_filter(pressure_change, sigma=0.6, mode='wrap', output=self._pressure_change_smooth)  # Reduced from 0.8
            
            # Apply pressure changes
            self.sim.pressure += self._pressure_change_smooth
            
            # 13. Ensure pressure stays within realistic bounds
            # First handle land areas with elevation-based minimum pressure
            self.sim.pressure[self._is_land] = np.maximum(self.sim.pressure[self._is_land], min_pressure[self._is_land])
            
            # For ocean areas, ensure more gradual pressure variation
            is_ocean = ~self._is_land
            
            # Get latitude dependent factors for ocean
            lat_rad = np.abs(np.radians(self.sim.latitude))
            
            # Create references for ocean pressure constraints based on latitude bands
            # These create smoother transitions between pressure zones in oceans
            # Near equator (low pressure)
            equator_band = np.exp(-(lat_rad / np.radians(15))**2)
            # Near 30° N/S (high pressure)
            subtropical_band = np.exp(-((lat_rad - np.radians(30)) / np.radians(15))**2)
            # Near 60° N/S (low pressure)
            subpolar_band = np.exp(-((lat_rad - np.radians(60)) / np.radians(15))**2)
            # Near poles (high pressure)
            polar_band = np.exp(-((lat_rad - np.radians(90)) / np.radians(20))**2)
            
            # Calculate reference pressure for each ocean point based on latitude
            base_ocean_pressure = P0 * np.ones_like(self.sim.pressure)
            
            # Apply latitude-based pressure modifications with reduced strength for oceans
            ocean_pressure_range = 3000.0
            base_ocean_pressure -= equator_band * 0.3 * ocean_pressure_range
            base_ocean_pressure += subtropical_band * 0.4 * ocean_pressure_range
            base_ocean_pressure -= subpolar_band * 0.3 * ocean_pressure_range
            base_ocean_pressure += polar_band * 0.3 * ocean_pressure_range
            
            # Allow pressure to deviate from base ocean pressure by more, increasing dynamism
            max_ocean_deviation = 3000.0  # Increased from 2500.0
            ocean_pressure_min = base_ocean_pressure - max_ocean_deviation
            ocean_pressure_max = base_ocean_pressure + max_ocean_deviation
            
            # Apply ocean-specific constraints before global clipping
            self.sim.pressure[is_ocean] = np.clip(
                self.sim.pressure[is_ocean],
                ocean_pressure_min[is_ocean],
                ocean_pressure_max[is_ocean]
            )
            
            # Final global clipping
            self.sim.pressure = np.clip(self.sim.pressure, 87000.0, 108600.0)
            
            # 14. Final smoothing - lighter to preserve weather features
            if not hasattr(self, '_pressure_smooth'):
                self._pressure_smooth = np.zeros_like(self.sim.pressure)
            
            # Reduced final smoothing for more detail
            gaussian_filter(self.sim.pressure, sigma=0.5, mode='wrap', output=self._pressure_smooth)  # Reduced from 0.8
            self.sim.pressure[:] = self._pressure_smooth
            
        except Exception as e:
            print(f"Error updating pressure: {e}")
            traceback.print_exc()

    def _add_pressure_system(self, field, lon_center, lat_center, radius, strength):
        """Add a pressure system (high or low) to a field at the specified location using vectorized operations"""
        # Convert longitude to 0-360 range if negative
        if lon_center < 0:
            lon_center += 360
        
        # Get grid longitudes and latitudes
        lon = self.sim.longitude
        lat = self.sim.latitude
        
        # Convert longitudes to match center's range for proper distance calculation
        # (e.g., if center is at 190°, we want points at -170° to be considered close)
        lon_adjusted = lon.copy()
        if lon_center > 180:
            # If center is in 180-360 range, adjust negative longitudes
            lon_adjusted[lon < 0] += 360
        elif lon_center < 0:
            # If center is negative, adjust positive longitudes > 180
            lon_center += 360
            lon_adjusted[lon > 180] -= 360
        
        # Convert to radians for distance calculation
        lon1, lat1 = np.radians(lon_adjusted), np.radians(lat)
        lon2, lat2 = np.radians(lon_center), np.radians(lat_center)
        
        # Calculate distance components using broadcasting
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        # Haversine formula (optimized for vectorized operations)
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = 6371 * c  # Distance in kilometers
        
        # Calculate pressure influence using gaussian profile
        # Faster than directly calculating e^(-x²)
        radius_km = radius  # Convert radius to km
        falloff = -(distance / radius_km)**2
        # Clip to avoid extreme small values
        falloff_clipped = np.clip(falloff, -10, 0)
        influence = strength * np.exp(falloff_clipped)
        
        # Add to field
        field += influence 