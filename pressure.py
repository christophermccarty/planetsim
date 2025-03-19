import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import time
import traceback
from map_generation import MapGenerator

class Pressure:
    def __init__(self, sim):
        """Initialize pressure module with reference to main simulation"""
        self.sim = sim
        self._oscillation_phase = 0
        self._time_varying_factor = 0
        self._temp_avg_counter = 0
        # Add adaptive resolution tracking
        self._adaptive_mode = False
        self._downsampling_factor = 2  # Default reduction factor
        
        # Pattern-Based Pressure System settings (Solution 1)
        self._update_counter = 0
        self._base_pattern_update_freq = 10    # Update base patterns less frequently
        self._perturbation_update_freq = 1     # Update perturbations every step
        
        # Base patterns - decompose pressure into components
        self._static_pattern = None          # Elevation-based component
        self._latitude_pattern = None        # Global circulation component
        self._thermal_pattern = None         # Temperature-driven component  
        self._dynamic_pattern = None         # Time-varying oscillatory component
        self._perturbation = None            # Fast-changing small-scale perturbations
        
        # Cached intermediate calculations
        self._last_perturbation_time = 0     # Track when perturbations were last updated
        self._persistent_systems = None      # Semi-persistent weather systems

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
            self.sim.pressure_normalized = MapGenerator.normalize_data(self.sim.pressure)
            print("Pressure initialization complete.")
            
        except Exception as e:
            print(f"Error in pressure initialization: {e}")
            traceback.print_exc()

    def update(self):
        """Update pressure using pattern-based approach for performance optimization"""
        try:
            dt = self.sim.time_step_seconds
            P0 = 101325.0  # Standard sea-level pressure
            
            # Check if high-speed mode is active
            high_speed = getattr(self.sim, 'high_speed_mode', False)
            
            # Use adaptive resolution if in high-speed mode
            if high_speed and high_speed != self._adaptive_mode:
                # Mode switched to high-speed, prepare for adaptive resolution
                self._adaptive_mode = True
                # Clear cached fields to force recalculation
                if hasattr(self, '_elevation_factor'):
                    del self._elevation_factor
                
            elif not high_speed and self._adaptive_mode:
                # Mode switched to normal, revert to full resolution
                self._adaptive_mode = False
                # Clear cached fields
                if hasattr(self, '_elevation_factor'):
                    del self._elevation_factor
            
            # Use adaptive resolution calculations in high-speed mode
            if self._adaptive_mode:
                self._update_adaptive()
                return
                
            # Increment update counter
            self._update_counter += 1
                
            # Regular full-resolution update with pattern-based optimization
            # Initialize masks and basic patterns if not already done
            if self._static_pattern is None or self._latitude_pattern is None:
                self._initialize_patterns()
            
            # Update time-varying components
            self._update_time_varying_components(dt)
            
            # --- PATTERN-BASED PRESSURE CALCULATION ---
            # The key idea is to decompose pressure into components that update at different rates
            
            # 1. Start with static elevation-based pattern (rarely changes)
            current_pressure = self._static_pattern.copy()
            
            # 2. Add latitude-based global circulation pattern (changes very slowly)
            if self._update_counter % self._base_pattern_update_freq == 0 or self._latitude_pattern is None:
                self._update_latitude_pattern()
            current_pressure += self._latitude_pattern
            
            # 3. Add thermal pattern (changes moderately with temperature)
            if self._update_counter % (self._base_pattern_update_freq // 2) == 0 or self._thermal_pattern is None:
                self._update_thermal_pattern()
            current_pressure += self._thermal_pattern
            
            # 4. Add dynamic oscillatory pattern (changes slowly over time)
            if self._update_counter % (self._base_pattern_update_freq // 5) == 0 or self._dynamic_pattern is None:
                self._update_dynamic_pattern(dt)
            current_pressure += self._dynamic_pattern
            
            # 5. Add fast-changing perturbations (updated every step)
            self._update_perturbations(dt)
            current_pressure += self._perturbation
            
            # 6. Apply constraints - ensure pressure stays within realistic bounds
            current_pressure = np.clip(current_pressure, 87000.0, 108600.0)
            
            # 7. Light smoothing to remove any artifacts from component combination
            gaussian_filter(current_pressure, sigma=0.5, mode='wrap', output=self.sim.pressure)
            
        except Exception as e:
            print(f"Error updating pressure: {e}")
            traceback.print_exc()
    
    def _initialize_patterns(self):
        """Initialize the base pressure patterns"""
        try:
            P0 = 101325.0  # Standard sea-level pressure
            
            # 1. STATIC PATTERN (elevation-based)
            is_land = self.sim.elevation > 0
            max_height = 1000.0
            limited_elevation = np.clip(self.sim.elevation, 0, max_height)
            elevation_factor = np.exp(-limited_elevation / 7400.0)
            
            # Create static pressure pattern
            self._static_pattern = P0 * elevation_factor
            self._is_land = is_land
            
            # 2. LATITUDE PATTERN (empty placeholder - will be filled on first update)
            self._latitude_pattern = np.zeros_like(self.sim.pressure)
            
            # 3. THERMAL PATTERN (empty placeholder - will be filled on first update)
            self._thermal_pattern = np.zeros_like(self.sim.pressure)
            
            # 4. DYNAMIC PATTERN (empty placeholder - will be filled on first update)
            self._dynamic_pattern = np.zeros_like(self.sim.pressure)
            
            # 5. PERTURBATIONS (empty placeholder - will be filled on first update)
            self._perturbation = np.zeros_like(self.sim.pressure)
            
            # 6. Initialize persistent weather systems
            self._initialize_persistent_systems()
            
        except Exception as e:
            print(f"Error initializing pressure patterns: {e}")
            traceback.print_exc()
            
    def _update_latitude_pattern(self):
        """Update the latitude-based global circulation pattern"""
        try:
            P0 = 101325.0  # Standard sea-level pressure
            lat_rad = np.abs(np.radians(self.sim.latitude))
            
            # Latitude bands with typical pressure systems
            # Equatorial trough (low pressure)
            equator_band = np.exp(-((lat_rad - np.radians(0)) / np.radians(10))**2)
            # Near 30° N/S (high pressure)
            subtropical_band = np.exp(-((lat_rad - np.radians(30)) / np.radians(15))**2)
            # Near 60° N/S (low pressure)
            subpolar_band = np.exp(-((lat_rad - np.radians(60)) / np.radians(15))**2)
            # Near poles (high pressure)
            polar_band = np.exp(-((lat_rad - np.radians(90)) / np.radians(20))**2)
            
            # Create latitude-based pressure pattern
            self._latitude_pattern = np.zeros_like(self.sim.pressure)
            
            # Apply different pressure ranges for land and ocean
            pressure_range = np.where(self._is_land, 2500.0, 3000.0)
            
            # Build latitude pattern with typical global circulation features
            self._latitude_pattern -= equator_band * 0.4 * pressure_range
            self._latitude_pattern += subtropical_band * 0.5 * pressure_range
            self._latitude_pattern -= subpolar_band * 0.4 * pressure_range
            self._latitude_pattern += polar_band * 0.3 * pressure_range
            
            # Add persistent semi-permanent systems
            if self._persistent_systems is not None:
                self._latitude_pattern += self._persistent_systems
            
        except Exception as e:
            print(f"Error updating latitude pattern: {e}")
            traceback.print_exc()
            
    def _update_thermal_pattern(self):
        """Update the temperature-based pressure pattern"""
        try:
            # Calculate global average temperature for reference
            global_avg_temp = np.mean(self.sim.temperature_celsius)
            
            # Temperature deviation from average
            temp_deviation = self.sim.temperature_celsius - global_avg_temp
            
            # 1. Basic thermal effect - hot air rises (low pressure), cold air sinks (high pressure)
            thermal_factor = -30.0  # Scale factor (negative because hot = low pressure)
            self._thermal_pattern = thermal_factor * temp_deviation
            
            # 2. Apply land-sea thermal contrast effects
            # Land-sea temperature differences drive pressure gradients
            is_land = self._is_land
            
            # Calculate separate land and ocean average temperatures
            if np.any(is_land):
                land_avg_temp = np.mean(self.sim.temperature_celsius[is_land])
            else:
                land_avg_temp = global_avg_temp
                
            if np.any(~is_land):
                ocean_avg_temp = np.mean(self.sim.temperature_celsius[~is_land])
            else:
                ocean_avg_temp = global_avg_temp
            
            # Enhance pressure differences along coastlines based on land-sea temperature contrast
            if np.any(is_land) and np.any(~is_land):
                # Find coastlines using simple dilation technique
                from scipy.ndimage import binary_dilation
                
                # Define a kernel for finding coastal regions
                kernel = np.ones((3, 3), dtype=bool)
                kernel[1, 1] = False
                
                # Find cells that are land but adjacent to ocean
                coastal_land = is_land & binary_dilation(~is_land, structure=kernel)
                
                # Find cells that are ocean but adjacent to land
                coastal_ocean = (~is_land) & binary_dilation(is_land, structure=kernel)
                
                # Enhance thermal pattern along coastlines
                coastal_factor = 15.0 * (land_avg_temp - ocean_avg_temp)
                if np.any(coastal_land):
                    self._thermal_pattern[coastal_land] += coastal_factor
                if np.any(coastal_ocean):
                    self._thermal_pattern[coastal_ocean] -= coastal_factor
            
            # 3. Apply light smoothing to thermal pattern
            self._thermal_pattern = gaussian_filter(self._thermal_pattern, sigma=1.0, mode='wrap')
            
        except Exception as e:
            print(f"Error updating thermal pattern: {e}")
            traceback.print_exc()
    
    def _update_dynamic_pattern(self, dt):
        """Update the dynamic oscillatory pressure pattern"""
        try:
            # Update time-varying oscillation phase
            self._time_varying_factor += 0.1 * dt / 86400.0  # Convert to days
            if self._time_varying_factor > 2*np.pi:
                self._time_varying_factor -= 2*np.pi
                
            # 1. Create spatial oscillation patterns
            height, width = self.sim.pressure.shape
            x = np.linspace(0, 2*np.pi, width)
            y = np.linspace(0, np.pi, height)
            X, Y = np.meshgrid(x, y)
            
            # 2. Generate phase-shifted oscillation pattern
            phase = self._oscillation_phase
            wave1 = np.sin(X/2 + phase) * np.cos(Y*2)
            wave2 = np.cos(X + phase*1.5) * np.sin(Y*3)
            wave3 = np.sin(X*3 + Y + phase*0.7)
            
            # 3. Combine waves with varying amplitudes
            self._dynamic_pattern = (wave1 * 400.0 + wave2 * 300.0 + wave3 * 250.0)
            
            # 4. Modulate strength with latitude (stronger in mid-latitudes)
            lat_rad = np.abs(np.radians(self.sim.latitude))
            midlat_factor = np.exp(-((lat_rad - np.radians(45)) / np.radians(30))**2)
            self._dynamic_pattern *= midlat_factor
            
            # 5. Advance oscillation phase for next update
            self._oscillation_phase += 0.02
            if self._oscillation_phase > 2*np.pi:
                self._oscillation_phase -= 2*np.pi
                
        except Exception as e:
            print(f"Error updating dynamic pattern: {e}")
            traceback.print_exc()
    
    def _update_perturbations(self, dt):
        """Update fast-changing small-scale pressure perturbations"""
        try:
            # Only regenerate perturbations periodically to save computation
            should_update = (time.time() - self._last_perturbation_time > 0.1) or (self._perturbation is None)
            
            if not should_update:
                return
                
            # Record update time
            self._last_perturbation_time = time.time()
            
            # 1. Generate small-scale noise pattern
            # Use deterministic seed based on time and time step for reproducibility
            np.random.seed(int((time.time() * 1000 + self.sim.time_step * 17) % 10000))
            
            # Generate different scales of noise
            noise_small = np.random.normal(0, 1.0, self.sim.pressure.shape)
            noise_medium = np.random.normal(0, 1.0, self.sim.pressure.shape)
            
            # 2. Smooth noise to create coherent patterns at different scales
            noise_small = gaussian_filter(noise_small, sigma=1.0, mode='wrap')
            noise_medium = gaussian_filter(noise_medium, sigma=3.0, mode='wrap')
            
            # 3. Combine noise scales with different weights
            combined_noise = noise_small * 50.0 + noise_medium * 150.0
            
            # 4. Apply wind influence on perturbations
            if hasattr(self.sim, 'u') and hasattr(self.sim, 'v'):
                # Calculate wind convergence/divergence
                du_dx = np.gradient(self.sim.u, axis=1)
                dv_dy = np.gradient(self.sim.v, axis=0)
                convergence = -(du_dx + dv_dy)  # Negative divergence
                
                # Scale and smooth convergence effect
                convergence_effect = gaussian_filter(convergence * 500.0, sigma=1.0, mode='wrap')
                
                # Add to perturbations (convergence lowers pressure)
                combined_noise += convergence_effect
            
            # 5. Update perturbation field with temporal smoothing
            if self._perturbation is None:
                self._perturbation = combined_noise
            else:
                # Blend with previous perturbation for temporal continuity
                blend_factor = 0.7
                self._perturbation = self._perturbation * blend_factor + combined_noise * (1 - blend_factor)
            
        except Exception as e:
            print(f"Error updating pressure perturbations: {e}")
            traceback.print_exc()
            
    def _initialize_persistent_systems(self):
        """Initialize semi-permanent pressure systems (e.g., Aleutian Low, Azores High)"""
        try:
            # Create empty field for persistent systems
            self._persistent_systems = np.zeros_like(self.sim.pressure)
            
            # Add known semi-permanent pressure systems
            # Parameters: lon_center, lat_center, radius (km), strength (Pa)
            
            # Northern Hemisphere systems
            self._add_pressure_system(self._persistent_systems, -30, 35, 1500, 1200)   # Azores/Bermuda High
            self._add_pressure_system(self._persistent_systems, -165, 55, 1800, -1500) # Aleutian Low
            self._add_pressure_system(self._persistent_systems, -100, 85, 1200, 800)   # North Polar High
            self._add_pressure_system(self._persistent_systems, 100, 45, 2000, 1000)   # Siberian High
            
            # Southern Hemisphere systems
            self._add_pressure_system(self._persistent_systems, -90, -40, 1500, 1200)  # South Pacific High
            self._add_pressure_system(self._persistent_systems, 20, -35, 1500, 1000)   # South Atlantic High
            self._add_pressure_system(self._persistent_systems, 120, -35, 1500, 1000)  # South Indian High
            self._add_pressure_system(self._persistent_systems, 0, -88, 1200, 800)     # South Polar High
            
            # Apply smoothing to blend pressure systems
            self._persistent_systems = gaussian_filter(self._persistent_systems, sigma=3.0, mode='wrap')
            
        except Exception as e:
            print(f"Error initializing persistent pressure systems: {e}")
            traceback.print_exc()
    
    def _update_time_varying_components(self, dt):
        """Update time-varying parameters that affect pressure patterns"""
        # Update oscillation phase - used for dynamic pattern
        self._oscillation_phase += 0.015 * dt / 3600.0  # Scale by hours
        if self._oscillation_phase > 2 * np.pi:
            self._oscillation_phase -= 2 * np.pi
            
        # Update global time-varying factor - used for seasonal effects
        self._time_varying_factor += 0.1 * dt / 86400.0  # Scale by days
        if self._time_varying_factor > 2 * np.pi:
            self._time_varying_factor -= 2 * np.pi

    def _update_adaptive(self):
        """Update pressure using lower resolution calculations for performance"""
        try:
            dt = self.sim.time_step_seconds
            P0 = 101325.0  # Standard sea-level pressure
            factor = self._downsampling_factor
            
            # Downsample key fields when needed
            if not hasattr(self, '_downsampled_fields') or self.sim.time_step % 20 == 0:
                # Create or update cached downsampled fields
                h, w = self.sim.map_height, self.sim.map_width
                h_low, w_low = h // factor, w // factor
                
                # Downsample elevation and land mask
                elevation_low = zoom(self.sim.elevation, 1/factor, order=1)
                is_land_low = elevation_low > 0
                
                # Cache elevation factor
                max_height = 1000.0
                limited_elevation = np.clip(elevation_low, 0, max_height)
                elevation_factor_low = np.exp(-limited_elevation / 7400.0)
                
                # Cache latitude-based patterns at lower resolution
                lat_rad = np.radians(zoom(self.sim.latitude, 1/factor, order=1))
                lon_deg = zoom(self.sim.longitude, 1/factor, order=1)
                
                # Create basic pressure pattern based on latitude (simplified)
                lat_pressure_low = (
                    800.0 * np.exp(-((np.abs(lat_rad) - np.radians(30)) / np.radians(15))**2) +  # Subtropical high
                    -600.0 * np.exp(-((np.abs(lat_rad) - np.radians(60)) / np.radians(15))**2) +  # Subpolar low
                    600.0 * np.exp(-((np.abs(lat_rad) - np.radians(90)) / np.radians(20))**2) +   # Polar high
                    -400.0 * np.exp(-(lat_rad / np.radians(10))**2)                               # Equatorial low
                )
                
                # Create persistent systems (simplified version)
                persistent_systems_low = np.zeros_like(lat_pressure_low)
                
                # Cache the downsampled fields
                self._downsampled_fields = {
                    'elevation': elevation_low,
                    'is_land': is_land_low,
                    'elevation_factor': elevation_factor_low,
                    'lat_pressure': lat_pressure_low,
                    'persistent_systems': persistent_systems_low,
                    'shape': (h_low, w_low)
                }
            
            # Get downsampled fields
            ds = self._downsampled_fields
            
            # Downsample current fields
            temp_low = zoom(self.sim.temperature_celsius, 1/factor, order=1)
            u_low = zoom(self.sim.u, 1/factor, order=1)
            v_low = zoom(self.sim.v, 1/factor, order=1)
            pressure_low = zoom(self.sim.pressure, 1/factor, order=1)
            
            # Calculate global average temperature
            global_avg_temp = np.mean(temp_low)
            
            # Update time-varying factor
            self._time_varying_factor += 0.1 * dt
            seasonal_factor = np.sin(self._time_varying_factor / 86400 * 2 * np.pi)
            
            # Calculate pressure changes (simplified for low resolution)
            # 1. Wind-driven changes
            du_dx = np.gradient(u_low, axis=1) 
            dv_dy = np.gradient(v_low, axis=0)
            wind_divergence = du_dx + dv_dy
            pressure_change_wind = -pressure_low * wind_divergence * dt * 0.35
            
            # 2. Temperature-driven changes 
            pressure_change_temp = -0.6 * (temp_low - global_avg_temp) * dt
            
            # 3. Background patterns
            pressure_change_lat = ds['lat_pressure'] * 0.003 * dt
            
            # 4. Oscillation
            self._oscillation_phase += 0.015
            if self._oscillation_phase > 2 * np.pi:
                self._oscillation_phase -= 2 * np.pi
                
            # Simple oscillation on low-res grid
            x = np.linspace(0, 2*np.pi, ds['shape'][1])
            y = np.linspace(0, np.pi, ds['shape'][0])
            X, Y = np.meshgrid(x, y)
            oscillation = np.sin(X + self._oscillation_phase) * np.sin(Y + self._oscillation_phase) * (1 + 0.5 * seasonal_factor)
            pressure_change_oscillation = oscillation * 0.4 * dt
            
            # 5. Simplified stochastic weather patterns
            np.random.seed(int((time.time() * 1000 + self.sim.time_step * 17) % 10000))
            noise_large = np.random.normal(0, 1.0, ds['shape'])
            gaussian_filter(noise_large, sigma=2.0, mode='wrap', output=noise_large)
            
            # Combine all pressure changes
            pressure_change_low = (pressure_change_wind + 
                               pressure_change_temp + 
                               pressure_change_lat + 
                               pressure_change_oscillation + 
                               noise_large * 2.0)
            
            # Apply light smoothing
            gaussian_filter(pressure_change_low, sigma=0.5, mode='wrap', output=pressure_change_low)
            
            # Update low-resolution pressure
            pressure_low += pressure_change_low
            
            # Apply constraints
            pressure_low = np.clip(pressure_low, 87000.0, 108600.0)
            
            # Upsample back to full resolution and update sim's pressure field
            full_res_pressure = zoom(pressure_low, factor, order=1)
            
            # Ensure proper dimensions (in case of rounding differences)
            if full_res_pressure.shape != self.sim.pressure.shape:
                full_res_pressure = np.resize(full_res_pressure, self.sim.pressure.shape)
                
            self.sim.pressure[:] = full_res_pressure
            
        except Exception as e:
            print(f"Error in adaptive pressure update: {e}")
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