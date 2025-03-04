import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time
import json
import multiprocessing
import threading
from scipy.ndimage import gaussian_filter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import RegularGridInterpolator
from numba import jit, set_num_threads
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
import psutil
import traceback
import gc
from temperature import Temperature
from pressure import Pressure
from wind import Wind

from map_generation import MapGenerator


class SimulationApp:
    def __init__(self):
        """Initialize the simulation"""
        try:
            # Initialize all variables
            self.map_size = 512
            self.map_width = None  # Will be set when image is loaded
            self.map_height = None # Will be set when image is loaded
            
            # Time and simulation variables
            self.time_step = 0
            self.time_step_seconds = 60 * 30  # Simulation time step in seconds (30 minutes)
            self.Omega = 7.2921159e-5
            self.P0 = 101325
            self.desired_simulation_step_time = 0.1  # 0.1 seconds between simulation steps

            # Master switch for print_systemn_stats
            self.print_stats_enabled = True

            # Add energy budget tracking
            self.energy_budget = {
                'incoming': 0,
                'outgoing': 0,
                'greenhouse': 0,
                'albedo': 0,
                'cloud_effect': 0,
                'ocean_flux': 0,
                'ocean_heat_content': 0,
                'ocean_heat_change': 0,
                'evaporation': 0,
                'precipitation': 0,
                'humidity_transport': 0,
                'temperature_history': []  # Initialize temperature history list
            }

            # Adjustable climate parameters
            self.climate_params = {
                'solar_constant': 1361,      # Solar constant (W/m²)
                'albedo_land': 0.3,          # Land albedo
                'albedo_ocean': 0.06,        # Ocean albedo
                'emissivity': 0.95,          # Increased to allow more heat escape
                'climate_sensitivity': 3.0,   # Reduced for more stable response
                'greenhouse_strength': 2.5,   # Reduced greenhouse effect
                'heat_capacity_land': 0.8e6,    # Doubled land heat capacity for stability
                'heat_capacity_ocean': 4.2e6,   # Doubled ocean heat capacity for stability
                'ocean_heat_transfer': 0.3,   # Ocean heat transfer
                'atmospheric_heat_transfer': 2.0,  # Increased from 0.5 to 2.0 for smoother diffusion
                'baseline_land_temp': 15,   # Baseline land temperature
                'max_ocean_temp': 32.0,       # Maximum ocean temperature
                'min_ocean_temp': -2.0        # Minimum ocean temperature
            }
            
            # Mouse tracking
            self.last_mouse_x = 0
            self.last_mouse_y = 0
            
            # Seeds
            self.seed = 4200
            self.elevation_seed = self.seed
            
            # Simulation parameters
            self.global_octaves = 5
            self.global_frequency = 2.0
            self.global_lacunarity = 2.0
            self.global_persistence = 0.5
            
            # Initialize arrays
            self.altitude = None
            self.elevation = None
            self.elevation_normalized = None
            self.temperature_celsius = None
            self.previous_temperature_celsius = 0
            self.temperature_normalized = None
            self.pressure = None
            self.pressure_normalized = None

            self.u = None
            self.v = None
            self.ocean_u = None
            self.ocean_v = None

            # Atmospheric parameters
            # Greenhouse gas concentrations
            # Values based on approximate current atmospheric concentrations
            self.co2_concentration = 420  # ppm
            self.ch4_concentration = 1900  # ppb
            self.n2o_concentration = 330  # ppb

            # Radiative forcing coefficients (W/m² per ppm or ppb)
            self.radiative_forcing_co2 = 5.35 * np.log(self.co2_concentration / 280)  # Reference CO2 is 280 ppm
            self.radiative_forcing_ch4 = 0.036 * (self.ch4_concentration - 1750) / 1000  # Approximation
            self.radiative_forcing_n2o = 0.12 * (self.n2o_concentration - 270) / 1000  # Approximation

            # Total Radiative Forcing from Greenhouse Gases
            self.total_radiative_forcing = self.radiative_forcing_co2 + self.radiative_forcing_ch4 + self.radiative_forcing_n2o
            
            # Multiprocessing setup
            self.num_processes = multiprocessing.cpu_count() - 20
            if self.num_processes < 1:
                self.num_processes = 1
            print(f'Number of CPU cores available: {self.num_processes}')
            
            # Create Tkinter window
            self.root = tk.Tk()
            self.root.title("Climate Simulation - Planet Sim")
            
            # Variable to track the selected layer
            self.selected_layer = tk.StringVar(value="Wind")
            
            # Load initial data (this will set map dimensions)
            # self.on_load("D:\\dev\\planetsim\\images\\1024x512_earth_8bit.png")
            self.on_load("D:\dev\planetsim\images\GRAY_HR_SR_W_stretched.tif")
            
            # Setup GUI components (after we know the map dimensions)
            self.setup_gui()

            # Initialize coordinate arrays first
            latitudes = np.linspace(90, -90, self.map_height)
            longitudes = np.linspace(-180, 180, self.map_width)
            self.latitude, self.longitude = np.meshgrid(latitudes, longitudes, indexing='ij')
            self.latitudes_rad = np.deg2rad(self.latitude)

            # Initialize basic humidity field (will be updated later)
            self.humidity = np.full((self.map_height, self.map_width), 0.5, dtype=np.float32)

            # Initialize temperature array
            self.temperature_celsius = np.zeros((self.map_height, self.map_width), dtype=np.float32)
            
            # Initialize temperature module
            self.temperature = Temperature(self)

            # Initialize temperature
            self.temperature.initialize()
            
            # Initialize pressure module
            self.pressure_system = Pressure(self)
            
            # Initialize pressure
            self.pressure_system.initialize()
            
            # Initialize wind module
            self.wind_system = Wind(self)

            # Initialize humidity
            self.initialize_humidity()

            # Initialize other attributes
            self.humidity_effect_coefficient = 0.1

            # Remove explicit calls to wind methods since they're now handled by the wind_system
            # Initialize global circulation
            # self.initialize_global_circulation()

            # Initialize wind field
            # self.initialize_wind()

            # Initialize wind via the wind system
            self.wind_system.initialize()

            # Initialize ocean currents
            self.initialize_ocean_currents()

            # Initialize threading event for visualization control
            self.visualization_active = threading.Event()
            self.visualization_active.set()  # Start in active state

            # Start visualization in separate thread
            self.visualization_thread = threading.Thread(target=self.update_visualization_loop)
            self.visualization_thread.daemon = True
            self.visualization_thread.start()

            # Start simulation using Tkinter's after method instead of a thread
            # This ensures all Tkinter interactions happen in the main thread
            self.root.after(100, self.simulate)

            # Trigger initial map update
            self.update_map()
            
            # Add this near the end of your __init__ method
            # Flag for ocean diagnostics
            self.ocean_data_available = False
            
            # Remove duplicated initialization
            # self.energy_budget = {}
            
            # Optimization: Precompute and cache temperature-dependent values
            self._temp_cache = {}
            self._saturation_vapor_cache = None
            self._temp_range_for_cache = None
            
            # Optimization: Reusable arrays for calculations to minimize memory allocations
            self._reusable_array = None
            self._reusable_array2 = None
            
            # Optimization: Arrays for wind calculations
            self._u_geo = None
            self._v_geo = None
            self._u_new = None
            self._v_new = None
            
            # Optimization: Arrays for temperature calculations
            self._base_albedo = None
            self._solar_in = None
            self._longwave_out = None
            self._net_flux = None
            self._heat_capacity = None
            self._delta_T = None
            
            # Optimization: Arrays for cloud calculations
            self._relative_humidity = None
            self._cloud_coverage = None
            
            # Create a Tkinter root window - ALREADY CREATED ABOVE, DON'T CREATE A SECOND ONE
            # self.root = tk.Tk()
            # self.root.title("Climate Simulation")
            
            # Update the title of the existing window to be more descriptive
            self.root.title("Climate Simulation - Planet Sim")
            
            # Initialize the energy budget dictionary
            self.energy_budget = {
                'incoming': 0,
                'outgoing': 0,
                'greenhouse': 0,
                'albedo': 0,
                'cloud_effect': 0,
                'ocean_flux': 0,
                'ocean_heat_content': 0,
                'ocean_heat_change': 0,
                'evaporation': 0,
                'precipitation': 0,
                'humidity_transport': 0,
                'temperature_history': []  # Initialize temperature history list
            }
            
            # Flag to track if ocean data is available and calculated successfully
            self.ocean_data_available = False
         
            # Initialize wind
            self.wind_system.initialize()

        except Exception as e:
            print(f"Error in initialization: {e}")
            traceback.print_exc()

    def setup_gui(self):
        # Configure the root window to prevent resizing
        self.root.resizable(False, False)
        
        # Calculate window dimensions
        # Add buffer for UI elements: 60px for controls and 30px for mouse_over label
        ui_buffer = 90
        window_width = self.map_width
        window_height = self.map_height + ui_buffer
        
        # Set window geometry
        self.root.geometry(f'{window_width}x{window_height}')
        
        # Create the menu bar
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self.on_new)
        file_menu.add_command(label="Open", command=self.on_load)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=False, fill=tk.BOTH)

        # Initialize the canvas with exact map dimensions
        self.canvas = tk.Canvas(main_frame, 
                            width=self.map_width,
                            height=self.map_height)
        self.canvas.pack(expand=False)

        # Frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X)
        
        # Create a frame for radio buttons
        checkbox_frame = tk.Frame(self.root)
        checkbox_frame.pack(anchor='w')

        # Add radio buttons for each map layer horizontally
        tk.Radiobutton(checkbox_frame, text="Elevation", variable=self.selected_layer, value="Elevation",
                    command=self.update_map).pack(side=tk.LEFT)
        tk.Radiobutton(checkbox_frame, text="Altitude", variable=self.selected_layer, value="Altitude", 
                    command=self.update_map).pack(side=tk.LEFT)
        tk.Radiobutton(checkbox_frame, text="Temperature", variable=self.selected_layer, value="Temperature",
                    command=self.update_map).pack(side=tk.LEFT)
        tk.Radiobutton(checkbox_frame, text="Wind", variable=self.selected_layer, value="Wind", 
                    command=self.update_map).pack(side=tk.LEFT)
        tk.Radiobutton(checkbox_frame, text="Ocean Temperature", variable=self.selected_layer, value="Ocean Temperature",
                    command=self.update_map).pack(side=tk.LEFT)
        tk.Radiobutton(checkbox_frame, text="Precipitation", variable=self.selected_layer, value="Precipitation",
                    command=self.update_map).pack(side=tk.LEFT)

        # Add new radio button for pressure map
        tk.Radiobutton(
            control_frame,
            text="Pressure",
            variable=self.selected_layer,
            value="Pressure",
            command=self.update_map
        ).pack(anchor=tk.W)

        # Label at the bottom to display the current values at the mouse position
        self.mouse_over_label = tk.Label(self.root, text="Pressure: --, Temperature: --, Wind Speed: --, Wind Direction: --")
        self.mouse_over_label.pack(anchor="w")

        # Bind the mouse motion event to the canvas
        self.canvas.bind("<Motion>", self.update_mouse_over)

        # Initialize zoom dialog
        self.zoom_dialog = ZoomDialog(self.root)
        self.zoom_dialog.withdraw()  # Hide initially
        
        # Bind mouse events
        self.canvas.bind("<Motion>", self.update_zoom_view)
        self.canvas.bind("<Leave>", lambda e: self.zoom_dialog.withdraw())
        self.canvas.bind("<Enter>", lambda e: self.zoom_dialog.deiconify())

        # Add cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Handle window closing"""
        self.cleanup()
        self.root.destroy()

    def on_new(self):
        # [Implement as in the original code...]
        try:
            # Generate new random terrain data and then start simulation
            self.on_generate()

            # Reset to default view
            self.selected_layer.set("Elevation")
            self.update_map()
        except Exception as e:
            print(f"Error in on_new: {e}")
            traceback.print_exc()

    def on_load(self, file_path=None):
        """Load and process elevation image"""
        if file_path:
            try:
                elevation_image = Image.open(file_path)
                self.map_width, self.map_height = elevation_image.size
                
                # Calculate grid spacing
                earth_circumference = 40075000  # meters
                self.grid_spacing_y = earth_circumference / self.map_height
                self.grid_spacing_x = earth_circumference / self.map_width

                # Convert image to numpy array
                img_array = np.array(elevation_image)
                img_array = img_array.astype(int)
                if len(img_array.shape) > 2:
                    img_array = img_array[:, :, 0]

                # Store raw pixel values
                self.altitude = img_array.copy()

                # Initialize elevation array
                self.elevation = np.zeros_like(img_array, dtype=float)

                # Create masks using pixel value 68 as sea level
                water_mask = img_array <= 68
                land_mask = img_array > 68

                # Calculate land elevations using modified logarithmic scale
                if np.any(land_mask):
                    max_elevation = 8848  # meters (Mount Everest)
                    land_pixels = img_array[land_mask]
                    
                    # Convert pixel values to 0-1 range for land (68 to 254)
                    normalized_pixels = (land_pixels - 68) / (254 - 68)
                    
                    # Apply combined logarithmic and power scaling for elevation distribution
                    log_factor = 0.5  # reduced for even gentler curve
                    power_factor = 2.5  # increased for more emphasis on lower elevations
                    
                    elevation_values = max_elevation * (
                        (np.log(1 + normalized_pixels * log_factor) / np.log(1 + log_factor)) ** power_factor
                    )
                    
                    self.elevation[land_mask] = elevation_values

                # Water tiles are 0m elevation
                self.elevation[water_mask] = 0.0

                # Normalize elevation for display
                elevation_min = self.elevation.min()
                elevation_max = self.elevation.max()
                if elevation_max > elevation_min:
                    self.elevation_normalized = (self.elevation - elevation_min) / (elevation_max - elevation_min)
                else:
                    self.elevation_normalized = np.zeros_like(self.elevation)
            except Exception as e:
                print(f"Error with on_load: {e}")
                raise e

    def on_generate(self, elevation_data=None):
        # Cancel the previous simulation if running
        if hasattr(self, 'simulation_after_id'):
            self.root.after_cancel(self.simulation_after_id)

        # Reset time_step
        self.time_step = 0

        # Generate coordinate arrays
        x = np.linspace(0, 1, self.map_width, endpoint=False)
        y = np.linspace(0, 1, self.map_height, endpoint=False)
        x_coords, y_coords = np.meshgrid(x, y)

        # Grid spacing in meters
        earth_circumference = 40075000
        self.grid_spacing_x = earth_circumference / self.map_width
        self.grid_spacing_y = earth_circumference / (2 * self.map_height)

        # Initialize humidity
        self.humidity = np.full((self.map_height, self.map_width), 50, dtype=np.float32)

        # Convert latitude to radians
        self.latitudes_rad = np.deg2rad(self.latitude)

        # Use fixed values instead of sliders
        self.global_octaves = 6
        self.global_frequency = 2.0
        self.global_lacunarity = 2.0
        self.global_persistence = 0.5

        # Initialize temperature
        self.temperature.initialize()

        # Rest of the on_generate implementation...
        # (Continue with the elevation generation and temperature calculation code)

    def simulate(self):
        """Run simulation step"""
        try:
            start_time = time.time()
            
            # Increment time step
            self.time_step += 1
            
            # Update physics fields in proper order
            self.update_humidity()           # Update humidity first (depends on prev. temperature)
            self.temperature.update_land_ocean()      # Update temperatures (depends on humidity)
            self.temperature.update_ocean()  # Update ocean temps (uses updated surface temps)
            self.pressure_system.update()          # Update pressure (affected by temperature)
            self.wind_system.update()              # Finally update winds (affected by all others)
            
            # Clear the pressure image cache after updating pressure to force redraw
            if hasattr(self, '_pressure_image'):
                delattr(self, '_pressure_image')
            # Also clear the cached pressure visualization data to force a fresh redraw
            if hasattr(self, '_cached_pressure_viz'):
                delattr(self, '_cached_pressure_viz')
            if hasattr(self, '_cached_pressure_data'):
                delattr(self, '_cached_pressure_data')
            
            # Schedule GUI updates to happen in the main thread
            self.root.after(0, self.update_map)
            self.root.after(0, self.print_system_stats)
            self.root.after(0, self.update_mouse_over)
            
            elapsed_time = time.time() - start_time
            # print(f"Cycle {self.time_step} completed in {elapsed_time:.4f} seconds")
            
            # Schedule next simulation step with a reduced or zero delay
            # Option 1: Set a smaller minimum delay
            min_delay = 1  # milliseconds
            delay = max(min_delay, int(elapsed_time * 1000))
            
            # Option 2: Remove min_delay to run as fast as possible
            # delay = max(0, int(elapsed_time * 1000))
            
            self.simulation_after_id = self.root.after(delay, self.simulate)
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            traceback.print_exc()

    def update_mouse_over(self, event=None):
        try:
            if event is None:
                x, y = self.last_mouse_x, self.last_mouse_y
            else:
                x, y = event.x, event.y
                self.last_mouse_x, self.last_mouse_y = x, y

            if 0 <= x < self.map_width and 0 <= y < self.map_height:
                # Retrieve data at mouse position
                pressure_value = self.pressure[y, x]
                temperature_value = self.temperature_celsius[y, x]
                wind_speed_value = np.sqrt(self.u[y, x] ** 2 + self.v[y, x] ** 2)
                elevation_value = self.elevation[y, x]
                latitude_value = self.latitude[y, x]
                longitude_value = self.longitude[y, x]
                grayscale_value = self.altitude[y, x]
                
                # Get precipitation value if available
                precipitation_value = 0
                if hasattr(self, 'precipitation') and self.precipitation is not None:
                    precipitation_value = self.precipitation[y, x]
                
                # Calculate wind direction (direction wind is blowing TO)
                wind_direction_rad = np.arctan2(-self.u[y, x], -self.v[y, x])
                wind_direction_deg = (90 - np.degrees(wind_direction_rad)) % 360

                # Update label text
                self.mouse_over_label.config(text=(
                    f"Pressure: {pressure_value:.2f} Pa, "
                    f"Temperature: {temperature_value:.2f} °C, "
                    f"Wind Speed: {wind_speed_value:.2f} m/s, "
                    f"Wind Direction: {wind_direction_deg:.2f}°, "
                    f"Precipitation: {precipitation_value:.2f} mm/hr\n"
                    f"Latitude: {latitude_value:.2f}°, Longitude: {longitude_value:.2f}°, "
                    f"Elevation: {elevation_value:.2f} m, "
                    f"Y: {y}, X: {x}"
                ))
        except Exception as e:
            print(f"Error in update_mouse_over: {e}")
            # Don't reraise to prevent thread crashes
            
    # [Implement the rest of the methods from the original code, adjusting them to use `self` instead of global variables.]

    def normalize_data(self, data):
        """Normalize data to range [0,1]"""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        return np.zeros_like(data)

    def calculate_laplacian(self, field):
        """Calculate the Laplacian of a field using finite difference method with optimized calculations."""
        # Convert to float32 for faster calculations with minimal precision loss
        field = field.astype(np.float32) if field.dtype != np.float32 else field
        
        # Initialize arrays if they don't exist
        if not hasattr(self, '_reusable_array') or self._reusable_array is None:
            self._reusable_array = np.zeros_like(field, dtype=np.float32)
        
        if not hasattr(self, '_reusable_array2') or self._reusable_array2 is None:
            self._reusable_array2 = np.zeros_like(field, dtype=np.float32)
        
        # Ensure arrays have correct shape
        if self._reusable_array.shape != field.shape:
            self._reusable_array = np.zeros_like(field, dtype=np.float32)
        
        if self._reusable_array2.shape != field.shape:
            self._reusable_array2 = np.zeros_like(field, dtype=np.float32)
        
        # Calculate laplacians using numpy's optimized functions
        np.add(np.roll(field, -1, axis=1), np.roll(field, 1, axis=1), out=self._reusable_array)
        np.subtract(self._reusable_array, 2 * field, out=self._reusable_array)
        np.divide(self._reusable_array, self.grid_spacing_x ** 2, out=self._reusable_array)
        
        np.add(np.roll(field, -1, axis=0), np.roll(field, 1, axis=0), out=self._reusable_array2)
        np.subtract(self._reusable_array2, 2 * field, out=self._reusable_array2)
        np.divide(self._reusable_array2, self.grid_spacing_y ** 2, out=self._reusable_array2)
        
        # Add the two components
        np.add(self._reusable_array, self._reusable_array2, out=self._reusable_array)
        
        return self._reusable_array


    def initialize_ocean_currents(self):
        """Initialize ocean current components"""
        # First make sure the arrays exist
        if not hasattr(self, 'ocean_u') or self.ocean_u is None:
            self.ocean_u = np.zeros((self.map_height, self.map_width))
        if not hasattr(self, 'ocean_v') or self.ocean_v is None:
            self.ocean_v = np.zeros((self.map_height, self.map_width))
        
        is_ocean = self.elevation <= 0
        
        # Calculate Coriolis parameter
        f = 2 * self.Omega * np.sin(np.deg2rad(self.latitude))
        
        # Set up major ocean currents
        for lat in range(self.map_height):
            lat_deg = self.latitude[lat, 0]
            
            # Northern Hemisphere currents
            if lat_deg > 45 and lat_deg < 65:
                self.ocean_u[lat, :] = 0.4
                self.ocean_v[lat, :] = -0.2 * np.sin(np.linspace(0, 2*np.pi, self.map_width))
            elif lat_deg > 25 and lat_deg < 45:
                self.ocean_u[lat, :] = -0.3
                self.ocean_v[lat, :] = 0.2 * np.sin(np.linspace(0, 2*np.pi, self.map_width))
            elif lat_deg > 5 and lat_deg < 25:
                self.ocean_u[lat, :] = -0.4
            elif lat_deg > -5 and lat_deg < 5:
                self.ocean_u[lat, :] = 0.3
            
            # Southern Hemisphere currents
            elif lat_deg < -45 and lat_deg > -65:
                self.ocean_u[lat, :] = 0.6
                self.ocean_v[lat, :] = 0.2 * np.sin(np.linspace(0, 2*np.pi, self.map_width))
            elif lat_deg < -25 and lat_deg > -45:
                self.ocean_u[lat, :] = -0.3
                self.ocean_v[lat, :] = -0.2 * np.sin(np.linspace(0, 2*np.pi, self.map_width))
            elif lat_deg < -5 and lat_deg > -25:
                self.ocean_u[lat, :] = -0.4
        
        # Apply land mask - no currents on land
        self.ocean_u[~is_ocean] = 0
        self.ocean_v[~is_ocean] = 0

    def update_ocean_temperature(self):
        """Delegate to temperature module"""
        self.temperature.update_ocean()

    def update_temperature_land_ocean(self):
        """Delegate to temperature module"""
        self.temperature.update_land_ocean()

    def update_temperature_land_ocean_old(self):
        """
        Update temperature fields with improved greenhouse effect,
        including cloud effects and humidity coupling.
        Optimized for performance with vectorized operations and reduced memory allocations.
        """
        try:
            # Initialize arrays if they don't exist
            if not hasattr(self, '_base_albedo') or self._base_albedo is None:
                self._base_albedo = np.zeros_like(self.elevation, dtype=np.float32)
                
            if not hasattr(self, '_solar_in') or self._solar_in is None:
                self._solar_in = np.zeros_like(self.temperature_celsius, dtype=np.float32)
                
            if not hasattr(self, '_longwave_out') or self._longwave_out is None:
                self._longwave_out = np.zeros_like(self.temperature_celsius, dtype=np.float32)
                
            if not hasattr(self, '_net_flux') or self._net_flux is None:
                self._net_flux = np.zeros_like(self.temperature_celsius, dtype=np.float32)
                
            if not hasattr(self, '_heat_capacity') or self._heat_capacity is None:
                self._heat_capacity = np.zeros_like(self.elevation, dtype=np.float32)
                
            if not hasattr(self, '_delta_T') or self._delta_T is None:
                self._delta_T = np.zeros_like(self.temperature_celsius, dtype=np.float32)
            
            params = self.climate_params
            
            # Type conversion for better performance
            if self.temperature_celsius.dtype != np.float32:
                self.temperature_celsius = self.temperature_celsius.astype(np.float32)
            
            # --- BASIC LAND-OCEAN MASKS ---
            is_land = self.elevation > 0
            is_ocean = ~is_land
                
            # --- SOLAR INPUT CALCULATION ---
            # Constants - use single precision
            S0 = np.float32(params['solar_constant'])  # Solar constant
            
            # Calculate solar zenith angle effect (vectorized)
            cos_phi = np.cos(self.latitudes_rad).astype(np.float32)
            day_length_factor = np.clip(cos_phi, 0, 1)  # Day/night cycle
            
            # Calculate average insolation (vectorized)
            S_avg = S0 / 4  # Spherical geometry factor
            S_lat = S_avg * day_length_factor
            
            # --- ALBEDO CALCULATION ---
            # Surface type masks
            is_land = self.elevation > 0
            
            # Pre-allocate base albedo array to avoid new allocation
            if not hasattr(self, '_base_albedo') or self._base_albedo.shape != is_land.shape:
                self._base_albedo = np.zeros_like(is_land, dtype=np.float32)
            
            # Apply land/ocean albedo (vectorized)
            np.place(self._base_albedo, is_land, params['albedo_land'])
            np.place(self._base_albedo, ~is_land, params['albedo_ocean'])
            
            # Zenith angle dependent albedo (vectorized)
            zenith_factor = np.clip(1.0 / (cos_phi + 0.1), 1.0, 2.0)
            
            # --- CLOUD EFFECTS ---
            if hasattr(self, 'cloud_cover'):
                # Cloud albedo calculation (vectorized)
                cloud_albedo = np.float32(0.6)
                effective_albedo = (1 - self.cloud_cover) * self._base_albedo * zenith_factor + self.cloud_cover * cloud_albedo
            else:
                effective_albedo = np.clip(self._base_albedo * zenith_factor, 0.0, 0.9)
            
            # --- ABSORBED SOLAR RADIATION ---
            # Reuse array if possible
            if not hasattr(self, '_solar_in') or self._solar_in.shape != S_lat.shape:
                self._solar_in = np.zeros_like(S_lat, dtype=np.float32)
            
            # Calculate absorbed radiation (vectorized)
            np.multiply(S_lat, 1 - effective_albedo, out=self._solar_in)
            self.energy_budget['solar_in'] = float(np.mean(self._solar_in))
            
            # --- GREENHOUSE EFFECT ---
            greenhouse_forcing = self.calculate_greenhouse_effect(params)
            self.energy_budget['greenhouse_effect'] = greenhouse_forcing
            
            # Add cloud greenhouse effect (vectorized)
            if hasattr(self, 'cloud_cover'):
                day_night_factor = 1.0 + (1.0 - day_length_factor) * 1.5
                cloud_greenhouse = 35.0 * self.cloud_cover * day_night_factor
                greenhouse_forcing += cloud_greenhouse
            
            # --- OUTGOING LONGWAVE RADIATION ---
            sigma = np.float32(5.670374419e-8)  # Stefan-Boltzmann constant
            T_kelvin = self.temperature_celsius + 273.15
            
            if hasattr(self, 'cloud_cover'):
                cloud_emissivity_factor = 0.2 + 0.1 * (1.0 - day_length_factor)
                effective_emissivity = params['emissivity'] * (1 - cloud_emissivity_factor * self.cloud_cover)
            else:
                effective_emissivity = params['emissivity']
            
            night_cooling_factor = 0.85 + 0.15 * day_length_factor
            
            # Calculate longwave radiation (vectorized)
            if not hasattr(self, '_longwave_out') or self._longwave_out.shape != T_kelvin.shape:
                self._longwave_out = np.zeros_like(T_kelvin, dtype=np.float32)
            
            np.power(T_kelvin, 4, out=self._longwave_out)
            np.multiply(self._longwave_out, sigma, out=self._longwave_out)
            np.multiply(self._longwave_out, effective_emissivity, out=self._longwave_out)
            np.multiply(self._longwave_out, night_cooling_factor, out=self._longwave_out)
            
            self.energy_budget['longwave_out'] = float(np.mean(self._longwave_out))
            
            # --- NET ENERGY FLUX ---
            # Calculate net flux (vectorized)
            if not hasattr(self, '_net_flux') or self._net_flux.shape != self._solar_in.shape:
                self._net_flux = np.zeros_like(self._solar_in, dtype=np.float32)
            
            np.add(self._solar_in, greenhouse_forcing, out=self._net_flux)
            np.subtract(self._net_flux, self._longwave_out, out=self._net_flux)
            
            self.energy_budget['net_flux'] = float(np.mean(self._net_flux))
            
            # --- HEAT ADVECTION ---
            # Calculate temperature gradients
            dT_dy, dT_dx = np.gradient(self.temperature_celsius, self.grid_spacing_y, self.grid_spacing_x)
            temperature_advection = -(self.u * dT_dx + self.v * dT_dy)
            
            # --- HEAT CAPACITY ---
            # Initialize heat capacity array (reuse if possible)
            if not hasattr(self, '_heat_capacity') or self._heat_capacity.shape != is_land.shape:
                self._heat_capacity = np.zeros_like(is_land, dtype=np.float32)
            
            # Set land/ocean heat capacities (vectorized)
            np.place(self._heat_capacity, is_land, params['heat_capacity_land'])
            np.place(self._heat_capacity, ~is_land, params['heat_capacity_ocean'])
            
            # Adjust for humidity (vectorized)
            if hasattr(self, 'humidity'):
                moisture_factor = 1.0 + self.humidity * 0.5
                self._heat_capacity[is_land] *= moisture_factor[is_land]
            
            # --- TEMPERATURE CHANGE ---
            # Calculate delta T (vectorized)
            if not hasattr(self, '_delta_T') or self._delta_T.shape != self._net_flux.shape:
                self._delta_T = np.zeros_like(self._net_flux, dtype=np.float32)
            
            np.divide(self._net_flux, self._heat_capacity, out=self._delta_T)
            np.multiply(self._delta_T, self.time_step_seconds, out=self._delta_T)
            np.add(self._delta_T, temperature_advection * self.time_step_seconds, out=self._delta_T)
            
            # Apply temperature change (vectorized)
            np.add(self.temperature_celsius, self._delta_T, out=self.temperature_celsius)
            
            # --- APPLY DIFFUSION WITH SEPARATE LAND AND OCEAN TREATMENT ---
            # Treat land and ocean diffusion separately
            temperature_land = np.copy(self.temperature_celsius)
            temperature_ocean = np.copy(self.temperature_celsius)
            
            # Apply different diffusion amounts to land and ocean
            temperature_land = gaussian_filter(temperature_land, sigma=params['atmospheric_heat_transfer'], mode='wrap')
            temperature_ocean = gaussian_filter(temperature_ocean, sigma=params['atmospheric_heat_transfer']*1.5, mode='wrap')
            
            # Recombine land and ocean temperatures
            self.temperature_celsius = np.where(is_land, temperature_land, temperature_ocean).astype(np.float32)
            
            # Apply an additional lateral mixing step along coast lines
            coastal_mask = np.zeros_like(self.elevation, dtype=bool)
            
            # Define a simple kernel for finding coastal regions
            kernel = np.ones((3, 3), dtype=bool)
            kernel[1, 1] = False
            
            # Find cells that are land but adjacent to ocean
            from scipy.ndimage import binary_dilation
            coastal_land = is_land & binary_dilation(is_ocean, structure=kernel)
            
            # Find cells that are ocean but adjacent to land
            coastal_ocean = is_ocean & binary_dilation(is_land, structure=kernel)
            
            # Combine both to get all coastal regions
            coastal_mask = coastal_land | coastal_ocean
            
            # Apply a targeted smoothing only to coastal regions
            if np.any(coastal_mask):
                smoothed_coastal = gaussian_filter(self.temperature_celsius, sigma=3.0, mode='wrap')
                # Blend original with smoothed along the coast (50% blend)
                self.temperature_celsius[coastal_mask] = (0.5 * self.temperature_celsius[coastal_mask] + 
                                                         0.5 * smoothed_coastal[coastal_mask])
            
            # Apply gentle global smoothing as a final step
            self.temperature_celsius = gaussian_filter(
                self.temperature_celsius, 
                sigma=0.5  # Reduced final smoothing - just enough to remove any remaining artifacts
            ).astype(np.float32)
            
            # Track temperature history
            if 'temperature_history' not in self.energy_budget:
                self.energy_budget['temperature_history'] = []
            
            # Append current average temperature to history
            self.energy_budget['temperature_history'].append(float(np.mean(self.temperature_celsius)))
            
        except Exception as e:
            print(f"Error updating temperature: {e}")
            traceback.print_exc()


    def map_to_grayscale(self, data):
        """Convert data to grayscale values"""
        grayscale = np.clip(data, 0, 1)
        grayscale = (grayscale * 255).astype(np.uint8)
        rgb_array = np.stack((grayscale,)*3, axis=-1)
        return rgb_array


    def map_altitude_to_color(self, elevation):
        """Map elevation data to RGB colors using standard elevation palette"""
        # Create RGB array
        rgb = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        
        # Get the actual grayscale range from the image
        min_gray = self.altitude.min()  # Should be 0
        max_gray = self.altitude.max()  # Should be 510
        sea_level = 3  # First land pixel value
        
        # Create masks based on raw grayscale values
        ocean = self.altitude <= 2
        land = self.altitude > 2
        
        # Ocean (blue)
        rgb[ocean] = [0, 0, 255]
        
        # Land areas
        if np.any(land):
            land_colors = np.array([
                [4, 97, 2],      # Dark Green (lowest elevation)
                [43, 128, 42],   # Light Green
                [94, 168, 93],   # Light yellow
                [235, 227, 87],  # Yellow
                [171, 143, 67],  # Brown
                [107, 80, 0],    # Dark Brown
                [227, 227, 227]  # Gray (highest elevation)
            ])
            
            # Get grayscale values for land only
            land_grayscale = self.altitude[land]
            
            # Normalize land heights to [0,1] range
            normalized_heights = (land_grayscale - sea_level) / (max_gray - sea_level)
            
            # Calculate indices for color interpolation
            color_indices = (normalized_heights * (len(land_colors) - 1)).astype(int)
            color_indices = np.clip(color_indices, 0, len(land_colors) - 2)
            
            # Calculate interpolation ratios
            ratios = (normalized_heights * (len(land_colors) - 1)) % 1
            
            # Interpolate colors
            start_colors = land_colors[color_indices]
            end_colors = land_colors[color_indices + 1]
            
            # Apply interpolated colors to land areas
            rgb[land] = (start_colors * (1 - ratios[:, np.newaxis]) + 
                        end_colors * ratios[:, np.newaxis]).astype(np.uint8)
                 
        return rgb


    def map_temperature_to_color(self, temperature_data):
        """
        Map temperature to a blue-red color gradient with more distinct transitions.
        Blue indicates cold, red indicates hot.
        """
        # Create RGBA array
        rgba_array = np.zeros((self.map_height, self.map_width, 4), dtype=np.uint8)
        
        # Set fixed temperature range for consistent coloring
        min_temp = -50.0  # Minimum expected temperature
        max_temp = 50.0   # Maximum expected temperature
        
        # Normalize temperature to [0,1] using fixed range
        temp_normalized = np.clip((temperature_data - min_temp) / (max_temp - min_temp), 0, 1)
        
        # Create more distinct temperature gradient
        # Red component (increases with temperature)
        rgba_array[..., 0] = (np.power(temp_normalized, 1.2) * 255).astype(np.uint8)
        
        # Green component (peaks in middle temperatures)
        green = np.minimum(2 * temp_normalized, 2 * (1 - temp_normalized))
        rgba_array[..., 1] = (green * 180).astype(np.uint8)  # Reduced max green for more vibrant colors
        
        # Blue component (decreases with temperature)
        rgba_array[..., 2] = (np.power(1 - temp_normalized, 1.2) * 255).astype(np.uint8)
        
        # Set alpha channel
        rgba_array[..., 3] = 255  # Full opacity
        
        # Make water areas slightly transparent
        is_water = self.elevation <= 0
        rgba_array[is_water, 3] = 192  # 75% opacity for water
        
        return rgba_array


    def map_ocean_temperature_to_color(self, data_normalized):
        """Map ocean temperature to color gradient"""
        if not isinstance(data_normalized, np.ndarray) or data_normalized.ndim != 2:
            raise ValueError("Expected 2D numpy array for data_normalized")
            
        rgb_array = np.zeros((data_normalized.shape[0], data_normalized.shape[1], 3), dtype=np.uint8)
        is_ocean = self.elevation <= 0
        
        # Define ocean temperature color breakpoints
        breakpoints = [
            (-2, (0, 0, 128)),      # Very cold (dark blue)
            (5, (0, 0, 255)),       # Cold (blue)
            (10, (0, 128, 255)),    # Cool (light blue)
            (15, (0, 255, 255)),    # Mild (cyan)
            (20, (0, 255, 128)),    # Warm (turquoise)
            (25, (0, 255, 0)),      # Hot (green)
            (30, (255, 255, 0))     # Very hot (yellow)
        ]

        # Convert temperatures to normalized values
        min_temp = -2
        max_temp = 30
        temp_range = max_temp - min_temp
        
        for i in range(len(breakpoints) - 1):
            temp1, color1 = breakpoints[i]
            temp2, color2 = breakpoints[i + 1]
            # Convert temperatures to normalized values
            val0 = (temp1 - min_temp) / temp_range
            val1 = (temp2 - min_temp) / temp_range
            mask = is_ocean & (data_normalized >= val0) & (data_normalized < val1)
            if mask.any():
                ratio = (data_normalized[mask] - val0) / (val1 - val0)
                for c in range(3):
                    rgb_array[mask, c] = color1[c] + ratio * (color2[c] - color1[c])

        # Set land areas to black
        rgb_array[~is_ocean] = 0
        
        return rgb_array


    def update_pressure(self):
        """Delegate to pressure module"""
        self.pressure_system.update()

    def _add_pressure_system(self, field, lon_center, lat_center, radius, strength):
        """Delegate to pressure module"""
        return self.pressure_system._add_pressure_system(field, lon_center, lat_center, radius, strength)

    def draw_wind_vectors(self):
        """Draw wind vectors at specific latitudes"""
        try:
            # Define specific latitudes where we want vectors (north and south)
            target_latitudes = [80, 70, 60, 50, 40, 30, 20, 10, 0, -10, -20, -30, -40, -50, -60, -70, -80]
            
            # Base step size for longitude spacing
            x_step = self.map_width // 30
            x_indices = np.arange(0, self.map_width, x_step)
            
            # Find y-coordinates for each target latitude
            y_indices = []
            for target_lat in target_latitudes:
                # Find the y-coordinate where latitude is closest to target
                y_coord = np.abs(self.latitude[:, 0] - target_lat).argmin()
                y_indices.append(y_coord)
            
            # Create coordinate grids
            X, Y = np.meshgrid(x_indices, y_indices)
            
            # Sample wind components
            u_sampled = self.u[Y, X]
            v_sampled = self.v[Y, X]
            
            # Calculate magnitudes for scaling
            magnitudes = np.sqrt(u_sampled**2 + v_sampled**2)
            max_magnitude = np.max(magnitudes) if magnitudes.size > 0 else 1.0
            
            # Scale factor for arrow length
            scale = x_step * 0.5 / max_magnitude if max_magnitude > 0 else x_step * 0.5
            
            # Draw vectors
            for i in range(len(y_indices)):
                for j in range(len(x_indices)):
                    x = X[i, j]
                    y = Y[i, j]
                    # Invert v component to match screen coordinates (y increases downward)
                    dx = u_sampled[i, j] * scale
                    dy = -v_sampled[i, j] * scale  # Negative to match screen coordinates
                    
                    if np.isfinite(dx) and np.isfinite(dy):
                        self.canvas.create_line(
                            x, y, x + dx, y + dy,
                            arrow=tk.LAST,
                            fill='white',
                            width=2,
                            arrowshape=(8, 10, 3)
                        )
                        
        except Exception as e:
            print(f"Error drawing wind vectors: {e}")


    def draw_current_arrows(self):
        """Draw ocean current arrows"""
        spacing = 30
        scale = 10
        is_ocean = self.elevation <= 0
        
        arrow_length = 1
        for y in range(0, self.map_height, spacing):
            for x in range(0, self.map_width, spacing):
                if is_ocean[y, x]:
                    # Skip if any surrounding tiles are land
                    y_start = max(0, y-1)
                    y_end = min(self.map_height, y+2)
                    x_start = max(0, x-1)
                    x_end = min(self.map_width, x+2)
                    
                    if not np.all(is_ocean[y_start:y_end, x_start:x_end]):
                        continue
                    
                    u_val = self.ocean_u[y, x]
                    v_val = self.ocean_v[y, x]
                    
                    magnitude = np.sqrt(u_val**2 + v_val**2)
                    if magnitude > 0.1:
                        dx = arrow_length * (u_val / magnitude)
                        dy = arrow_length * (v_val / magnitude)
                        
                        x1, y1 = x, y
                        x2 = x1 + dx * scale
                        y2 = y1 + dy * scale
                        
                        angle = np.arctan2(dy, dx)
                        arrow_head_length = 3
                        arrow_head_angle = np.pi/6
                        
                        x_head1 = x2 - arrow_head_length * np.cos(angle + arrow_head_angle)
                        y_head1 = y2 - arrow_head_length * np.sin(angle + arrow_head_angle)
                        x_head2 = x2 - arrow_head_length * np.cos(angle - arrow_head_angle)
                        y_head2 = y2 - arrow_head_length * np.sin(angle - arrow_head_angle)
                        
                        self.canvas.create_line(x1, y1, x2, y2, fill='blue', width=1)
                        self.canvas.create_line(x2, y2, x_head1, y_head1, fill='blue', width=1)
                        self.canvas.create_line(x2, y2, x_head2, y_head2, fill='blue', width=1)


    def update_map(self):
        """Update the map display with improved visualization"""
        try:
            # Check if we're running in the main thread
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                print("Skipping update_map: root widget doesn't exist or isn't ready")
                return
            
            # Track if the selected layer has changed to force a full update
            if not hasattr(self, '_last_selected_layer'):
                self._last_selected_layer = None
            layer_changed = self._last_selected_layer != self.selected_layer.get()
            self._last_selected_layer = self.selected_layer.get()
            
            # Defer expensive operations when switching layers
            if layer_changed and self.selected_layer.get() == "Pressure":
                # When first switching to pressure, show a loading indicator
                self.canvas.delete("all")
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2, 
                    self.canvas.winfo_height() // 2,
                    text="Loading pressure map...",
                    fill="white",
                    font=("Arial", 14)
                )
                self.root.update_idletasks()  # Force update to show loading message
                
                # Schedule the actual pressure map update after a short delay
                self.root.after(10, self._delayed_pressure_update)
                return
            
            # Similar treatment for precipitation which can be computationally expensive  
            if layer_changed and self.selected_layer.get() == "Precipitation":
                # When first switching to precipitation, show a loading indicator
                self.canvas.delete("all")
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2, 
                    self.canvas.winfo_height() // 2,
                    text="Loading precipitation map...",
                    fill="white",
                    font=("Arial", 14)
                )
                self.root.update_idletasks()  # Force update to show loading message
                
                # Schedule the precipitation map update after a short delay
                self.root.after(10, self._delayed_precipitation_update)
                return
                
            if self.selected_layer.get() == "Pressure":
                # Always use _delayed_pressure_update for pressure map
                # to ensure consistent updates
                self._delayed_pressure_update()
                return
            
            elif self.selected_layer.get() == "Precipitation":
                # Use delayed precipitation update
                self._delayed_precipitation_update()
                return
                
            elif self.selected_layer.get() == "Elevation":
                display_data = self.map_to_grayscale(self.elevation_normalized)
            elif self.selected_layer.get() == "Altitude":
                display_data = self.map_altitude_to_color(self.elevation)
            elif self.selected_layer.get() == "Temperature":
                display_data = self.map_temperature_to_color(self.temperature_celsius)
            elif self.selected_layer.get() == "Ocean Temperature":
                # Create normalized ocean temperature data
                # Scale from -2°C to 30°C (typical ocean temperature range)
                min_ocean_temp = -2
                max_ocean_temp = 30
                normalized_temp = (self.temperature_celsius - min_ocean_temp) / (max_ocean_temp - min_ocean_temp)
                np.clip(normalized_temp, 0, 1, out=normalized_temp)  # Ensure values are between 0-1
                
                # Map ocean temperature to colors
                display_data = self.map_ocean_temperature_to_color(normalized_temp)
            elif self.selected_layer.get() == "Wind":
                # Use altitude map as background for wind vectors
                display_data = self.map_altitude_to_color(self.elevation)
                image = Image.fromarray(display_data.astype('uint8'))
                # Store the photo image as an instance variable to prevent garbage collection
                self.current_image = ImageTk.PhotoImage(image)
                
                # Update canvas with terrain background
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
                
                # Then draw wind vectors on top of it
                self.draw_wind_vectors()
                return
            elif self.selected_layer.get() == "Clouds":
                # Get cloud visualization
                display_data = self.map_clouds_to_color()
            else:
                print(f"Unknown layer: {self.selected_layer.get()}")
                return
            
            # Convert to PIL Image and create PhotoImage
            image = Image.fromarray(display_data.astype('uint8'))
            # Store the photo image as an instance variable to prevent garbage collection
            self.current_image = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
            
        except Exception as e:
            print(f"Error in update_map: {e}")
            traceback.print_exc()
    
    def _delayed_pressure_update(self):
        """Handle pressure map update in a way that doesn't block the UI"""
        try:
            # Force regeneration of pressure visualization by clearing the cache
            if hasattr(self, '_cached_pressure_viz'):
                delattr(self, '_cached_pressure_viz')
            if hasattr(self, '_cached_pressure_data'):
                delattr(self, '_cached_pressure_data')
                
            # Get combined pressure and terrain visualization
            display_data = self.map_pressure_to_color(self.pressure)
            
            # Convert to PIL Image and create PhotoImage
            image = Image.fromarray(display_data)
            self._pressure_image = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._pressure_image)
        except Exception as e:
            print(f"Error in delayed pressure update: {e}")
            traceback.print_exc()

    def initialize_humidity(self):
        """Initialize spatial humidity map."""
        # Base humidity values
        equator_humidity = 0.8  # High humidity near the equator
        pole_humidity = 0.4     # Lower humidity near the poles

        # Create a latitude-based gradient for humidity
        # Higher humidity near the equator, decreasing towards the poles
        latitude_humidity_gradient = pole_humidity + (equator_humidity - pole_humidity) * np.cos(np.deg2rad(self.latitude))

        # Enhance humidity near oceans
        is_ocean = self.elevation <= 0
        ocean_humidity = 0.7  # Higher humidity over oceans (was 0.9, lowered slightly)
        land_humidity = 0.5   # Moderate humidity over land (was 0.6, lowered slightly)

        # Initialize humidity map based on land and ocean
        self.humidity = np.where(is_ocean, ocean_humidity, land_humidity)

        # Blend with latitude gradient for more realism
        blend_factor = 0.4  # Increased influence of latitude on land humidity (was 0.3)
        self.humidity = self.humidity * (1 - blend_factor) + latitude_humidity_gradient * blend_factor

        # Add random variability for more natural distribution
        noise = np.random.normal(loc=0.0, scale=0.05, size=self.humidity.shape)
        self.humidity += noise
        self.humidity = np.clip(self.humidity, 0.4, 0.9)  # Ensure reasonable humidity range (was 0.0, 1.0)

        print("Humidity map initialized.")


    def initialize_temperature(self):
        """Delegate to temperature module"""
        self.temperature.initialize()

    def initialize_pressure(self):
        """Delegate to pressure module"""
        self.pressure_system.initialize()


    def map_pressure_to_color(self, pressure_data):
        """Convert pressure data to color visualization with less frequent isolines"""
        try:
            # Reuse cached pressure visualization if pressure data hasn't changed significantly
            if hasattr(self, '_cached_pressure_viz') and hasattr(self, '_cached_pressure_data'):
                # Only recalculate if pressure has changed significantly
                if np.array_equal(pressure_data, self._cached_pressure_data):
                    return self._cached_pressure_viz
                
                # Check if changes are minor - use a smaller threshold to detect more changes
                # Reduced threshold further to ensure more frequent updates
                elif np.abs(np.mean(pressure_data - self._cached_pressure_data)) < 0.0001 * np.mean(np.abs(self._cached_pressure_data)):
                    if hasattr(self, '_pressure_viz_skip_counter'):
                        self._pressure_viz_skip_counter += 1
                        # Only skip every other frame at most
                        if self._pressure_viz_skip_counter < 1:
                            return self._cached_pressure_viz
                        else:
                            self._pressure_viz_skip_counter = 0
                    else:
                        self._pressure_viz_skip_counter = 0
            
            # Get terrain colors first (cache if possible)
            if not hasattr(self, '_cached_terrain_colors'):
                self._cached_terrain_colors = self.map_elevation_to_color(self.elevation)
            terrain_colors = self._cached_terrain_colors
            
            # Convert pressure to hPa once
            pressure_hpa = pressure_data / 100.0
            
            # Set fixed pressure range (870-1086 hPa)
            p_min, p_max = 870, 1086
            
            # Pre-allocate the RGBA array
            rgba_array = np.zeros((*terrain_colors.shape[:2], 4), dtype=np.uint8)
            
            # Normalize pressure in one step
            normalized_pressure = 2 * (pressure_hpa - (p_min + p_max)/2) / (p_max - p_min)
            
            # Vectorized color calculations (all at once instead of channel by channel)
            pressure_colors = np.zeros((*terrain_colors.shape[:2], 4), dtype=np.uint8)
            
            # Create color lookup tables
            pressure_ranges = np.array([-1, -0.5, 0, 0.5, 1])
            r_values = np.array([65, 150, 255, 255, 255])
            g_values = np.array([200, 230, 255, 230, 200])
            b_values = np.array([255, 255, 255, 150, 65])
            
            # Vectorized interpolation for all channels at once
            pressure_colors[..., 0] = np.interp(normalized_pressure, pressure_ranges, r_values)
            pressure_colors[..., 1] = np.interp(normalized_pressure, pressure_ranges, g_values)
            pressure_colors[..., 2] = np.interp(normalized_pressure, pressure_ranges, b_values)
            pressure_colors[..., 3] = 170  # Constant opacity
            
            # Efficient alpha blending
            alpha = pressure_colors[..., 3:] / 255.0
            rgba_array[..., :3] = ((pressure_colors[..., :3] * alpha) + 
                                 (terrain_colors[..., :3] * (1 - alpha))).astype(np.uint8)
            rgba_array[..., 3] = 255
            
            # Always draw isobars - removed the skipping logic
            # Optimize isobar calculation
            if not hasattr(self, '_pressure_levels'):
                # Calculate pressure levels once and cache
                min_level = np.floor(p_min / 25) * 25
                max_level = np.ceil(p_max / 25) * 25
                self._pressure_levels = np.arange(min_level, max_level + 25, 25)
            
            # Reduce computation by using cached smoothed pressure when possible
            if not hasattr(self, '_last_smoothed_pressure_data') or not np.array_equal(pressure_data, self._last_smoothed_pressure_data):
                # Use pre-smoothed pressure field for isobars
                smoothed_pressure = gaussian_filter(pressure_hpa, sigma=1.0)
                self._last_smoothed_pressure = smoothed_pressure
                self._last_smoothed_pressure_data = pressure_data.copy()
            else:
                smoothed_pressure = self._last_smoothed_pressure
            
            # Reduce number of isobar levels for better performance - use fewer pressure levels
            reduced_pressure_levels = self._pressure_levels[::2]  # Only use every other level
            
            # Vectorized isobar calculation
            isobar_mask = np.zeros_like(pressure_hpa, dtype=bool)
            for level in reduced_pressure_levels:
                isobar_mask |= np.abs(smoothed_pressure - level) < 0.5
            
            # Apply smoothing to mask more efficiently
            isobar_mask = gaussian_filter(isobar_mask.astype(float), sigma=0.5) > 0.3
            
            # Apply isobars efficiently
            rgba_array[isobar_mask, :3] = 255  # White lines
            rgba_array[isobar_mask, 3] = 180  # Slightly transparent
            
            # Cache the result
            self._cached_pressure_viz = rgba_array
            self._cached_pressure_data = pressure_data.copy()
            
            return rgba_array
                
        except Exception as e:
            print(f"Error in map_pressure_to_color: {e}")
            if hasattr(self, '_cached_pressure_viz'):
                return self._cached_pressure_viz
            return np.zeros((*terrain_colors.shape[:2], 4), dtype=np.uint8)


    def update_visualization_loop(self):
        """Continuous loop for updating visualization in separate thread"""
        try:
            while self.visualization_active.is_set():
                # Just schedule the update in the main thread, don't do any GUI operations here
                try:
                    self.root.after(0, self.update_visualization)
                except Exception as e:
                    print(f"Error scheduling visualization update: {e}")
                time.sleep(0.033)  # ~30 FPS, adjust as needed
        except Exception as e:
            print(f"Error in visualization loop: {e}")
            traceback.print_exc()

    def update_visualization(self):
        """Update the map display with current data"""
        try:
            # This method is now called in the main thread via root.after
            
            # Limit update frequency
            if not hasattr(self, '_last_update_time'):
                self._last_update_time = time.time()
            
            current_time = time.time()
            if current_time - self._last_update_time < 0.033:  # ~30 FPS max
                return
                
            self._last_update_time = current_time
            
            # Directly call the visualization update since we're already in the main thread
            self._update_visualization_safe()
        except Exception as e:
            print(f"Error in visualization update: {e}")
            traceback.print_exc()

    def _update_visualization_safe(self):
        """Perform actual visualization update in main thread"""
        try:
            # Clear previous layers
            self.canvas.delete("data_layer")
            
            # If the selected layer is Pressure, update the pressure map
            if self.selected_layer.get() == "Pressure":
                # Don't use update_map for pressure, handle it directly here
                # to ensure consistent updates
                colors = self.map_pressure_to_color(self.pressure)
                
                # Save the resulting image for potential reuse
                image = Image.fromarray(colors)
                self.current_photo = ImageTk.PhotoImage(image)
                self._pressure_image = self.current_photo  # Keep reference for update_map
                
                # Update canvas
                self.canvas.create_image(
                    0, 0, 
                    anchor=tk.NW, 
                    image=self.current_photo,
                    tags="data_layer"
                )
            elif self.selected_layer.get() == "Elevation":
                # Handle elevation layer
                colors = self.map_to_grayscale(self.elevation_normalized)
                
                # Convert to PhotoImage format
                image = Image.fromarray(colors)
                self.current_photo = ImageTk.PhotoImage(image)
                
                # Update canvas
                self.canvas.create_image(
                    0, 0, 
                    anchor=tk.NW, 
                    image=self.current_photo,
                    tags="data_layer"
                )
            elif self.selected_layer.get() == "Altitude":
                # Handle altitude layer
                colors = self.map_altitude_to_color(self.elevation)
                
                # Convert to PhotoImage format
                image = Image.fromarray(colors)
                self.current_photo = ImageTk.PhotoImage(image)
                
                # Update canvas
                self.canvas.create_image(
                    0, 0, 
                    anchor=tk.NW, 
                    image=self.current_photo,
                    tags="data_layer"
                )
            elif self.selected_layer.get() == "Precipitation":
                # Handle precipitation layer efficiently
                try:
                    # Only update precipitation visualization if needed
                    if not hasattr(self, '_cached_precip_image') or self._cached_precip_image is None:
                        # Generate fresh precipitation visualization
                        colors = self.map_precipitation_to_color(self.precipitation)
                        
                        # Save the resulting image for potential reuse
                        image = Image.fromarray(colors)
                        self.current_photo = ImageTk.PhotoImage(image)
                        self._cached_precip_image = self.current_photo  # Keep reference
                    else:
                        # Use cached image
                        self.current_photo = self._cached_precip_image
                    
                    # Update canvas
                    self.canvas.create_image(
                        0, 0, 
                        anchor=tk.NW, 
                        image=self.current_photo,
                        tags="data_layer"
                    )
                except Exception as e:
                    print(f"Error updating precipitation visualization: {e}")
                    traceback.print_exc()
                    # Fallback to something simple
                    self.canvas.create_text(
                        self.canvas.winfo_width() // 2,
                        self.canvas.winfo_height() // 2,
                        text="Precipitation visualization error",
                        fill="white",
                        tags="data_layer"
                    )
            elif self.selected_layer.get() == "Temperature":
                colors = self.map_temperature_to_color(self.temperature_celsius)
                
                # Convert to PhotoImage format
                image = Image.fromarray(colors)
                self.current_photo = ImageTk.PhotoImage(image)
                
                # Update canvas
                self.canvas.create_image(
                    0, 0, 
                    anchor=tk.NW, 
                    image=self.current_photo,
                    tags="data_layer"
                )
            elif self.selected_layer.get() == "Ocean Temperature":
                # Create normalized ocean temperature data
                # Scale from -2°C to 30°C (typical ocean temperature range)
                min_ocean_temp = -2
                max_ocean_temp = 30
                normalized_temp = (self.temperature_celsius - min_ocean_temp) / (max_ocean_temp - min_ocean_temp)
                np.clip(normalized_temp, 0, 1, out=normalized_temp)  # Ensure values are between 0-1
                
                # Map ocean temperature to colors
                colors = self.map_ocean_temperature_to_color(normalized_temp)
                
                # Convert to PhotoImage format
                image = Image.fromarray(colors)
                self.current_photo = ImageTk.PhotoImage(image)
                
                # Update canvas
                self.canvas.create_image(
                    0, 0, 
                    anchor=tk.NW, 
                    image=self.current_photo,
                    tags="data_layer"
                )
            elif self.selected_layer.get() == "Wind":
                # Use altitude map as background for wind vectors
                display_data = self.map_altitude_to_color(self.elevation)
                image = Image.fromarray(display_data.astype('uint8'))
                # Store the photo image as an instance variable to prevent garbage collection
                self.current_image = ImageTk.PhotoImage(image)
                
                # Update canvas with terrain background
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
                
                # Then draw wind vectors on top of it
                self.draw_wind_vectors()
                return
            elif self.selected_layer.get() == "Clouds":
                # Get cloud visualization
                display_data = self.map_clouds_to_color()
            else:
                print(f"Unknown layer: {self.selected_layer.get()}")
                return
            
        except Exception as e:
            print(f"Error in _update_visualization_safe: {e}")
            traceback.print_exc()

    def cleanup(self):
        """Clean up threads before closing"""
        self.visualization_active.clear()  # Signal threads to stop
        if hasattr(self, 'visualization_thread'):
            self.visualization_thread.join(timeout=1.0)

    def calculate_greenhouse_effect(self, params):
        """Delegate to temperature module"""
        return self.temperature.calculate_greenhouse_effect(params)

    def calculate_water_vapor_saturation(self, T):
        """Delegate to temperature module"""
        return self.temperature.calculate_water_vapor_saturation(T)

    def calculate_relative_humidity(self, vapor_pressure, T):
        """Delegate to temperature module"""
        return self.temperature.calculate_relative_humidity(vapor_pressure, T)

    def initialize_global_circulation(self):
        """Delegate to wind module"""
        self.wind_system.initialize_global_circulation()
        
    def calculate_wind(self):
        """Delegate to wind module"""
        self.wind_system.calculate()
        
    def update_wind(self):
        """Delegate to wind module"""
        self.wind_system.update()
        
    def calculate_wind_direction(self, u, v):
        """Delegate to wind module"""
        return self.wind_system.calculate_direction(u, v)

    def update_zoom_view(self, event):
        """Update zoom dialog with magnified view around mouse cursor"""
        try:
            # Get current mouse position
            x, y = event.x, event.y
            
            # Calculate zoom view boundaries
            half_view = self.zoom_dialog.view_size // 2
            x_start = max(0, x - half_view)
            y_start = max(0, y - half_view)
            x_end = min(self.map_width, x + half_view + 1)
            y_end = min(self.map_height, y + half_view + 1)
            
            # Get current layer data
            if self.selected_layer.get() == "Elevation":
                view_data = self.map_to_grayscale(self.elevation_normalized)[y_start:y_end, x_start:x_end]
            elif self.selected_layer.get() == "Altitude":
                view_data = self.map_altitude_to_color(self.elevation)[y_start:y_end, x_start:x_end]
            elif self.selected_layer.get() == "Temperature":
                view_data = self.map_temperature_to_color(self.temperature_celsius)[y_start:y_end, x_start:x_end]
            elif self.selected_layer.get() == "Pressure":
                view_data = self.map_pressure_to_color(self.pressure)[y_start:y_end, x_start:x_end]
            elif self.selected_layer.get() == "Wind":
                view_data = self.map_altitude_to_color(self.elevation)[y_start:y_end, x_start:x_end]
            elif self.selected_layer.get() == "Ocean Temperature":
                is_ocean = self.elevation <= 0
                ocean_temp = np.copy(self.temperature_celsius)
                ocean_temp[~is_ocean] = np.nan
                ocean_temp_normalized = np.zeros_like(ocean_temp)
                ocean_mask = ~np.isnan(ocean_temp)
                if ocean_mask.any():
                    ocean_temp_normalized[ocean_mask] = self.normalize_data(ocean_temp[ocean_mask])
                view_data = self.map_ocean_temperature_to_color(ocean_temp_normalized)[y_start:y_end, x_start:x_end]
            elif self.selected_layer.get() == "Precipitation":
                view_data = self.map_precipitation_to_color(self.precipitation)[y_start:y_end, x_start:x_end]
            else:
                # Default to altitude map
                print(f"Unknown Layer: {self.selected_layer.get()}")
                view_data = self.map_altitude_to_color(self.elevation)[y_start:y_end, x_start:x_end]
            
            # Create zoomed image
            img = Image.fromarray(view_data)
            img = img.resize((self.zoom_dialog.view_size * self.zoom_dialog.zoom_factor,) * 2, 
                            Image.Resampling.NEAREST)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            self.zoom_dialog.canvas.delete("image")  # Only delete image, keep crosshair
            self.zoom_dialog.canvas.create_image(0, 0, anchor="nw", image=photo, tags="image")
            self.zoom_dialog.canvas.tag_raise('crosshair')  # Ensure crosshair stays on top
            self.zoom_dialog.image = photo  # Keep reference
            
            # Position dialog near cursor but not under it
            dialog_x = self.root.winfo_rootx() + event.x + 20
            dialog_y = self.root.winfo_rooty() + event.y + 20
            
            # Ensure dialog stays within screen bounds
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            dialog_width = self.zoom_dialog.view_size * self.zoom_dialog.zoom_factor
            dialog_height = dialog_width
            
            if dialog_x + dialog_width > screen_width:
                dialog_x = event.x - dialog_width - 20
            if dialog_y + dialog_height > screen_height:
                dialog_y = event.y - dialog_height - 20
            
            self.zoom_dialog.geometry(f"+{dialog_x}+{dialog_y}")
            
            # Update mouse over info
            self.update_mouse_over(event)
            
        except Exception as e:
            print(f"Error updating zoom view: {e}")
            print(f"Current layer: {self.selected_layer.get()}")


    def print_system_stats(self):
        """Print system statistics with live updates on the same lines"""
        try:
            # Check if stats should be printed
            if not hasattr(self, 'print_stats_enabled') or not self.print_stats_enabled:
                return
                
            # Calculate cycle time
            current_time = time.time()
            if not hasattr(self, 'last_cycle_time'):
                self.last_cycle_time = current_time
            cycle_time = max(current_time - self.last_cycle_time, 0.000001)  # Prevent division by zero
            self.last_cycle_time = current_time
            
            # Get average/min/max values
            avg_temp = np.mean(self.temperature_celsius)
            min_temp = np.min(self.temperature_celsius)
            max_temp = np.max(self.temperature_celsius)
            
            avg_pressure = np.mean(self.pressure)
            min_pressure = np.min(self.pressure)
            max_pressure = np.max(self.pressure)
            
            wind_speed = np.sqrt(self.u**2 + self.v**2)
            min_wind = np.min(wind_speed)
            avg_wind = np.mean(wind_speed)
            max_wind = np.max(wind_speed)
            
            # Get energy budget values
            solar_in = self.energy_budget.get('solar_in', 0)
            greenhouse = self.energy_budget.get('greenhouse_effect', 0)
            longwave_out = self.energy_budget.get('longwave_out', 0)
            net_flux = self.energy_budget.get('net_flux', 0)
            
            # Calculate net changes
            if not hasattr(self, 'last_values'):
                self.last_values = {
                    'temp': avg_temp,
                    'pressure': avg_pressure,
                    'solar': solar_in,
                    'greenhouse': greenhouse,
                    'net_flux': net_flux,
                    'wind': avg_wind
                }
            
            # Calculate all deltas
            temp_change = avg_temp - self.last_values['temp']
            pressure_change = avg_pressure - self.last_values['pressure']
            solar_change = solar_in - self.last_values['solar']
            greenhouse_change = greenhouse - self.last_values['greenhouse']
            net_flux_change = net_flux - self.last_values['net_flux']
            wind_change = avg_wind - self.last_values['wind']

            # Update last values
            self.last_values.update({
                'temp': avg_temp,
                'pressure': avg_pressure,
                'solar': solar_in,
                'greenhouse': greenhouse,
                'net_flux': net_flux,
                'wind': avg_wind
            })
            
            # --- NEW DIAGNOSTIC METRICS ---
            
            # 1. Humidity and Cloud Metrics (if available)
            humidity_stats = ""
            cloud_stats = ""
            
            if hasattr(self, 'humidity'):
                avg_humidity = np.mean(self.humidity) * 100  # Convert to percentage
                min_humidity = np.min(self.humidity) * 100
                max_humidity = np.max(self.humidity) * 100
                
                # Store and calculate change
                if 'humidity' not in self.last_values:
                    self.last_values['humidity'] = avg_humidity
                humidity_change = avg_humidity - self.last_values['humidity']
                self.last_values['humidity'] = avg_humidity
                
                humidity_stats = f"Humidity          | Avg: {avg_humidity:6.1f}% | Min: {min_humidity:6.1f}% | Max: {max_humidity:6.1f}% | Δ: {humidity_change:+6.2f}%"
            
            if hasattr(self, 'cloud_cover'):
                avg_cloud = np.mean(self.cloud_cover) * 100  # Convert to percentage
                max_cloud = np.max(self.cloud_cover) * 100
                cloud_area = np.sum(self.cloud_cover > 0.1) / self.cloud_cover.size * 100  # % area with clouds
                
                # Store and calculate change
                if not hasattr(self.last_values, 'cloud'):
                    self.last_values['cloud'] = avg_cloud
                cloud_change = avg_cloud - self.last_values['cloud']
                self.last_values['cloud'] = avg_cloud
                
                cloud_stats = f"Cloud Cover       | Avg: {avg_cloud:6.1f}% | Max: {max_cloud:6.1f}% | Area: {cloud_area:6.1f}% | Δ: {cloud_change:+6.2f}%"
            
            # 2. Land vs Ocean Temperature Difference (important for circulation)
            is_land = self.elevation > 0
            is_ocean = ~is_land
            
            if np.any(is_land) and np.any(is_ocean):
                land_temp = np.mean(self.temperature_celsius[is_land])
                ocean_temp = np.mean(self.temperature_celsius[is_ocean])
                temp_diff = land_temp - ocean_temp
                
                # Store and calculate change in differential
                if not hasattr(self.last_values, 'temp_diff'):
                    self.last_values['temp_diff'] = temp_diff
                diff_change = temp_diff - self.last_values['temp_diff']
                self.last_values['temp_diff'] = temp_diff
                
                land_ocean_stats = f"Land-Ocean Temp   | Land: {land_temp:6.1f}°C | Ocean: {ocean_temp:6.1f}°C | Diff: {temp_diff:+6.1f}°C | Δ: {diff_change:+6.2f}°C"
            else:
                land_ocean_stats = "Land-Ocean Temp   | No valid data"
            
            # 3. Wind patterns - directional bias and correlation with pressure gradients
            # Get predominant wind direction
            u_mean = np.mean(self.u)
            v_mean = np.mean(self.v)
            mean_direction = self.calculate_wind_direction(u_mean, v_mean)
            
            # Calculate wind vorticity (curl) - indicator of cyclonic/anticyclonic behavior
            dy = self.grid_spacing_y
            dx = self.grid_spacing_x
            dvdx = np.gradient(self.v, axis=1) / dx
            dudy = np.gradient(self.u, axis=0) / dy
            vorticity = dvdx - dudy
            mean_vorticity = np.mean(vorticity)
            
            # Store and calculate change
            if not hasattr(self.last_values, 'vorticity'):
                self.last_values['vorticity'] = mean_vorticity
            vorticity_change = mean_vorticity - self.last_values['vorticity']
            self.last_values['vorticity'] = mean_vorticity
            
            wind_pattern_stats = f"Wind Patterns     | Dir: {mean_direction:5.1f}° | Vorticity: {mean_vorticity:+7.2e} | Δ: {vorticity_change:+7.2e}"
            
            # 4. Energy balance check (incoming vs outgoing)
            energy_in = solar_in + greenhouse
            energy_out = self.energy_budget.get('longwave_out', 0)
            energy_imbalance = energy_in - energy_out
            imbalance_percent = np.abs(energy_imbalance) / energy_in * 100 if energy_in > 0 else 0

            # Calculate day/night effect for context
            cos_phi = np.cos(self.latitudes_rad)
            day_factor = np.clip(np.mean(cos_phi), 0, 1)  # 0=full night, 1=full day

            # Only show warning if imbalance is high AND it's not due to normal day/night cycle
            # Higher threshold at night (15% vs 10% during day)
            night_threshold = 15.0  # Higher threshold at night when imbalance is expected
            day_threshold = 10.0    # Lower threshold during day

            # Dynamic threshold based on day/night cycle
            threshold = night_threshold * (1.0 - day_factor) + day_threshold * day_factor

            warning = "⚠️ " if imbalance_percent > threshold else ""

            energy_balance = f"Energy Balance    | In: {energy_in:6.1f} W/m² | Out: {energy_out:6.1f} W/m² | Imbalance: {energy_imbalance:+6.1f} W/m² | {warning}{imbalance_percent:4.1f}%"
            
            # 5. Physical Stability Indicators
            temp_variability = np.std(self.temperature_celsius)
            pressure_variability = np.std(self.pressure) / avg_pressure * 100  # Percentage variability
            
            # Calculate rate of change - looking for explosive instabilities
            temp_rate = abs(temp_change / cycle_time) if cycle_time > 0 else 0
            pressure_rate = abs(pressure_change / cycle_time) if cycle_time > 0 else 0
            
            # Warning flags
            temp_warning = "⚠️ " if temp_rate > 0.1 else ""  # Warning if temp changing >0.1°C per second
            pressure_warning = "⚠️ " if pressure_rate > 10 else ""  # Warning if pressure changing >10 Pa per second
            
            stability_stats = f"Stability Check   | Temp Δ/s: {temp_warning}{temp_rate:5.3f}°C | Pres Δ/s: {pressure_warning}{pressure_rate:5.1f}Pa | T-var: {temp_variability:5.2f}°C | P-var: {pressure_variability:5.2f}%"
            
            # 6. Hemisphere asymmetry - sanity check for a realistic climate
            northern_temp = np.mean(self.temperature_celsius[self.latitude > 0])
            southern_temp = np.mean(self.temperature_celsius[self.latitude < 0])
            hemisphere_diff = northern_temp - southern_temp
            
            # Store and calculate change
            if not hasattr(self.last_values, 'hemisphere_diff'):
                self.last_values['hemisphere_diff'] = hemisphere_diff
            hemi_change = hemisphere_diff - self.last_values['hemisphere_diff']
            self.last_values['hemisphere_diff'] = hemisphere_diff
            
            hemisphere_stats = f"Hemisphere Check  | North: {northern_temp:6.1f}°C | South: {southern_temp:6.1f}°C | Diff: {hemisphere_diff:+6.1f}°C | Δ: {hemi_change:+6.2f}°C"
            
            # Calculate FPS with safety check
            fps = min(1/cycle_time, 999.9)  # Cap FPS display at 999.9
            
            # Create the output strings with fixed width
            cycle_str = f"Simulation Cycle: {self.time_step:6d} | Cycle Time: {cycle_time:6.3f}s | FPS: {fps:5.1f}"
            temp_str = f"Temperature (°C)   | Avg: {avg_temp:6.1f}  | Min: {min_temp:6.1f} | Max: {max_temp:6.1f} | Δ: {temp_change:+6.2f}"
            pres_str = f"Pressure (Pa)      | Avg: {avg_pressure:8.0f} | Min: {min_pressure:8.0f} | Max: {max_pressure:8.0f} | Δ: {pressure_change:+6.0f}"
            wind_str = f"Wind Speed (m/s)   | Avg: {avg_wind:6.1f}  | Min: {min_wind:6.1f} | Max: {max_wind:6.1f} | Δ: {wind_change:+6.2f}"
            
            # Energy budget strings with net changes
            energy_in_str = f"Energy In (W/m²)   | Solar: {solar_in:6.1f} | Δ: {solar_change:+6.2f} | Greenhouse: {greenhouse:6.1f} | Δ: {greenhouse_change:+6.2f}"
            
            # Calculate the number of lines to print (base + new metrics)
            base_lines = 6  # Original number of lines
            extra_lines = 0
            
            extra_metrics = []
            if humidity_stats:
                extra_metrics.append(humidity_stats)
                extra_lines += 1
            if cloud_stats:
                extra_metrics.append(cloud_stats)
                extra_lines += 1
            
            # Always add these metrics
            extra_metrics.extend([
                land_ocean_stats,
                wind_pattern_stats,
                energy_balance,
                stability_stats,
                
                # 1. DIURNAL CYCLE TRACKING
                # Calculate day/night phase more precisely
                f"Day-Night Status   | Phase: {'Day' if day_factor > 0.5 else 'Night'} | Sol. Factor: {day_factor:0.2f} | Cloud Bias: {np.mean(self.cloud_cover) * 100 - (self.cloud_cover * (0.7 - day_factor)).mean() * 100:+.1f}%",
                
                # 2. ENERGY FLUX BREAKDOWN
                # Calculate percentage contribution of each energy component
                f"Energy Components  | Solar: {solar_in/max(energy_in, 1e-5)*100:4.1f}% | GHG: {greenhouse/max(energy_in, 1e-5)*100:4.1f}% | Cloud: {self.energy_budget.get('cloud_effect', 0)/max(energy_in, 1e-5)*100:4.1f}% | Imbalance: {energy_imbalance/max(energy_in, 1e-5)*100:+4.1f}%",
                
                # 3. HUMIDITY TRANSPORT METRICS
                # Add moisture budget if humidity available
                f"Moisture Budget   | Evap: {self.energy_budget.get('evaporation', 0):5.2f} | Precip: {self.energy_budget.get('precipitation', 0):5.2f} | Transport: {self.energy_budget.get('humidity_transport', 0):+5.2f} | Balance: {(self.energy_budget.get('evaporation', 0) - self.energy_budget.get('precipitation', 0)):+5.2f}" if hasattr(self, 'humidity') else "Moisture Budget   | No humidity data available",
                
                # 4. OCEAN-ATMOSPHERE HEAT EXCHANGE
                # Track ocean heat content and exchange
                f"Ocean Heat        | Flux: {self.energy_budget.get('ocean_flux', 0):+6.1f} W/m² | Storage: {self.energy_budget.get('ocean_heat_content', 0):.1e} J/m² | Change: {self.energy_budget.get('ocean_heat_change', 0):+5.2f}%" if hasattr(self, 'ocean_data_available') and self.ocean_data_available else "Ocean Heat        | No ocean data available",
                
                hemisphere_stats
            ])
            extra_lines += 9  # Added 4 new diagnostic lines
            
            total_lines = base_lines + extra_lines
            
            # Only print these lines once at the start
            if self.time_step == 1:
                print("\n" * total_lines)  # Create blank lines
                print(f"\033[{total_lines}A", end='')  # Move cursor up
            
            # Update all lines in place
            print(f"\033[K{cycle_str}", end='\r\n')      # Clear line and move to next
            print(f"\033[K{temp_str}", end='\r\n')       # Clear line and move to next
            print(f"\033[K{pres_str}", end='\r\n')       # Clear line and move to next
            print(f"\033[K{wind_str}", end='\r\n')       # Clear line and move to next
            print(f"\033[K{energy_in_str}", end='\r\n')  # Clear line and move to next
            
            # Print extra metrics
            for metric in extra_metrics:
                print(f"\033[K{metric}", end='\r\n')     # Clear line and move to next
            
            # Last line doesn't need newline
            print(f"\033[K{energy_balance}", end='\r')
            
            # Move cursor back up to the start position
            print(f"\033[{total_lines}A", end='')
        except Exception as e:
            print(f"Error in print_system_stats: {e}")
            traceback.print_exc()


    def calculate_ocean_co2_absorption(self):
        """
        Calculate ocean CO2 absorption based on temperature.
        Returns a modifier for greenhouse effect.
        """
        # Identify ocean cells
        is_ocean = self.elevation <= 0
        
        if not np.any(is_ocean):
            return 0.0
        
        # Get ocean temperatures
        ocean_temps = self.temperature_celsius[is_ocean]
        
        # CO2 solubility decreases with temperature
        # Maximum absorption around 4°C, decreasing above and below
        # Using a simplified solubility curve
        optimal_temp = 4.0
        temp_diff = np.abs(ocean_temps - optimal_temp)
        
        # Calculate absorption factor (1.0 at optimal temp, decreasing as temp differs)
        absorption_factor = np.exp(-temp_diff / 10.0)  # exponential decay with temperature difference
        
        # Calculate mean absorption across all ocean cells
        mean_absorption = np.mean(absorption_factor)
        
        # Scale the greenhouse effect modification
        # At optimal absorption (1.0), reduce greenhouse effect by up to 25%
        greenhouse_modifier = -0.25 * mean_absorption
        
        return greenhouse_modifier


    def map_elevation_to_color(self, elevation_data):
        """Convert elevation data to color visualization"""
        try:
            # Create RGB array
            rgb_array = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
            
            # Normalize elevation data
            normalized_elevation = (elevation_data - np.min(elevation_data)) / (np.max(elevation_data) - np.min(elevation_data))
            
            # Create terrain colors
            # Below sea level (blues)
            ocean_mask = elevation_data <= 0
            rgb_array[ocean_mask, 0] = (50 * normalized_elevation[ocean_mask]).astype(np.uint8)  # R
            rgb_array[ocean_mask, 1] = (100 * normalized_elevation[ocean_mask]).astype(np.uint8)  # G
            rgb_array[ocean_mask, 2] = (150 + 105 * normalized_elevation[ocean_mask]).astype(np.uint8)  # B
            
            # Above sea level (greens and browns)
            land_mask = elevation_data > 0
            normalized_land = normalized_elevation[land_mask]
            rgb_array[land_mask, 0] = (100 + 155 * normalized_land).astype(np.uint8)  # R
            rgb_array[land_mask, 1] = (100 + 100 * normalized_land).astype(np.uint8)  # G
            rgb_array[land_mask, 2] = (50 + 50 * normalized_land).astype(np.uint8)   # B
            
            return rgb_array
            
        except Exception as e:
            print(f"Error in elevation mapping: {e}")
            return np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)

    def calculate_cloud_cover(self):
        """Calculate cloud cover based on humidity and temperature"""
        try:
            # Temperature in Celsius (ensure it's float32)
            T = self.temperature_celsius
            
            # Initialize arrays if they don't exist
            if not hasattr(self, '_relative_humidity') or self._relative_humidity is None:
                self._relative_humidity = np.zeros_like(self.temperature_celsius, dtype=np.float32)
            
            if not hasattr(self, '_cloud_coverage') or self._cloud_coverage is None:
                self._cloud_coverage = np.zeros_like(self.temperature_celsius, dtype=np.float32)
            
            # Constants for cloud formation thresholds (single precision)
            rh_threshold = np.float32(0.7)  # Relative humidity threshold for cloud formation
            rh_max = np.float32(1.0)        # Maximum relative humidity
            
            # Use the cached saturation vapor pressure calculation
            saturation_vapor_pressure = self.calculate_water_vapor_saturation(T)
            
            # Calculate actual vapor pressure (vectorized)
            actual_vapor_pressure = self.humidity * saturation_vapor_pressure
            
            # Calculate relative humidity (vectorized with in-place operations)
            # Ensure _relative_humidity is the right shape
            if self._relative_humidity.shape != saturation_vapor_pressure.shape:
                self._relative_humidity = np.zeros_like(saturation_vapor_pressure, dtype=np.float32)
                
            # Use safe division to avoid divide by zero
            np.divide(actual_vapor_pressure, saturation_vapor_pressure, out=self._relative_humidity, where=saturation_vapor_pressure!=0)
            np.clip(self._relative_humidity, 0, 1, out=self._relative_humidity)
            
            # Reuse array if possible - make sure it's not None and has correct shape
            if self._cloud_coverage is None or self._cloud_coverage.shape != T.shape:
                self._cloud_coverage = np.zeros_like(T, dtype=np.float32)
            else:
                self._cloud_coverage.fill(0)
            
            # Conditions for cloud formation (vectorized)
            forming_clouds = self._relative_humidity > rh_threshold
            
            if np.any(forming_clouds):
                # Calculate only where clouds are forming
                rh_diff = np.subtract(self._relative_humidity[forming_clouds], rh_threshold)
                rh_range = rh_max - rh_threshold
                cloud_values = np.divide(rh_diff, rh_range)
                self._cloud_coverage[forming_clouds] = cloud_values
            
            # Apply temperature effects on cloud types (vectorized)
            temp_factor = 1.0 - 0.5 * np.abs((T - 15) / 50)**2
            np.clip(temp_factor, 0, 1, out=temp_factor)
            
            # Apply temperature modification (vectorized)
            np.multiply(self._cloud_coverage, temp_factor, out=self._cloud_coverage)
            
            # --- PERSISTENCE FACTOR ---
            if hasattr(self, 'cloud_cover') and self.cloud_cover is not None:
                # Blend new clouds with existing clouds (vectorized)
                persistence = np.float32(0.3)
                np.multiply(self._cloud_coverage, 1 - persistence, out=self._cloud_coverage)
                np.add(self._cloud_coverage, persistence * self.cloud_cover, out=self._cloud_coverage)
            
            # --- DIURNAL DAMPING ---
            cos_phi = np.cos(self.latitudes_rad).astype(np.float32)
            day_length_factor = np.clip(cos_phi, 0, 1)
            
            diurnal_adjustment = 0.2 - 0.3 * day_length_factor
            np.multiply(self._cloud_coverage, 1.0 + diurnal_adjustment * 0.2, out=self._cloud_coverage)
            
            # Final cloud cover (vectorized)
            np.clip(self._cloud_coverage, 0, 1, out=self._cloud_coverage)
            return self._cloud_coverage.copy()  # Return a copy to avoid reference issues
            
        except Exception as e:
            print(f"Error in calculate_cloud_cover: {e}")
            traceback.print_exc()
            # Return a safe fallback in case of error
            return np.zeros_like(self.temperature_celsius, dtype=np.float32)

    def update_humidity(self):
        """Update humidity using moisture transport and evaporation"""
        try:
            # Time-efficient update approach for water cycle
            if not hasattr(self, 'humidity'):
                self.initialize_humidity()
                
            # Reset the cached precipitation image since precipitation data will change
            if hasattr(self, '_cached_precip_image'):
                self._cached_precip_image = None
            
            # Initialize if not exists
            if not hasattr(self, 'humidity') or self.humidity is None:
                self.humidity = np.full((self.map_height, self.map_width), 0.5, dtype=np.float32)
            
            # Initialize precipitation field if it doesn't exist
            if not hasattr(self, 'precipitation') or self.precipitation is None:
                self.precipitation = np.zeros((self.map_height, self.map_width), dtype=np.float32)
                self._prev_precipitation = np.zeros_like(self.precipitation)
            
            # Check for valid wind fields
            if not hasattr(self, 'u') or self.u is None:
                self.u = np.zeros_like(self.temperature_celsius, dtype=np.float32)
                self.v = np.zeros_like(self.temperature_celsius, dtype=np.float32)
            
            # Ensure time_step_seconds is defined
            if not hasattr(self, 'time_step_seconds'):
                self.time_step_seconds = 3600.0  # Default to 1 hour
            
            # --- SURFACE EVAPORATION ---
            # Ocean evaporation is temperature dependent
            is_ocean = self.elevation <= 0
            is_land = ~is_ocean
            
            # Calculate saturation vapor pressure (Clausius-Clapeyron)
            T = self.temperature_celsius
            saturation_vapor_pressure = self.calculate_water_vapor_saturation(T)
            
            # Vectorized evaporation calculation
            evaporation_base_rate = np.full_like(self.humidity, 0.005, dtype=np.float32)  # Start with land rate
            evaporation_base_rate[is_ocean] = 0.02  # Set ocean rate
            
            # Temperature effect on evaporation (vectorized)
            T_ref = 15.0  # Reference temperature
            temp_factor = 1.0 + 0.07 * (T - T_ref)
            np.clip(temp_factor, 0.2, 5.0, out=temp_factor)  # In-place clipping
            
            # Combine factors into evaporation rate - mm/hour
            evaporation_rate = evaporation_base_rate * temp_factor
            
            # --- PRECIPITATION ---
            # For precipitation, we need to know RH
            vapor_pressure = self.humidity * saturation_vapor_pressure
            relative_humidity = vapor_pressure / saturation_vapor_pressure
            
            # Calculate precipitation rate based on RH (vectorized)
            # Only precipitate when RH is high enough
            precipitation_threshold = 0.7
            new_precipitation_rate = np.zeros_like(relative_humidity, dtype=np.float32)
            precip_mask = relative_humidity > precipitation_threshold
            
            if np.any(precip_mask):
                # Precipitation rate increases with excess RH
                excess_humidity = relative_humidity[precip_mask] - precipitation_threshold
                new_precipitation_rate[precip_mask] = excess_humidity**2 * 15.0  # mm/hour
            
            # Store previous precipitation for persistence calculation
            if not hasattr(self, '_prev_precipitation'):
                self._prev_precipitation = self.precipitation.copy()
            
            # Apply orographic effect (more rain on mountains facing wind)
            # Calculate terrain slope
            if not hasattr(self, '_slope_x') or self._slope_x is None:
                dy, dx = np.gradient(self.elevation)
                self._slope_x = dx / self.grid_spacing_x
                self._slope_y = dy / self.grid_spacing_y
            
            # Orographic factor (precipitation enhancement on windward slopes)
            orographic_factor = np.zeros_like(new_precipitation_rate, dtype=np.float32)
            
            # Apply only on land
            if np.any(is_land):
                # Wind direction effect (wind flowing uphill causes more rain)
                wind_upslope = -(self.u * self._slope_x + self.v * self._slope_y)
                # Only enhance when wind blows uphill
                enhancement_mask = is_land & (wind_upslope > 0)
                if np.any(enhancement_mask):
                    orographic_factor[enhancement_mask] = wind_upslope[enhancement_mask] * 0.5
            
            # Apply orographic enhancement (vectorized)
            orographic_enhancement = np.zeros_like(new_precipitation_rate, dtype=np.float32)
            orographic_enhancement[is_land] = orographic_factor[is_land]
            new_precipitation_rate += orographic_enhancement
            
            # Add persistence to precipitation (rain doesn't stop immediately)
            # This smooths out day/night differences
            precipitation_persistence = 0.85  # Higher value means more persistent rain patterns
            
            # Use spatial smoothing to soften transitions
            smoothed_precip = gaussian_filter(new_precipitation_rate, sigma=2.0)
            
            # Smooth the precipitation field over time to reduce stark day/night contrasts
            self.precipitation = (precipitation_persistence * self._prev_precipitation + 
                                 (1 - precipitation_persistence) * smoothed_precip)
            
            # Additional smoothing to further reduce stark contrasts between regions
            self.precipitation = gaussian_filter(self.precipitation, sigma=1.5)
            
            # Store current precipitation for next cycle
            self._prev_precipitation = self.precipitation.copy()
            
            # --- CLOUD FORMATION ---
            # Update cloud cover with a single call
            self.cloud_cover = self.calculate_cloud_cover()
            
            # --- UPDATE HUMIDITY ---
            # Use actual precipitation for humidity changes (not smoothed precipitation)
            total_factor = self.time_step_seconds
            humidity_change = (evaporation_rate - new_precipitation_rate) * total_factor
            self.humidity += humidity_change
            
            # Apply diffusion and constraints (keep original diffusion for physical accuracy)
            self.humidity = gaussian_filter(self.humidity, sigma=1.0)
            np.clip(self.humidity, 0.01, 1.0, out=self.humidity)  # In-place clipping
            
            # --- TRACK WATER CYCLE BUDGET ---
            # Store diagnostic values
            self.energy_budget['evaporation'] = float(np.mean(evaporation_rate))
            self.energy_budget['precipitation'] = float(np.mean(self.precipitation))
            
        except Exception as e:
            print(f"Error updating humidity: {e}")
            traceback.print_exc()

    def map_precipitation_to_color(self, precipitation_data):
        """Map precipitation to color"""
        try:
            # Performance optimization: Check if precipitation is all zeros
            if np.all(precipitation_data < 0.0001):
                # No precipitation, just return terrain
                return self.map_elevation_to_color(self.elevation)
                
            # Get base map colors first (use terrain for context)
            terrain_colors = self.map_elevation_to_color(self.elevation)
            
            # Pre-allocate the RGBA array
            rgba_array = np.zeros((*terrain_colors.shape[:2], 4), dtype=np.uint8)
            
            # Copy terrain as base
            rgba_array[..., :3] = terrain_colors[..., :3]
            rgba_array[..., 3] = 255
            
            # Create masks for different precipitation levels
            # Use less expensive operations and avoid unnecessary computations
            light_precip_mask = precipitation_data > 0.0005  # Very light precipitation
            
            # If no precipitation above threshold, just return terrain
            if not np.any(light_precip_mask):
                return rgba_array
                
            precip_mask = precipitation_data > 0.001  # Regular precipitation
            heavy_precip_mask = precipitation_data > 0.05  # Heavy precipitation
            
            # Normalize precipitation for coloring (logarithmic scale for better visualization)
            # Cap at reasonable rainfall values (0-50 mm/hr)
            capped_precip = np.clip(precipitation_data, 0, 50)
            
            # Use log scale for better visualization (log(x+1) to handle zeros)
            # More efficient scaling to avoid excessive computation
            log_precip = np.log1p(capped_precip * 5) / np.log1p(250)  # Simplified scaling
            
            # Create blue-scale for rain intensity - optimize memory usage
            rain_colors = np.zeros((*precipitation_data.shape, 4), dtype=np.uint8)
            
            # More gradual color transitions for very light rain
            # Optimize by only computing where needed
            if np.any(light_precip_mask):
                rain_colors[light_precip_mask, 0] = 55 + 20 * log_precip[light_precip_mask]  # Slight red
                rain_colors[light_precip_mask, 1] = 125 + 20 * log_precip[light_precip_mask]  # Some green
                rain_colors[light_precip_mask, 2] = 180 + 20 * log_precip[light_precip_mask]  # Blue component
                rain_colors[light_precip_mask, 3] = 30 + 40 * log_precip[light_precip_mask]  # Low opacity
            
            # Regular precipitation (more visible)
            if np.any(precip_mask):
                rain_colors[precip_mask, 0] = 55 + 100 * log_precip[precip_mask]  # Slight red
                rain_colors[precip_mask, 1] = 125 + 100 * log_precip[precip_mask]  # Some green
                rain_colors[precip_mask, 2] = 200 + 55 * log_precip[precip_mask]  # Strong blue
                rain_colors[precip_mask, 3] = 50 + 150 * log_precip[precip_mask]  # More opacity
            
            # Heavy precipitation (most intense coloring) - only compute if needed
            if np.any(heavy_precip_mask):
                rain_colors[heavy_precip_mask, 0] = 90 + 50 * log_precip[heavy_precip_mask]
                rain_colors[heavy_precip_mask, 1] = 160 + 60 * log_precip[heavy_precip_mask]
                rain_colors[heavy_precip_mask, 2] = 240
                rain_colors[heavy_precip_mask, 3] = 150 + 105 * log_precip[heavy_precip_mask]
            
            # Blend rain with terrain
            alpha = rain_colors[..., 3:] / 255.0
            rgba_array[..., :3] = ((rain_colors[..., :3] * alpha) + 
                                (terrain_colors[..., :3] * (1 - alpha))).astype(np.uint8)
            
            # Special effects for heavy rain - only add if we have heavy rain
            # and limit the number of raindrops for performance
            if np.any(precipitation_data > 0.1):  # Only for significant rainfall
                heavy_rain = log_precip > 0.6
                if np.sum(heavy_rain) > 0:
                    # Limit the number of droplets for performance (max 200 droplets)
                    max_droplets = min(200, int(np.sum(heavy_rain) * 0.03))
                    if max_droplets > 0:
                        # Get coordinates of heavy rain areas
                        heavy_y, heavy_x = np.where(heavy_rain)
                        if len(heavy_y) > 0:
                            # Randomly select a subset of coordinates
                            selected_indices = np.random.choice(len(heavy_y), 
                                                              size=min(max_droplets, len(heavy_y)), 
                                                              replace=False)
                            # Apply droplets only at selected points
                            for idx in selected_indices:
                                y, x = heavy_y[idx], heavy_x[idx]
                                rgba_array[y, x, 0] = 180
                                rgba_array[y, x, 1] = 220
                                rgba_array[y, x, 2] = 255
            
            return rgba_array
            
        except Exception as e:
            print(f"Error in map_precipitation_to_color: {e}")
            traceback.print_exc()
            # Return a safe fallback
            return self.map_elevation_to_color(self.elevation)

    def map_clouds_to_color(self):
        """Map cloud cover to color visualization"""
        try:
            # Get terrain as the background
            terrain_colors = self.map_altitude_to_color(self.elevation)
            
            # Pre-allocate the RGBA array
            rgba_array = np.zeros((*terrain_colors.shape[:2], 4), dtype=np.uint8)
            
            # Copy terrain as base
            rgba_array[..., :3] = terrain_colors[..., :3]
            rgba_array[..., 3] = 255
            
            # Make sure cloud data exists
            if not hasattr(self, 'cloud_cover') or self.cloud_cover is None:
                return rgba_array
            
            # Cloud mask where there are clouds
            cloud_mask = self.cloud_cover > 0.05  # Threshold for visible clouds
            
            if np.any(cloud_mask):
                # Create cloud colors
                cloud_colors = np.zeros((*self.cloud_cover.shape, 4), dtype=np.uint8)
                
                # Scale opacity with cloud cover (more opaque for thicker clouds)
                # White clouds with variable opacity
                cloud_colors[cloud_mask, 0] = 255
                cloud_colors[cloud_mask, 1] = 255
                cloud_colors[cloud_mask, 2] = 255
                
                # Scale opacity with cloud density
                cloud_colors[cloud_mask, 3] = (self.cloud_cover[cloud_mask] * 200).astype(np.uint8)
                
                # Blend clouds with terrain
                alpha = cloud_colors[..., 3:] / 255.0
                rgba_array[..., :3] = ((cloud_colors[..., :3] * alpha) + 
                                    (terrain_colors[..., :3] * (1 - alpha))).astype(np.uint8)
            
            return rgba_array
            
        except Exception as e:
            print(f"Error in map_clouds_to_color: {e}")
            traceback.print_exc()
            # Return a safe fallback
            return terrain_colors

    def _delayed_precipitation_update(self):
        """Handle precipitation map updates separately to prevent UI freezing"""
        try:
            # Make sure precipitation data exists
            if not hasattr(self, 'precipitation') or self.precipitation is None:
                self.precipitation = np.zeros((self.map_height, self.map_width), dtype=np.float32)
            
            # Check if we already have a cached precipitation image
            if hasattr(self, '_cached_precip_image') and self._cached_precip_image is not None:
                # Use cached image to speed up display
                self.current_image = self._cached_precip_image
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
                return
                
            # Generate new precipitation visualization
            display_data = self.map_precipitation_to_color(self.precipitation)
            
            # Convert to PIL Image and create PhotoImage
            image = Image.fromarray(display_data.astype('uint8'))
            self.current_image = ImageTk.PhotoImage(image)
            
            # Cache the precipitation image for future use
            self._cached_precip_image = self.current_image
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
            
        except Exception as e:
            print(f"Error in _delayed_precipitation_update: {e}")
            traceback.print_exc()
            
            # Fallback to showing terrain if precipitation update fails
            try:
                display_data = self.map_elevation_to_color(self.elevation)
                image = Image.fromarray(display_data.astype('uint8'))
                self.current_image = ImageTk.PhotoImage(image)
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
            except Exception as fallback_error:
                print(f"Fallback error: {fallback_error}")
                pass


class ZoomDialog(tk.Toplevel):
    def __init__(self, parent, zoom_factor=4):
        super().__init__(parent)
        
        # Remove window decorations and make it stay on top
        self.overrideredirect(True)
        self.attributes('-topmost', True)
        
        # Set size for zoomed view (odd numbers ensure center pixel)
        self.view_size = 51  # Reduced from 101 to 51 pixels
        self.zoom_factor = zoom_factor
        
        # Calculate canvas size
        canvas_size = self.view_size * self.zoom_factor
        
        # Create canvas for zoomed view
        self.canvas = tk.Canvas(self, width=canvas_size, height=canvas_size)
        self.canvas.pack()
        
        # Create crosshair in the center
        self.crosshair_size = 10  # Increased size for better visibility
        center = canvas_size // 2
        
        # Create crosshair with multiple components for better visibility
        # Main crosshair in red
        self.canvas.create_line(center - self.crosshair_size, center, 
                              center + self.crosshair_size, center, 
                              fill='red', width=2, tags='crosshair')
        self.canvas.create_line(center, center - self.crosshair_size, 
                              center, center + self.crosshair_size, 
                              fill='red', width=2, tags='crosshair')
        
        # White outline for contrast
        outline_offset = 1
        self.canvas.create_line(center - self.crosshair_size, center - outline_offset, 
                              center + self.crosshair_size, center - outline_offset, 
                              fill='white', width=1, tags='crosshair')
        self.canvas.create_line(center - self.crosshair_size, center + outline_offset, 
                              center + self.crosshair_size, center + outline_offset, 
                              fill='white', width=1, tags='crosshair')
        self.canvas.create_line(center - outline_offset, center - self.crosshair_size, 
                              center - outline_offset, center + self.crosshair_size, 
                              fill='white', width=1, tags='crosshair')
        self.canvas.create_line(center + outline_offset, center - self.crosshair_size, 
                              center + outline_offset, center + self.crosshair_size, 
                              fill='white', width=1, tags='crosshair')


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Context already set, continue with current settings
        pass
        
    app = SimulationApp()
    app.root.mainloop()