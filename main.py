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

from map_generation import MapGenerator


# Global variables
map_width = 800  # Adjust value as needed
map_height = 600  # Adjust value as needed
selected_layer = 'elevation'  # Default layer
num_processes = 8  # Number of processes for multiprocessing

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
            self.time_step_seconds = 360
            self.Omega = 7.2921159e-5
            self.P0 = 101325
            self.desired_simulation_step_time = 0.1  # 0.1 seconds between simulation steps

            # Add energy budget tracking
            self.energy_budget = {
                'solar_in': 0.0,
                'greenhouse_effect': 0.0,
                'longwave_out': 0.0,
                'net_flux': 0.0,
                'temperature_history': []
            }

            # Adjustable climate parameters
            self.climate_params = {
                'solar_constant': 1361.1,    # Solar constant (W/m²)
                'albedo_land': 0.3,          # Increased back to standard value
                'albedo_ocean': 0.06,        # Ocean albedo
                'emissivity': 0.9,          # Increased to allow more heat escape
                'climate_sensitivity': 2.0,   # Reduced for more stable response
                'greenhouse_strength': 2.5,   # Reduced greenhouse effect
                'heat_capacity_land': 2e6,    # Doubled land heat capacity for stability
                'heat_capacity_ocean': 4.2e6,   # Doubled ocean heat capacity for stability
                'ocean_heat_transfer': 0.3,   # Ocean heat transfer
                'atmospheric_heat_transfer': 0.5,  # Atmospheric heat transfer
                'baseline_land_temp': 13.9,   # Baseline land temperature
                'max_ocean_temp': 30.0,       # Maximum ocean temperature
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
            self.wind_speed_normalized = None
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
            self.num_processes = multiprocessing.cpu_count() - 16
            if self.num_processes < 1:
                self.num_processes = 1
            print(f'Number of CPU cores available: {self.num_processes}')
            
            # Create Tkinter window
            self.root = tk.Tk()
            self.root.title("Map Layers")
            
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

            # Initialize temperature field
            self.initialize_temperature()

            # Initialize humidity
            self.initialize_humidity()

            # Initialize other attributes
            self.humidity_effect_coefficient = 0.1

            # Initialize global circulation
            self.initialize_global_circulation()

            # Initialize wind field
            self.initialize_wind()

            # Initialize ocean currents
            self.initialize_ocean_currents()

            # Start simulation thread
            simulation_thread = threading.Thread(target=self.simulate)
            simulation_thread.daemon = True
            simulation_thread.start()

            # Start visualization loop
            self.update_visualization()

            self.water_vapor = np.zeros_like(self.temperature_celsius)  # Water vapor mixing ratio
            self.heat_capacity = 1e7  # Heat capacity (J/m²/K)
            self.stefan_boltzmann = 5.67e-8  # Stefan-Boltzmann constant
            self.solar_input = 340.0  # Mean solar input (W/m²)

            # Trigger initial map update
            self.update_map()
            
        except Exception as e:
            print(f"Error in initialization: {e}")

    def setup_gui(self):
        # Configure the root window to prevent resizing
        self.root.resizable(False, False)
        
        # Calculate window dimensions
        # Add buffer for UI elements: 100px for controls, 30px for radio buttons, 30px for mouse_over label
        ui_buffer = 160
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

        # Add sliders for parameters
        self.octaves_slider = tk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL, label='Octaves', resolution=1)
        self.octaves_slider.set(self.global_octaves)
        self.octaves_slider.pack(side=tk.LEFT, padx=5, pady=5)

        self.frequency_slider = tk.Scale(control_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, label='Frequency', resolution=0.1)
        self.frequency_slider.set(self.global_frequency)
        self.frequency_slider.pack(side=tk.LEFT, padx=5, pady=5)

        self.lacunarity_slider = tk.Scale(control_frame, from_=1.0, to=4.0, orient=tk.HORIZONTAL, label='Lacunarity', resolution=0.1)
        self.lacunarity_slider.set(self.global_lacunarity)
        self.lacunarity_slider.pack(side=tk.LEFT, padx=5, pady=5)

        self.persistence_slider = tk.Scale(control_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, label='Persistence', resolution=0.05)
        self.persistence_slider.set(self.global_persistence)
        self.persistence_slider.pack(side=tk.LEFT, padx=5, pady=5)

        # Save slider values whenever they are changed
        self.octaves_slider.config(command=lambda value: self.save_slider_values())
        self.frequency_slider.config(command=lambda value: self.save_slider_values())
        self.lacunarity_slider.config(command=lambda value: self.save_slider_values())
        self.persistence_slider.config(command=lambda value: self.save_slider_values())
        
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

        # Add new radio button for pressure map
        tk.Radiobutton(
            control_frame,
            text="Pressure",
            variable=self.selected_layer,
            value="Pressure"
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

    def save_slider_values(self):
        # [Implement as in the original code...]
        settings = {
            "octaves": self.octaves_slider.get(),
            "frequency": self.frequency_slider.get(),
            "lacunarity": self.lacunarity_slider.get(),
            "persistence": self.persistence_slider.get()
        }
        with open("slider_settings.json", "w") as f:
            json.dump(settings, f)
    
    def load_slider_values(self):
        # [Implement as in the original code...]
        try:
            with open("slider_settings.json", "r") as f:
                settings = json.load(f)
                self.octaves_slider.set(settings["octaves"])
                self.frequency_slider.set(settings["frequency"])
                self.lacunarity_slider.set(settings["lacunarity"])
                self.persistence_slider.set(settings["persistence"])
        except FileNotFoundError:
            print("Slider settings file not found. Using default values.")

    def on_new(self):
        # [Implement as in the original code...]
        self.map_width = self.map_size
        self.map_height = self.map_size
        # Update canvas size
        self.canvas.config(width=self.map_width, height=self.map_height)
        # Generate new terrain
        self.on_generate()
    
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
        # Save the current slider values to JSON
        self.save_slider_values()

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

        # Get values from sliders
        self.global_octaves = self.octaves_slider.get()
        self.global_frequency = self.frequency_slider.get()
        self.global_lacunarity = self.lacunarity_slider.get()
        self.global_persistence = self.persistence_slider.get()

        # Rest of the on_generate implementation...
        # (Continue with the elevation generation and temperature calculation code)

    def simulate(self):
        """Run simulation step"""
        try:
            start_time = time.time()
            
            self.time_step += 1
            
            # Update simulations
            self.update_ocean_temperature()
            self.update_temperature_land_ocean()
            self.update_pressure()
            self.update_wind()
            
            # Update the map display with new values
            self.update_map()

            # Print system statistics
            self.print_system_stats()
            
            # Update displayed values for current mouse position
            self.update_mouse_over()
            
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
            raise e

    def update_mouse_over(self, event=None):
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
            
            # Calculate wind direction in degrees and adjust to 1-360 range
            wind_direction_rad = np.arctan2(-self.u[y, x], -self.v[y, x])  # Swap u and v, and negate for compass direction
            wind_direction_deg = np.degrees(wind_direction_rad)
            wind_direction = (wind_direction_deg + 360) % 360

            # Convert to compass direction (1-360)
            if wind_direction == 0:
                wind_direction = 360

            # Update label text
            self.mouse_over_label.config(text=(
                f"Pressure: {pressure_value:.2f} Pa, "
                f"Temperature: {temperature_value:.2f} °C, "
                f"Wind Speed: {wind_speed_value:.2f} m/s, "
                f"Wind Direction: {wind_direction:.2f}°\n"
                f"Latitude: {latitude_value:.2f}°, Longitude: {longitude_value:.2f}°, "
                f"Elevation: {elevation_value:.2f} m, "
                f"Grayscale: {grayscale_value}, "
                f"Y: {y}, X: {x}"
            ))

    # [Implement the rest of the methods from the original code, adjusting them to use `self` instead of global variables.]

    def normalize_data(self, data):
        """Normalize data to range [0,1]"""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        return np.zeros_like(data)

    def calculate_laplacian(self, field):
        """Calculate the Laplacian of a field using finite difference method."""
        laplacian_x = (np.roll(field, -1, axis=1) - 2 * field + np.roll(field, 1, axis=1)) / (self.grid_spacing_x ** 2)
        laplacian_y = (np.roll(field, -1, axis=0) - 2 * field + np.roll(field, 1, axis=0)) / (self.grid_spacing_y ** 2)
        laplacian = laplacian_x + laplacian_y
        return laplacian

    def calculate_water_density(self, temperature):
        """Calculate water density based on temperature"""
        reference_density = 1000  # kg/m³
        thermal_expansion = 0.0002  # per degree C
        temperature_difference = np.abs(temperature - 4)
        return reference_density * (1 - thermal_expansion * temperature_difference)

    def initialize_wind(self):
        """Initialize wind fields with small random perturbations."""
        # Initialize wind arrays to zeros
        self.u = np.zeros((self.map_height, self.map_width))
        self.v = np.zeros((self.map_height, self.map_width))
        
        # Add small random perturbations to initiate dynamics
        perturbation_scale = 0.1  # Adjust as needed
        self.u += np.random.normal(0, perturbation_scale, self.u.shape)
        self.v += np.random.normal(0, perturbation_scale, self.v.shape)
        
        print("Initialized wind fields with small random perturbations.")

    def update_wind(self):
        """Update wind components dynamically based on atmospheric physics."""     
        # 1. Pressure Gradient Force (PGF)
        pressure_gradient_y = np.gradient(self.pressure, axis=0) / self.grid_spacing_y
        pressure_gradient_x = np.gradient(self.pressure, axis=1) / self.grid_spacing_x
        
        # Air density
        air_density = self.pressure / (287.05 * (self.temperature_celsius + 273.15))
        pgf_u = -pressure_gradient_x / air_density
        pgf_v = -pressure_gradient_y / air_density

        # 2. Coriolis Force (CF)
        f = 2 * self.Omega * np.sin(self.latitudes_rad)
        coriolis_u = f * self.v
        coriolis_v = -f * self.u

        # 3. Friction (Linear Drag)
        friction_coefficient = 1e-5  # Adjust as needed
        friction_u = -friction_coefficient * self.u
        friction_v = -friction_coefficient * self.v

        # 4. Update velocities
        self.u += (pgf_u + coriolis_u + friction_u) * self.time_step_seconds
        self.v += (pgf_v + coriolis_v + friction_v) * self.time_step_seconds

        # 5. Apply Periodic Boundary Conditions for Left and Right Edges
        self.u[:, 0] = self.u[:, -2]
        self.u[:, -1] = self.u[:, 1]
        self.v[:, 0] = self.v[:, -2]
        self.v[:, -1] = self.v[:, 1]

        # 6. Apply Reflective Boundary Conditions for Top and Bottom Edges
        self.u[0, :] = self.u[1, :]       # Bottom edge
        self.u[-1, :] = self.u[-2, :]     # Top edge
        self.v[0, :] = -self.v[1, :]      # Bottom edge (invert v)
        self.v[-1, :] = -self.v[-2, :]    # Top edge (invert v)

        # 7. Limit maximum wind speed
        max_wind_speed = 100.0  # m/s
        wind_speed = np.sqrt(self.u**2 + self.v**2)
        scaling_factor = np.minimum(1.0, max_wind_speed / (wind_speed + 1e-8))
        self.u *= scaling_factor
        self.v *= scaling_factor

        # 8. Apply smoothing (optional)
        self.u = gaussian_filter(self.u, sigma=1.0)
        self.v = gaussian_filter(self.v, sigma=1.0)


    def initialize_ocean_currents(self):
        """Initialize ocean current components"""
        self.ocean_u = np.zeros((self.map_height, self.map_width))
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

        # Apply temperature and density effects
        T_gradient_y, T_gradient_x = np.gradient(self.temperature_celsius)
        density_factor = 0.1
        self.ocean_v += -density_factor * T_gradient_y * is_ocean
        self.ocean_u += -density_factor * T_gradient_x * is_ocean
        
        # Modify currents near continental boundaries
        for y in range(1, self.map_height-1):
            for x in range(1, self.map_width-1):
                if is_ocean[y, x]:
                    if not is_ocean[y-1:y+2, x-1:x+2].all():
                        boundary_factor = 1.5
                        self.ocean_u[y, x] *= boundary_factor
                        self.ocean_v[y, x] *= boundary_factor
        
        # Apply masking and smoothing
        self.ocean_u = np.where(is_ocean, self.ocean_u, 0)
        self.ocean_v = np.where(is_ocean, self.ocean_v, 0)
        self.ocean_u = gaussian_filter(self.ocean_u, sigma=1.0)
        self.ocean_v = gaussian_filter(self.ocean_v, sigma=1.0)

    def update_ocean_temperature(self):
        """Update ocean temperatures"""
        is_ocean = self.elevation <= 0
        
        try:
            # Existing ocean current advection
            dT_dy = np.gradient(self.temperature_celsius, axis=0) / self.grid_spacing_y
            dT_dx = np.gradient(self.temperature_celsius, axis=1) / self.grid_spacing_x
            temperature_advection = -(self.ocean_u * dT_dx + self.ocean_v * dT_dy)
            
            # Thermohaline circulation effects
            density = self.calculate_water_density(self.temperature_celsius)
            density_gradient_y = np.gradient(density, axis=0)
            density_gradient_x = np.gradient(density, axis=1)
            vertical_mixing = 0.1 * density_gradient_y
            
            # Heat transport parameters
            heat_capacity_water = 4186  # J/(kg·K)
            horizontal_diffusivity = 1e3
            vertical_diffusivity = 1e-4
            
            # NEW: Air-sea heat exchange (maintaining array shapes)
            temp_difference = np.zeros_like(self.temperature_celsius)
            temp_difference[is_ocean] = (
                self.temperature_celsius[is_ocean] - np.mean(self.temperature_celsius[is_ocean])
            )
            heat_exchange_coefficient = 40  # W/(m²·K)
            air_sea_heat_exchange = heat_exchange_coefficient * temp_difference
            
            # NEW: Latent heat from evaporation (maintaining array shapes)
            relative_humidity = 0.7  # assumed average
            latent_heat_flux = np.zeros_like(self.temperature_celsius)
            ocean_temp = self.temperature_celsius[is_ocean]
            
            # Calculate saturation vapor pressure only for ocean cells
            saturation_vapor_pressure = 610.7 * np.exp(17.27 * ocean_temp / (ocean_temp + 237.3))
            actual_vapor_pressure = relative_humidity * saturation_vapor_pressure
            evap_flux = -2.5e6 * 1e-3 * (saturation_vapor_pressure - actual_vapor_pressure)
            
            # Assign calculated flux to ocean cells
            latent_heat_flux[is_ocean] = evap_flux
            
            # Calculate temperature changes (all arrays now same shape)
            temperature_change = (
                temperature_advection + 
                horizontal_diffusivity * self.calculate_laplacian(self.temperature_celsius) +
                vertical_diffusivity * vertical_mixing +
                (air_sea_heat_exchange + latent_heat_flux) / (heat_capacity_water * 1000)  # density of water ≈ 1000 kg/m³
            ) * self.time_step_seconds / heat_capacity_water
            
            # Apply changes only to ocean cells
            self.temperature_celsius[is_ocean] += temperature_change[is_ocean]
            
            # Clip ocean temperatures to realistic values
            self.temperature_celsius[is_ocean] = np.clip(
                self.temperature_celsius[is_ocean], 
                -2,  # Minimum ocean temperature
                30   # Maximum ocean temperature
            )
            
        except Exception as e:
            print(f"Error in update_ocean_temperature: {e}")
            raise e


    def update_temperature_land_ocean(self):
        """Update temperature fields with improved greenhouse effect."""
        params = self.climate_params
        
        # Solar input calculation
        S0 = params['solar_constant']
        S_avg = S0 / 4  # Average over sphere
        cos_phi = np.clip(np.cos(self.latitudes_rad), 0.1, None)
        S_lat = S_avg * cos_phi
        
        # Calculate albedo based on surface type
        is_land = self.elevation > 0
        albedo = np.where(is_land, params['albedo_land'], params['albedo_ocean'])
        
        # Incoming solar radiation
        solar_in = (1 - albedo) * S_lat
        self.energy_budget['solar_in'] = np.mean(solar_in)
        
        # Calculate greenhouse effect
        greenhouse_forcing = self.calculate_greenhouse_effect(params)
        self.energy_budget['greenhouse_effect'] = greenhouse_forcing
        
        # Outgoing longwave radiation
        sigma = 5.670374419e-8  # Stefan-Boltzmann constant
        T_kelvin = self.temperature_celsius + 273.15
        longwave_out = params['emissivity'] * sigma * T_kelvin**4
        self.energy_budget['longwave_out'] = np.mean(longwave_out)
        
        # Net energy flux
        net_flux = solar_in + greenhouse_forcing - longwave_out
        self.energy_budget['net_flux'] = np.mean(net_flux)
        
        # Temperature change based on heat capacity
        heat_capacity = np.where(is_land, 
                               params['heat_capacity_land'], 
                               params['heat_capacity_ocean'])
        
        delta_T = (net_flux / heat_capacity) * self.time_step_seconds
        
        # Apply temperature change
        self.temperature_celsius += delta_T
        
        # Apply heat transfer between cells
        self.temperature_celsius = gaussian_filter(
            self.temperature_celsius, 
            sigma=params['atmospheric_heat_transfer']
        )
        
        # Track temperature history
        self.energy_budget['temperature_history'].append(np.mean(self.temperature_celsius))


    def map_to_grayscale(self, data):
        """Convert data to grayscale values"""
        grayscale = np.clip(data, 0, 1)
        grayscale = (grayscale * 255).astype(np.uint8)
        rgb_array = np.stack((grayscale,)*3, axis=-1)
        return rgb_array


    def debug_elevation_point(self, x, y):
        """Debug elevation and color assignment at a specific point"""
        grayscale = (self.elevation_normalized * 255).astype(np.uint8)[y, x]
        elev = self.elevation[y, x]
        print(f"\nDebug at point ({x}, {y}):")
        print(f"Grayscale value: {grayscale}")
        print(f"Elevation: {elev:.2f}m")
        print(f"Is ocean (elev <= 0): {elev <= 0}")
        print(f"Should be land (grayscale > 113): {grayscale > 113}")
        return grayscale, elev


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
        """Update pressure field over time"""
        try:
            # Ensure pressure is float type
            self.pressure = self.pressure.astype(np.float64)
            
            # Get pressure gradients
            dy, dx = np.gradient(self.pressure)
            
            # Calculate pressure tendency
            tendency = np.zeros_like(self.pressure, dtype=np.float64)
            
            # Add small-scale variations based on temperature
            temp_effect = 0.1 * (self.temperature_celsius - np.mean(self.temperature_celsius))
            tendency += gaussian_filter(temp_effect, sigma=2.0)
            
            # Add terrain effects
            terrain_effect = -0.05 * self.elevation.astype(np.float64)
            tendency += gaussian_filter(terrain_effect, sigma=2.0)
            
            # Apply tendency with small time step
            dt = 0.1
            self.pressure += tendency * dt
            
            # Maintain reasonable pressure bounds
            self.pressure = np.clip(self.pressure, 95000.0, 105000.0)
            
            # Smooth the field
            self.pressure = gaussian_filter(self.pressure, sigma=1.0)
            
            # Update normalized version
            self.pressure_normalized = self.normalize_data(self.pressure)
            
        except Exception as e:
            print(f"Error updating pressure: {e}")


    def draw_wind_vectors(self):
        """Simplified wind vector drawing"""
        try:
            # Increase spacing between vectors
            step = max(self.map_width, self.map_height) // 30
            
            # Create coordinate grids
            x_indices = np.arange(0, self.map_width, step)
            y_indices = np.arange(0, self.map_height, step)
            X, Y = np.meshgrid(x_indices, y_indices)
            
            # Sample wind components
            u_sampled = self.u[Y, X]
            v_sampled = self.v[Y, X]
            
            # Calculate magnitudes for scaling
            magnitudes = np.sqrt(u_sampled**2 + v_sampled**2)
            max_magnitude = np.max(magnitudes) if magnitudes.size > 0 else 1.0
            
            # Scale factor for arrow length
            scale = step * 0.5 / max_magnitude if max_magnitude > 0 else step * 0.5
            
            # Draw vectors
            for i in range(len(y_indices)):
                for j in range(len(x_indices)):
                    x = X[i, j]
                    y = Y[i, j]
                    dx = u_sampled[i, j] * scale
                    dy = v_sampled[i, j] * scale
                    
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
        """Update the map display based on selected layer"""
        try:
            if self.selected_layer.get() == "Pressure":
                # Create pressure visualization
                display_data = self.map_pressure_to_color(self.pressure)
                
                # Convert to PIL Image and display
                image = Image.fromarray(display_data.astype('uint8'))  # Ensure uint8 type
                photo = ImageTk.PhotoImage(image)
                self.current_image = photo
                
                # Clear canvas and update scrollregion
                self.canvas.delete("all")
                self.canvas.config(scrollregion=(0, 0, self.map_width, self.map_height))
                
                # Create image on canvas
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                
                return
            
            elif self.selected_layer.get() == "Elevation":
                display_data = self.map_to_grayscale(self.elevation_normalized)
            elif self.selected_layer.get() == "Altitude":
                display_data = self.map_altitude_to_color(self.elevation)
            elif self.selected_layer.get() == "Temperature":
                display_data = self.map_temperature_to_color(self.temperature_celsius)
            elif self.selected_layer.get() == "Wind":
                display_data = self.map_altitude_to_color(self.elevation)
                image = Image.fromarray(display_data.astype('uint8'))
                photo = ImageTk.PhotoImage(image)
                self.current_image = photo
                
                # Clear canvas and update scrollregion
                self.canvas.delete("all")
                self.canvas.config(scrollregion=(0, 0, self.map_width, self.map_height))
                
                # Create image on canvas
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

                # Draw wind vectors
                self.draw_wind_vectors()
                
                return
            elif self.selected_layer.get() == "Ocean Temperature":
                # Create mask for ocean areas
                is_ocean = self.elevation <= 0
                
                # Create a copy of temperature data just for oceans
                ocean_temp = np.copy(self.temperature_celsius)
                ocean_temp[~is_ocean] = np.nan  # Mask land areas
                
                # Normalize ocean temperatures
                ocean_temp_normalized = np.zeros_like(ocean_temp)
                ocean_mask = ~np.isnan(ocean_temp)
                if ocean_mask.any():
                    ocean_temp_normalized[ocean_mask] = self.normalize_data(ocean_temp[ocean_mask])
                
                display_data = self.map_ocean_temperature_to_color(ocean_temp_normalized)
            else:
                display_data = self.map_to_grayscale(self.elevation_normalized)

            # Convert to PIL Image and display
            image = Image.fromarray(display_data.astype('uint8'))
            photo = ImageTk.PhotoImage(image)
            self.current_image = photo
            
            # Clear canvas and update scrollregion
            self.canvas.delete("all")
            self.canvas.config(scrollregion=(0, 0, self.map_width, self.map_height))
            
            # Create image on canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

            # Draw additional elements if needed (wind vectors, etc.)
            if self.selected_layer.get() == "Wind":
                self.draw_wind_vectors()

        except Exception as e:
            print(f"Error updating map: {e}")
            raise e

    
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
        ocean_humidity = 0.9  # Higher humidity over oceans
        land_humidity = 0.6   # Moderate humidity over land

        # Initialize humidity map based on land and ocean
        self.humidity = np.where(is_ocean, ocean_humidity, land_humidity)

        # Blend with latitude gradient for more realism
        blend_factor = 0.3  # Determines the influence of latitude on land humidity
        self.humidity = self.humidity * (1 - blend_factor) + latitude_humidity_gradient * blend_factor

        # Clip humidity values to ensure they remain within [0.0, 1.0]
        self.humidity = np.clip(self.humidity, 0.0, 1.0)

        # Optional: Add random variability for more natural distribution
        # noise = np.random.normal(loc=0.0, scale=0.05, size=self.humidity.shape)
        # self.humidity += noise
        # self.humidity = np.clip(self.humidity, 0.0, 1.0)

        print("Humidity map initialized.")


    def initialize_temperature(self):
        """Initialize temperature field with improved baseline for real-world values, including greenhouse effect."""
        # Constants
        S0 = 1361  # Solar constant in W/m²
        albedo = 0.3  # Earth's average albedo
        sigma = 5.670374419e-8  # Stefan-Boltzmann constant
        lapse_rate = 6.5  # K per km
        humidity_effect_coefficient = 0.1  # K per unit humidity

        # Calculate effective temperature
        S_avg = S0 / 4
        cos_phi = np.clip(np.cos(self.latitudes_rad), 0, None)
        S_lat = S_avg * cos_phi
        T_eff = (((1 - albedo) * S_lat) / sigma) ** 0.25

        # Calculate greenhouse temperature adjustment
        climate_sensitivity = 2.0  # K per W/m²
        greenhouse_temperature_adjustment = climate_sensitivity * self.total_radiative_forcing

        # Calculate humidity adjustment
        humidity_effect_adjustment = humidity_effect_coefficient * self.humidity  # Ensure `self.humidity` is defined appropriately

        # Total surface temperature
        T_surface = T_eff + greenhouse_temperature_adjustment + humidity_effect_adjustment

        # Convert to Celsius
        self.temperature_celsius = T_surface - 273.15  # Convert Kelvin to Celsius

        # Define baseline land temperature
        baseline_land_temperature = 15.0  # °C at sea level

        # Apply lapse rate relative to baseline
        is_land = self.elevation > 0
        altitude_km = np.maximum(self.elevation, 0) / 1000.0
        delta_T_altitude = -lapse_rate * altitude_km
        self.temperature_celsius[is_land] = baseline_land_temperature + delta_T_altitude[is_land]

        # Initialize ocean temperatures with latitude-based gradient
        is_ocean = self.elevation <= 0
        self.temperature_celsius[is_ocean] = np.clip(
            20 - (np.abs(self.latitude[is_ocean]) / 90) * 40,  # Temperature decreases with latitude
            -2,  # Minimum ocean temperature
            30   # Maximum ocean temperature
        )

        # Clip temperatures to realistic bounds
        self.temperature_celsius = np.clip(self.temperature_celsius, -50.0, 50.0)

        # Normalize temperature for visualization
        self.temperature_normalized = self.normalize_data(self.temperature_celsius)

        # Initialize pressure after temperature
        self.initialize_pressure()


    def initialize_pressure(self):
        """Initialize realistic global pressure patterns"""
        try:
            # Standard sea-level pressure in Pa (explicitly as float)
            P0 = 101325.0
            
            # Create latitude and longitude grids
            lats = np.linspace(90, -90, self.map_height)
            lons = np.linspace(-180, 180, self.map_width)
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
            lat_rad = np.deg2rad(lat_grid)
            
            # Base pressure field (explicitly as float)
            self.pressure = np.full((self.map_height, self.map_width), P0, dtype=np.float64)
            
            # Add major pressure systems
            self.pressure += 1500.0 * np.exp(-((lat_rad - np.deg2rad(30))**2) / (2 * (np.pi/15)**2))
            self.pressure += 1500.0 * np.exp(-((lat_rad + np.deg2rad(30))**2) / (2 * (np.pi/15)**2))
            
            # Subpolar lows
            self.pressure -= 2000.0 * np.exp(-((lat_rad - np.deg2rad(60))**2) / (2 * (np.pi/15)**2))
            self.pressure -= 2000.0 * np.exp(-((lat_rad + np.deg2rad(60))**2) / (2 * (np.pi/15)**2))
            
            # ITCZ (Equatorial low)
            self.pressure -= 500.0 * np.exp(-(lat_rad**2) / (2 * (np.pi/20)**2))
            
            # Apply elevation effects
            self.pressure *= np.exp(-self.elevation.astype(np.float64) / 7400.0)
            
            # Smooth the field
            self.pressure = gaussian_filter(self.pressure, sigma=3.0)
            
            # Store normalized version for visualization
            self.pressure_normalized = self.normalize_data(self.pressure)
            
        except Exception as e:
            print(f"Error initializing pressure: {e}")


    def map_pressure_to_color(self, pressure_data):
        """
        Map pressure to a red-blue color gradient.
        Red indicates low pressure, blue indicates high pressure.
        """
        # Create RGBA array
        rgba_array = np.zeros((self.map_height, self.map_width, 4), dtype=np.uint8)
        
        # Set fixed pressure range for consistent coloring (in Pa)
        min_pressure = 95000.0  # Typical low pressure
        max_pressure = 105000.0 # Typical high pressure
        
        # Normalize pressure to [0,1] using fixed range
        pressure_normalized = np.clip((pressure_data - min_pressure) / (max_pressure - min_pressure), 0, 1)
        
        # Create pressure color gradient (red=low, blue=high)
        rgba_array[..., 0] = ((1 - pressure_normalized) * 255).astype(np.uint8)  # Red
        rgba_array[..., 2] = (pressure_normalized * 255).astype(np.uint8)        # Blue
        
        # Add green component for better visualization
        rgba_array[..., 1] = (np.minimum(pressure_normalized, 1 - pressure_normalized) * 255).astype(np.uint8)
        
        # Set alpha channel
        rgba_array[..., 3] = 192  # 75% opacity
        
        # Make water areas more transparent
        is_water = self.elevation <= 0
        rgba_array[is_water, 3] = 128  # 50% transparency for water
        
        return rgba_array


    def advect_field(self, field, u, v):
        """Advect a scalar field using the velocities u and v with a semi-Lagrangian scheme."""
        ny, nx = field.shape
        x = np.arange(nx)
        y = np.arange(ny)
        interpolator = RegularGridInterpolator((y, x), field, bounds_error=False, fill_value=None)

        # Create a meshgrid of coordinates
        Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

        # Compute departure points
        X_depart = X - (u * self.time_step_seconds) / self.grid_spacing_x
        Y_depart = Y - (v * self.time_step_seconds) / self.grid_spacing_y

        # Apply Periodic Boundary Conditions for X (longitude)
        X_depart = X_depart % nx

        # Apply Reflective Boundary Conditions for Y (latitude)
        Y_depart = np.clip(Y_depart, 0, ny - 1)

        # Prepare points for interpolation
        points = np.stack([Y_depart.ravel(), X_depart.ravel()], axis=-1)

        # Interpolate to get advected field
        field_advected = interpolator(points).reshape((ny, nx))

        return field_advected


    def update_visualization(self):
        """Update the map display with current data"""
        try:
            # Clear previous layers
            self.canvas.delete("data_layer")
            
            if self.selected_layer.get() == "Temperature":
                # Update temperature colors
                colors = self.map_temperature_to_color(self.temperature_celsius)
            elif self.selected_layer.get() == "Pressure":
                # Update pressure colors
                colors = self.map_pressure_to_color(self.pressure)
            else:
                return
            
            # Convert to PhotoImage format
            image = Image.fromarray(colors)
            self.current_photo = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.canvas.create_image(
                0, 0, 
                image=self.current_photo, 
                anchor="nw",
                tags="data_layer"
            )
            
            # Update other visualization elements
            self.update_mouse_over()
            
        except Exception as e:
            print(f"Error updating visualization: {e}")
            print(f"Current mode: {self.selected_layer.get()}")
            print(f"Pressure shape: {self.pressure.shape if hasattr(self, 'pressure') else 'No pressure data'}")
        
        # Schedule next update
        self.root.after(30, self.update_visualization)


    def calculate_greenhouse_effect(self, params):
        """
        Calculate greenhouse effect with more physically-based parameters.
        Returns forcing in W/m²
        """
        # Base greenhouse gases forcing (W/m²)
        # Values based on IPCC AR6 estimates
        base_forcings = {
            'co2': 5.35 * np.log(self.co2_concentration / 278),  # Pre-industrial CO2 was 278 ppm
            'ch4': 0.036 * (np.sqrt(self.ch4_concentration) - np.sqrt(722)),  # Pre-industrial CH4 was 722 ppb
            'n2o': 0.12 * (np.sqrt(self.n2o_concentration) - np.sqrt(270))   # Pre-industrial N2O was 270 ppb
        }
        
        # Calculate water vapor contribution
        # Water vapor accounts for about 60% of Earth's greenhouse effect
        T_kelvin = self.temperature_celsius + 273.15
        reference_temp = 288.15  # 15°C in Kelvin
        
        # Base water vapor effect (should be ~90 W/m² at 15°C)
        base_water_vapor = 90.0
        
        # Calculate water vapor forcing with temperature dependence
        # Avoid negative values by using exponential relationship
        temp_factor = np.exp(0.07 * (np.mean(T_kelvin) - reference_temp))
        water_vapor_forcing = base_water_vapor * temp_factor
        
        # Other greenhouse gases (ozone, CFCs, etc.)
        other_ghg = 25.0
        
        # Calculate ocean absorption modifier
        ocean_absorption = self.calculate_ocean_co2_absorption()
        
        # Total greenhouse forcing with ocean absorption
        ghg_forcing = sum(base_forcings.values())  # CO2, CH4, N2O
        total_forcing = ((ghg_forcing + water_vapor_forcing + other_ghg) * 
                        params['greenhouse_strength'] * 
                        (1.0 + ocean_absorption))  # Apply ocean absorption modifier
        
        # Store individual components for debugging
        self.energy_budget.update({
            'co2_forcing': base_forcings['co2'],
            'ch4_forcing': base_forcings['ch4'],
            'n2o_forcing': base_forcings['n2o'],
            'water_vapor_forcing': water_vapor_forcing,
            'other_ghg_forcing': other_ghg,
            'ocean_absorption': ocean_absorption,
            'total_greenhouse_forcing': total_forcing
        })

        return total_forcing


    def calculate_water_vapor_saturation(self, T):
        """
        Calculate water vapor saturation pressure using the Clausius-Clapeyron equation
        T: Temperature in Celsius
        Returns: Saturation vapor pressure in Pa
        """
        T_kelvin = T + 273.15
        # Constants for water vapor
        L = 2.5e6  # Latent heat of vaporization (J/kg)
        Rv = 461.5  # Gas constant for water vapor (J/(kg·K))
        T0 = 273.15  # Reference temperature (K)
        e0 = 611.0  # Reference vapor pressure (Pa)
        
        return e0 * np.exp((L/Rv) * (1/T0 - 1/T_kelvin))


    def calculate_relative_humidity(self, vapor_pressure, T):
        """Calculate relative humidity from vapor pressure and temperature"""
        saturation_pressure = self.calculate_water_vapor_saturation(T)
        return np.clip(vapor_pressure / saturation_pressure, 0, 1)


    def calculate_cloud_albedo(self, relative_humidity):
        """Reduce cloud cooling effect"""
        base_cloud_albedo = 0.15  # Reduced from 0.2
        max_cloud_albedo = 0.45   # Reduced from 0.55
        cloud_threshold = 0.8     # Increased from 0.75
        
        cloud_cover = np.clip((relative_humidity - cloud_threshold) / (1 - cloud_threshold), 0, 1)
        return base_cloud_albedo + (max_cloud_albedo - base_cloud_albedo) * cloud_cover


    def calculate_surface_albedo(self, T):
        """Soften ice-albedo feedback"""
        ice_albedo = 0.65        # Reduced from 0.7
        land_albedo = 0.3
        ocean_albedo = 0.07      # Slightly increased from 0.06
        
        # Widen temperature range for ice formation
        ice_threshold = -2.0     # Lowered from 0
        transition_range = 15.0  # Increased from 10
        
        ice_coverage = np.clip((ice_threshold - T) / transition_range, 0, 1)
        
        is_ocean = self.elevation <= 0
        base_albedo = np.where(is_ocean, ocean_albedo, land_albedo)
        
        return np.where(T < ice_threshold, 
                       base_albedo * (1 - ice_coverage) + ice_albedo * ice_coverage,
                       base_albedo)


    def calculate_lapse_rate_feedback(self, T):
        standard_lapse_rate = 6.5
        min_lapse_rate = 5.5     # Increase minimum
        max_lapse_rate = 7.5     # Decrease maximum
        # Adjust temperature range for variation
        return standard_lapse_rate + (max_lapse_rate - standard_lapse_rate) * \
               np.clip((15 - T) / 25, -1, 1)  # Reduced range from 30 to 25


    def initialize_global_circulation(self):
        """Initialize wind patterns with lower speeds"""
        try:
            # Create latitude array (90 at top to -90 at bottom)
            latitudes = np.linspace(90, -90, self.map_height)
            
            # Initialize wind components
            self.u = np.zeros((self.map_height, self.map_width))
            self.v = np.zeros((self.map_height, self.map_width))
            
            # For each row (latitude band)
            for y in range(self.map_height):
                lat = latitudes[y]
                lat_rad = np.deg2rad(lat)
                
                # Reduced latitude-dependent speed scaling
                speed_scale = 0.5 + 0.2 * np.cos(4 * lat_rad)  # Reduced from 0.65 + 0.35
                
                # Northern Hemisphere
                if lat > 60:  # Polar easterlies
                    self.u[y, :] = 1.5 * speed_scale  # Reduced from 3.0
                    self.v[y, :] = -0.2 * speed_scale  # Reduced from -0.5
                elif lat > 30:  # Westerlies
                    self.u[y, :] = -3.5 * speed_scale  # Reduced from -7.0
                    self.v[y, :] = 0.5 * speed_scale  # Reduced from 1.0
                elif lat > 0:  # Trade winds
                    self.u[y, :] = 2.5 * speed_scale  # Reduced from 5.0
                    self.v[y, :] = -0.5 * speed_scale  # Reduced from -1.0
                # Southern Hemisphere (mirror of northern)
                elif lat > -30:  # Trade winds
                    self.u[y, :] = 2.5 * speed_scale  # Reduced from 5.0
                    self.v[y, :] = 0.5 * speed_scale  # Reduced from 1.0
                elif lat > -60:  # Westerlies
                    self.u[y, :] = -3.5 * speed_scale  # Reduced from -7.0
                    self.v[y, :] = -0.5 * speed_scale  # Reduced from -1.0
                else:  # Polar easterlies
                    self.u[y, :] = 1.5 * speed_scale  # Reduced from 3.0
                    self.v[y, :] = 0.2 * speed_scale  # Reduced from 0.5
                
                # Reduced longitude-dependent variations
                wave_pattern = 0.1 * np.sin(3 * np.pi * np.linspace(0, 1, self.map_width))  # Reduced from 0.15
                self.u[y, :] *= (1 + wave_pattern)
                
                # Terrain effects
                terrain_factor = np.exp(-np.maximum(0, self.elevation[y, :]) / 5000)
                self.u[y, :] *= terrain_factor
                self.v[y, :] *= terrain_factor
            
            # Apply cosine latitude scaling
            cos_lat = np.cos(np.deg2rad(latitudes))
            cos_lat = cos_lat.reshape(-1, 1)
            self.u *= cos_lat
            self.v *= cos_lat
            
            # Smooth transitions
            self.u = gaussian_filter(self.u, sigma=1.5)
            self.v = gaussian_filter(self.v, sigma=1.5)
            
            # Minimal random variations
            random_variation = 0.02  # Reduced from 0.05
            self.u += random_variation * np.random.randn(*self.u.shape)
            self.v += random_variation * np.random.randn(*self.v.shape)
            
        except Exception as e:
            print(f"Error in global circulation initialization: {e}")



    def calculate_wind(self):
        """Calculate wind modifications based on pressure and temperature"""
        try:
            # Start with base circulation
            if not hasattr(self, 'u') or not hasattr(self, 'v'):
                self.initialize_global_circulation()
            
            # Get pressure gradients
            dy, dx = np.gradient(self.pressure)
            dpdy = dy / self.grid_spacing_y
            dpdx = dx / self.grid_spacing_x
            
            # Get temperature gradients
            dTdy, dTdx = np.gradient(self.temperature_celsius)
            dTdy = dTdy / self.grid_spacing_y
            dTdx = dTdx / self.grid_spacing_x
            
            # Calculate thermal wind contribution
            thermal_factor = 0.1
            u_thermal = -thermal_factor * dTdy
            v_thermal = thermal_factor * dTdx
            
            # Calculate pressure gradient contribution
            pressure_factor = 1e-5
            u_pressure = -pressure_factor * dpdy
            v_pressure = pressure_factor * dpdx
            
            # Add modifications to base circulation
            self.u += u_thermal + u_pressure
            self.v += v_thermal + v_pressure
            
            # Apply friction over land
            is_land = self.elevation > 0
            friction_factor = np.where(is_land, 0.5, 0.9)
            self.u *= friction_factor
            self.v *= friction_factor
            
            # Smooth the wind field
            self.u = gaussian_filter(self.u, sigma=1.0)
            self.v = gaussian_filter(self.v, sigma=1.0)
            
            # Limit maximum wind speeds
            max_speed = 30.0  # m/s
            speed = np.sqrt(self.u**2 + self.v**2)
            speed_factor = np.where(speed > max_speed, max_speed/speed, 1.0)
            self.u *= speed_factor
            self.v *= speed_factor
            
        except Exception as e:
            print(f"Error in wind calculation: {e}")


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
            else:
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
        # Get average/min/max values
        avg_temp = np.mean(self.temperature_celsius)
        min_temp = np.min(self.temperature_celsius)
        max_temp = np.max(self.temperature_celsius)
        
        avg_pressure = np.mean(self.pressure)
        min_pressure = np.min(self.pressure)
        max_pressure = np.max(self.pressure)
        
        wind_speed = np.sqrt(self.u**2 + self.v**2)
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
                'net_flux': net_flux
            }
        
        # Calculate all deltas
        temp_change = avg_temp - self.last_values['temp']
        pressure_change = avg_pressure - self.last_values['pressure']
        solar_change = solar_in - self.last_values['solar']
        greenhouse_change = greenhouse - self.last_values['greenhouse']
        net_flux_change = net_flux - self.last_values['net_flux']
        
        # Update last values
        self.last_values.update({
            'temp': avg_temp,
            'pressure': avg_pressure,
            'solar': solar_in,
            'greenhouse': greenhouse,
            'net_flux': net_flux
        })
        
        # Create the output strings with fixed width
        cycle_str = f"Simulation Cycle: {self.time_step:6d}"
        temp_str = f"Temperature (°C)    | Avg: {avg_temp:6.1f} | Min: {min_temp:6.1f} | Max: {max_temp:6.1f} | Δ: {temp_change:+6.2f}"
        pres_str = f"Pressure (Pa)      | Avg: {avg_pressure:8.0f} | Min: {min_pressure:8.0f} | Max: {max_pressure:8.0f} | Δ: {pressure_change:+6.0f}"
        wind_str = f"Wind Speed (m/s)   | Avg: {avg_wind:6.1f} | Max: {max_wind:6.1f}"
        
        # Energy budget strings with net changes
        energy_in_str = f"Energy In (W/m²)   | Solar: {solar_in:6.1f} | Δ: {solar_change:+6.2f} | Greenhouse: {greenhouse:6.1f} | Δ: {greenhouse_change:+6.2f} | Total: {solar_in + greenhouse:6.1f}"
        energy_out_str = f"Energy Out (W/m²)  | Longwave: {longwave_out:6.1f} | Net Flux: {net_flux:6.1f} | Δ: {net_flux_change:+6.2f}"
        
        # Only print these lines once at the start
        if self.time_step == 1:
            print("\n" * 6)  # Create six blank lines
            print("\033[6A", end='')  # Move cursor up six lines
        
        # Update all lines in place
        print(f"\033[K{cycle_str}", end='\r\n')      # Clear line and move to next
        print(f"\033[K{temp_str}", end='\r\n')       # Clear line and move to next
        print(f"\033[K{pres_str}", end='\r\n')       # Clear line and move to next
        print(f"\033[K{wind_str}", end='\r\n')       # Clear line and move to next
        print(f"\033[K{energy_in_str}", end='\r\n')  # Clear line and move to next
        print(f"\033[K{energy_out_str}", end='\r')   # Clear line and stay there
        
        # Move cursor back up to the start position
        print("\033[6A", end='')


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
    multiprocessing.set_start_method("spawn")
    app = SimulationApp()
    app.root.mainloop()  # Add this line to start the Tkinter event loop