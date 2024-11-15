import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time
import json
import multiprocessing
from scipy.ndimage import gaussian_filter

from map_generation import MapGenerator

# Global variables
map_width = 800  # Adjust value as needed
map_height = 600  # Adjust value as needed
selected_layer = 'elevation'  # Default layer
num_processes = 8  # Number of processes for multiprocessing

class SimulationApp:
    def __init__(self):
        # Initialize all variables
        self.map_size = 512
        self.map_width = None  # Will be set when image is loaded
        self.map_height = None # Will be set when image is loaded
        
        # Initialize grid spacing
        earth_circumference = 40075000  # meters
        self.grid_spacing_x = None  # Will be set after map dimensions are known
        self.grid_spacing_y = None  # Will be set after map dimensions are known
        
        # Time and simulation variables
        self.time_step = 0
        self.time_step_seconds = 360
        self.Omega = 7.2921159e-5
        self.P0 = 101325
        
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
        self.temperature_normalized = None
        self.pressure = None
        self.pressure_normalized = None
        self.wind_speed_normalized = None
        self.u = None
        self.v = None
        self.ocean_u = None
        self.ocean_v = None
        
        # Multiprocessing setup
        self.num_processes = multiprocessing.cpu_count() - 16
        if self.num_processes < 1:
            self.num_processes = 1
        print(f'Number of CPU cores available: {self.num_processes}')
        
        # Create Tkinter window
        self.root = tk.Tk()
        self.root.title("Map Layers")
        
        # Variable to track the selected layer
        self.selected_layer = tk.StringVar(value='Wind')
        
        # Load initial data (this will set map dimensions)
        self.on_load("D:\\dev\\planetsim\\images\\1024x512_earth_8bit.png")
        
        # Setup GUI components (after we know the map dimensions)
        self.setup_gui()

        # Start simulation after initialization
        self.simulate()

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

        # Label at the bottom to display the current values at the mouse position
        self.mouse_over_label = tk.Label(self.root, text="Pressure: --, Temperature: --, Wind Speed: --, Wind Direction: --")
        self.mouse_over_label.pack(anchor="w")

        # Bind the mouse motion event to the canvas
        self.canvas.bind("<Motion>", self.update_mouse_over)

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
                print(f"Loaded image size: {self.map_width}x{self.map_height}")
                print(f"Image mode: {elevation_image.mode}")

                # Convert image to numpy array
                img_array = np.array(elevation_image)
                img_array = img_array.astype(int)
                if len(img_array.shape) > 2:  # If image is RGB/RGBA
                    img_array = img_array[:, :, 0]  # Take first channel

                # Get actual grayscale range
                min_gray = int(img_array.min())
                max_gray = img_array.max()
                print(f"Grayscale range: {min_gray} to {max_gray}")

                # Define sea level grayscale value
                sea_level = 114
                print(f"Using sea_level grayscale value: {sea_level}")

                # Initialize elevation array
                self.elevation = np.zeros_like(img_array, dtype=float)

                # Create masks for ocean and land
                land_mask = img_array > sea_level
                print(f"Land mask shape: {land_mask.shape}")
                ocean_mask = img_array < sea_level
                print(f"Ocean mask shape: {ocean_mask.shape}")
                sea_level_mask = img_array == sea_level
                print(f"Sea level mask shape: {sea_level_mask.shape}")

                # Calculate number of steps
                ocean_steps = sea_level - min_gray
                print(f"Ocean steps: {ocean_steps}")
                land_steps = max_gray - sea_level
                print(f"Land steps: {land_steps}")

                # Avoid division by zero
                if ocean_steps == 0:
                    ocean_steps = 1
                if land_steps == 0:
                    land_steps = 1

                # Calculate meters per step
                meters_per_step_ocean = 10984 / ocean_steps
                meters_per_step_land = 8848 / land_steps

                # Debugging prints
                print(f"Ocean steps: {ocean_steps}, Meters per ocean step: {meters_per_step_ocean}")
                print(f"Number of ocean pixels: {np.sum(ocean_mask)}")

                # Map ocean depths
                if np.any(ocean_mask):
                    depth_grayscale = sea_level - img_array[ocean_mask]
                    print(f"Sample depth grayscale values: {depth_grayscale[:5]}")
                    self.elevation[ocean_mask] = -depth_grayscale * meters_per_step_ocean
                    print(f"Sample ocean elevations: {self.elevation[ocean_mask][:5]}")

                # Map land elevations
                if np.any(land_mask):
                    # Adjust grayscale values to start from 0 for the first land level
                    min_land_gray = sea_level + 1  # First grayscale value above sea level
                    height_grayscale = img_array[land_mask] - min_land_gray
                    # Now normalize from 0 to max possible height
                    normalized_heights = height_grayscale / (max_gray - min_land_gray)
                    # Map normalized heights to elevation range 0m to 8848m
                    self.elevation[land_mask] = normalized_heights * 8848

                # Set sea level points to exactly 0
                self.elevation[sea_level_mask] = 0

                # Validation checks
                land_max = np.max(self.elevation[land_mask]) if np.any(land_mask) else 0
                land_min = np.min(self.elevation[land_mask]) if np.any(land_mask) else 0
                ocean_max = np.max(self.elevation[ocean_mask]) if np.any(ocean_mask) else 0
                ocean_min = np.min(self.elevation[ocean_mask]) if np.any(ocean_mask) else 0

                print("\nElevation Range Validation:")
                print(f"Land elevation range: {land_min:.2f}m to {land_max:.2f}m")
                print(f"Ocean elevation range: {ocean_min:.2f}m to {ocean_max:.2f}m")

                if land_max > 8848:
                    print(f"WARNING: Land elevation exceeds Mount Everest ({land_max:.2f}m > 8848m)")
                if ocean_min < -10984:
                    print(f"WARNING: Ocean depth exceeds Mariana Trench ({ocean_min:.2f}m < -10984m)")

                # Calculate normalized elevation (0-1 range)
                self.elevation_normalized = (self.elevation - self.elevation.min()) / (self.elevation.max() - self.elevation.min())

                # Initialize grid spacing
                earth_circumference = 40075000  # meters
                self.grid_spacing_x = earth_circumference / self.map_width
                self.grid_spacing_y = earth_circumference / (2 * self.map_height)  # Divide by 2 because latitude only covers -90 to 90

                # Initialize temperature field
                self.initialize_temperature()

                # Initialize wind field
                self.initialize_wind()

                # Initialize ocean currents
                self.initialize_ocean_currents()

            except Exception as e:
                print(f"Error loading image: {e}")
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

        # Latitude and Longitude arrays
        latitudes = np.linspace(90, -90, self.map_height)
        longitudes = np.linspace(-180, 180, self.map_width)
        self.latitude, self.longitude = np.meshgrid(latitudes, longitudes, indexing='ij')

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
            self.update_map()
            self.update_mouse_over()
            
            elapsed_time = time.time() - start_time
            print(f"Cycle {self.time_step} completed in {elapsed_time:.4f} seconds")
            
            # Schedule next simulation step with a minimum delay
            min_delay = 100  # milliseconds
            delay = max(min_delay, int(elapsed_time * 1000))
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
            
            # Calculate wind direction in degrees and adjust to 1-360 range
            wind_direction_rad = np.arctan2(-self.u[y, x], -self.v[y, x])  # Swap u and v, and negate for compass direction
            wind_direction_deg = np.degrees(wind_direction_rad)
            wind_direction = (wind_direction_deg + 360) % 360

            # Convert to compass direction (1-360)
            if wind_direction == 0:
                wind_direction = 360

            # Retrieve latitude, longitude, and elevation
            latitude_value = self.latitude[y, x]
            longitude_value = self.longitude[y, x]
            elevation_value = self.elevation[y, x]
            
            # Calculate grayscale value (0-255)
            grayscale_value = int(self.elevation_normalized[y, x] * 255)

            # Update label text
            self.mouse_over_label.config(text=(
                f"Pressure: {pressure_value:.2f} Pa, "
                f"Temperature: {temperature_value:.2f} °C, "
                f"Wind Speed: {wind_speed_value:.2f} m/s, "
                f"Wind Direction: {wind_direction:.2f}°\n"
                f"Latitude: {latitude_value:.2f}°, Longitude: {longitude_value:.2f}°, "
                f"Elevation: {elevation_value:.2f} m, Grayscale: {grayscale_value}"
                f"Altitude: {self.altitude[y, x]:.2f} m, "
                f"Y: {y}, X: {x}"
            ))

    # [Implement the rest of the methods from the original code, adjusting them to use `self` instead of global variables.]

    def normalize_data(self, data):
        """Normalize data to range [0,1]"""
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val) if max_val > min_val else data

    def calculate_laplacian(self, field):
        """Calculate the Laplacian of a field."""
        laplacian = (
            np.roll(field, -1, axis=0) + np.roll(field, 1, axis=0) +
            np.roll(field, -1, axis=1) + np.roll(field, 1, axis=1) -
            4 * field
        ) / (self.grid_spacing_x * self.grid_spacing_y)
        return laplacian

    def calculate_water_density(self, temperature):
        """Calculate water density based on temperature"""
        reference_density = 1000  # kg/m³
        thermal_expansion = 0.0002  # per degree C
        temperature_difference = np.abs(temperature - 4)
        return reference_density * (1 - thermal_expansion * temperature_difference)

    def initialize_wind(self):
        """Initialize wind components based on known wind patterns for each atmospheric cell."""
        
        # Initialize wind component arrays
        self.u = np.zeros((self.map_height, self.map_width))
        self.v = np.zeros((self.map_height, self.map_width))

        # Constants for each cell's wind strength
        trade_wind_strength = 5.0     # Hadley cell (Trade winds)
        westerlies_strength = 7.0     # Ferrel cell (Westerlies)
        polar_strength = 3.0          # Polar easterlies
        
        # Define the cell boundaries
        hadley_boundary = 30
        ferrel_boundary = 60

        # Generate wind patterns for each cell
        for y, lat in enumerate(np.linspace(90, -90, self.map_height)):
            abs_lat = abs(lat)

            # Doldrums (near the equator)
            if abs_lat < 5:
                self.u[y, :] = 0
                self.v[y, :] = 0

            # Hadley Cell (5° to 30°) - Trade Winds
            elif abs_lat < hadley_boundary:
                intensity = trade_wind_strength
                self.u[y, :] = -intensity  # East to West
                self.v[y, :] = intensity * np.sign(lat)  # Southward in Northern Hemisphere, Northward in Southern Hemisphere

            # Horse Latitudes (28° to 32°)
            elif 28 < abs_lat < 32:
                self.u[y, :] = 0
                self.v[y, :] = 0

            # Ferrel Cell (30° to 60°) - Westerlies
            elif abs_lat < ferrel_boundary:
                intensity = westerlies_strength
                self.u[y, :] = intensity * np.sign(lat)  # West to East in both hemispheres
                self.v[y, :] = -intensity * 0.2 * np.sign(lat)  # Equatorward flow

            # Polar Cell (60° to 90°) - Polar Easterlies
            else:
                intensity = polar_strength
                self.u[y, :] = -intensity  # East to West
                self.v[y, :] = intensity * np.sign(lat)  # Southward in Northern Hemisphere, Northward in Southern Hemisphere

        # Apply land-sea differences
        land_mask = self.elevation > 0
        ocean_mask = self.elevation <= 0
        self.u[ocean_mask] *= 1.2  # Amplify over oceans
        self.v[ocean_mask] *= 1.2
        self.u[land_mask] *= 0.8  # Reduce over land
        self.v[land_mask] *= 0.8

        # Apply smoothing to prevent abrupt changes
        self.u = gaussian_filter(self.u, sigma=2.0)
        self.v = gaussian_filter(self.v, sigma=2.0)

        # Calculate wind speed for normalization
        wind_speed = np.sqrt(self.u**2 + self.v**2)
        self.wind_speed_normalized = self.normalize_data(wind_speed)


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
            # Ocean current advection
            # Calculate gradients separately for y and x directions
            dT_dy = np.gradient(self.temperature_celsius, axis=0) / self.grid_spacing_y
            dT_dx = np.gradient(self.temperature_celsius, axis=1) / self.grid_spacing_x
            temperature_advection = -(self.ocean_u * dT_dx + self.ocean_v * dT_dy)
            
            # Thermohaline circulation effects
            density = self.calculate_water_density(self.temperature_celsius)
            density_gradient_y = np.gradient(density, axis=0)
            density_gradient_x = np.gradient(density, axis=1)
            vertical_mixing = 0.1 * density_gradient_y
            
            # Heat transport parameters
            heat_capacity_water = 4186
            horizontal_diffusivity = 1e3
            vertical_diffusivity = 1e-4
            
            # Calculate temperature changes
            temperature_change = (
                temperature_advection + 
                horizontal_diffusivity * self.calculate_laplacian(self.temperature_celsius) +
                vertical_diffusivity * vertical_mixing
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
        """Update temperature for both land and ocean"""
        print(f"Starting temperature update at step {self.time_step}")  # Debug
        
        # Store initial temperature for comparison
        initial_temp = self.temperature_celsius.copy()
        
        heat_capacity = 4e6
        sigma = 5.670374419e-8
        albedo_land = 0.3
        albedo_ocean = 0.06
        emissivity = 0.9

        self.altitude = np.where(self.elevation > 0, self.elevation, 0)
        is_land = self.elevation > 0
        is_ocean = self.elevation <= 0

        # Solar radiation calculation
        S0 = 1361
        S_avg = S0 / 4
        solar_radiation_factor = 0.5 * (1 + np.cos(self.latitudes_rad))
        solar_radiation_factor = gaussian_filter(solar_radiation_factor, sigma=3)
        solar_radiation_factor = np.clip(solar_radiation_factor, 0, None)

        S_lat = S_avg * solar_radiation_factor

        # Incoming radiation
        incoming_radiation = np.zeros_like(self.temperature_celsius)
        incoming_radiation[is_land] = (1 - albedo_land) * S_lat[is_land]
        incoming_radiation[is_ocean] = (1 - albedo_ocean) * S_lat[is_ocean]

        # Temperature calculations
        T_kelvin = self.temperature_celsius + 273.15
        T_kelvin = np.clip(T_kelvin, 200.0, 320.0)
        outgoing_radiation = emissivity * sigma * T_kelvin ** 4
        net_radiation = incoming_radiation - outgoing_radiation
        radiative_heating = (net_radiation / heat_capacity) * self.time_step_seconds

        self.update_ocean_temperature()
        self.temperature_celsius += radiative_heating
        self.temperature_celsius = gaussian_filter(self.temperature_celsius, sigma=0.5)
        self.temperature_celsius = np.clip(self.temperature_celsius, -80.0, 50.0)
        
        # Debug temperature changes
        temp_diff = self.temperature_celsius - initial_temp
        print(f"Temperature changes - Min: {np.min(temp_diff):.3f}°C, "
              f"Max: {np.max(temp_diff):.3f}°C, "
              f"Mean: {np.mean(temp_diff):.3f}°C")

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
        min_gray = self.altitude.min()  # Should be around 72
        min_gray = -1098
        max_gray = self.altitude.max()  # Should be around 232
        #sea_level = (max_gray + min_gray) // 2  # Should be around 152
        sea_level = 113
        
        # Create masks based on raw grayscale values
        ocean = self.altitude <= sea_level
        land = self.altitude > sea_level
        
        # Ocean (blue)
        rgb[ocean] = [0, 0, 255]
        
        # Land areas
        if np.any(land):
            land_colors = np.array([
                [4, 97, 2],    # Dark Green (lowest elevation)
                [43, 128, 42],   # Light Green
                [94, 168, 93],   # Light yellow
                [235, 227, 87],   # Yellow
                [171, 143, 67],   # Brown
                [107, 80, 0],   # Dark Brown
                [227, 227, 227]     # Gray (highest elevation)
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

    def map_temperature_to_color(self, data_normalized):
        """Map temperature to color gradient"""
        breakpoints = [
            (0.0, (0, 0, 128)),      # Deep blue (very cold)
            (0.15, (0, 128, 255)),   # Light blue
            (0.30, (128, 255, 255)), # Cyan
            (0.45, (255, 255, 128)), # Light yellow
            (0.60, (255, 200, 0)),   # Yellow-orange
            (0.75, (255, 128, 0)),   # Orange
            (1.0, (128, 0, 0))       # Deep red (very hot)
        ]

        rgb_array = np.zeros((data_normalized.shape[0], data_normalized.shape[1], 3), dtype=np.uint8)

        for i in range(len(breakpoints) - 1):
            val0, color0 = breakpoints[i]
            val1, color1 = breakpoints[i + 1]
            mask = (data_normalized >= val0) & (data_normalized <= val1)
            data_segment = (data_normalized[mask] - val0) / (val1 - val0)
            for c in range(3):
                rgb_array[mask, c] = (data_segment * (color1[c] - color0[c]) + color0[c]).astype(np.uint8)

        return rgb_array

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

    def update_wind(self):
        """Update wind components dynamically, incorporating temperature gradients, time step, and momentum."""
        # Constants for each cell's wind strength
        trade_wind_strength = 5.0
        westerlies_strength = 7.0
        polar_strength = 3.0
        
        # Define the cell boundaries
        hadley_boundary = 30
        ferrel_boundary = 60

        # Calculate temperature gradients
        temp_gradient_y, temp_gradient_x = np.gradient(self.temperature_celsius)
        
        # Recalculate wind patterns for each cell
        for y, lat in enumerate(np.linspace(90, -90, self.map_height)):
            abs_lat = abs(lat)

            # Hadley Cell (0° to 30°) - Trade Winds
            if abs_lat < hadley_boundary:
                intensity = np.sin(np.deg2rad(abs_lat * 3)) * trade_wind_strength
                self.u[y, :] = -intensity  # East to West
                self.v[y, :] = -intensity * 0.5 * np.sign(lat)  # Southward in Northern Hemisphere, Northward in Southern Hemisphere

            # Ferrel Cell (30° to 60°) - Westerlies
            elif abs_lat < ferrel_boundary:
                cell_lat = abs_lat - hadley_boundary
                intensity = np.sin(np.deg2rad(cell_lat * 3)) * westerlies_strength
                self.u[y, :] = intensity * np.sign(lat)  # West to East in both hemispheres
                self.v[y, :] = np.sign(lat) * intensity * 0.2  # Poleward flow

            # Polar Cell (60° to 90°) - Polar Easterlies
            else:
                cell_lat = abs_lat - ferrel_boundary
                intensity = (cell_lat / 30) * polar_strength
                self.u[y, :] = -intensity  # East to West
                self.v[y, :] = intensity * 0.1 * np.sign(lat)  # Slight equatorward flow

        # Apply temperature gradient effects scaled by time step
        self.u += 0.1 * temp_gradient_x * self.time_step_seconds
        self.v += 0.1 * temp_gradient_y * self.time_step_seconds

        # Introduce momentum by blending with previous wind
        momentum_factor = 0.9
        self.u = momentum_factor * self.u + (1 - momentum_factor) * self.u
        self.v = momentum_factor * self.v + (1 - momentum_factor) * self.v

        # Reapply convergence zones
        for y, lat in enumerate(np.linspace(90, -90, self.map_height)):
            # Intertropical Convergence Zone near equator
            if abs(lat) < 5:
                self.u[y, :] *= 0.2
                self.v[y, :] *= 0.2
            # Horse latitudes (30°N/S)
            if 28 < abs(lat) < 32:
                self.u[y, :] *= 0.3
                self.v[y, :] *= 0.3

        # Reapply land-sea differences
        land_mask = self.elevation > 0
        ocean_mask = self.elevation <= 0
        self.u[ocean_mask] *= 1.2
        self.v[ocean_mask] *= 1.2
        self.u[land_mask] *= 0.8
        self.v[land_mask] *= 0.8

        # Apply random perturbations for natural variability
        random_perturbation = np.random.normal(0, 0.001, self.u.shape)
        self.u += random_perturbation * self.time_step_seconds
        self.v += random_perturbation * self.time_step_seconds

        # Apply smoothing
        self.u = gaussian_filter(self.u, sigma=2.0)
        self.v = gaussian_filter(self.v, sigma=2.0)

        # Update normalized wind speed
        wind_speed = np.sqrt(self.u**2 + self.v**2)
        self.wind_speed_normalized = self.normalize_data(wind_speed)


    def update_pressure(self):
        """Update atmospheric pressure, ensuring realistic values and incorporating time step."""
        
        # Constants
        R = 287.05  # Gas constant for dry air (J/(kg·K))
        g = 9.81    # Gravitational acceleration (m/s²)

        # Convert temperature to Kelvin
        T = self.temperature_celsius + 273.15

        # Calculate base pressure using the hypsometric equation
        base_pressure = self.P0 * np.exp(-g * self.elevation / (R * T))

        # Calculate pressure change based on temperature and altitude
        # Use a small factor to ensure gradual changes
        pressure_change_factor = 0.0001  # Adjust this factor to control the rate of change
        pressure_change = (base_pressure - self.pressure) * pressure_change_factor * self.time_step_seconds

        # Update pressure
        self.pressure += pressure_change
        self.pressure_normalized = self.normalize_data(self.pressure)

    def draw_wind_vectors(self):
        """Draw wind vectors on canvas."""
        step = max(self.map_width, self.map_height) // 40
        scale = 0.3  # Adjust scale for better visualization

        x_indices, y_indices = np.meshgrid(
            np.arange(0, self.map_width, step),
            np.arange(0, self.map_height, step)
        )

        u_sampled = self.u[y_indices, x_indices]
        v_sampled = self.v[y_indices, x_indices]

        x_coords = x_indices.flatten()
        y_coords = y_indices.flatten()
        dx = u_sampled.flatten() * scale
        dy = -v_sampled.flatten() * scale  # Negative due to GUI coordinate system

        for x, y, delta_x, delta_y in zip(x_coords, y_coords, dx, dy):
            self.canvas.create_line(x, y, x + delta_x, y + delta_y, arrow=tk.LAST, fill='white')

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
            if self.selected_layer.get() == "Elevation":
                display_data = self.map_to_grayscale(self.elevation_normalized)
            elif self.selected_layer.get() == "Altitude":
                display_data = self.map_altitude_to_color(self.elevation)
            elif self.selected_layer.get() == "Temperature":
                display_data = self.map_temperature_to_color(self.temperature_normalized)
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

                # Draw additional elements if needed (wind vectors, etc.)
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

    def initialize_temperature(self):
        """Initialize temperature field with improved baseline for real-world values."""
        # Constants
        S0 = 1361  # Solar constant in W/m²
        albedo = 0.3  # Earth's average albedo
        sigma = 5.670374419e-8  # Stefan-Boltzmann constant
        lapse_rate = 6.5  # K per km
        greenhouse_effect = 33  # K, consider making this dynamic
        humidity_effect = 5  # Additional warming due to water vapor, adjust as needed

        # Generate coordinate arrays if not already done
        if not hasattr(self, 'latitudes_rad'):
            latitudes = np.linspace(90, -90, self.map_height)
            longitudes = np.linspace(-180, 180, self.map_width)
            self.latitude, self.longitude = np.meshgrid(latitudes, longitudes, indexing='ij')
            self.latitudes_rad = np.deg2rad(self.latitude)

        S_avg = S0 / 4
        cos_phi = np.clip(np.cos(self.latitudes_rad), 0, None)
        S_lat = S_avg * cos_phi
        T_eff = (((1 - albedo) * S_lat) / sigma) ** 0.25

        # Adjust the baseline temperature to better match real-world values
        baseline_adjustment = 40  # Increase this value to raise the overall temperature
        T_surface = T_eff + greenhouse_effect + humidity_effect + baseline_adjustment

        # Calculate base temperature without elevation effects
        self.temperature_celsius = T_surface - 273.15  # Convert to Celsius

        # Apply elevation effects using lapse rate
        altitude_km = np.maximum(self.elevation, 0) / 1000.0
        delta_T_altitude = -lapse_rate * altitude_km
        self.temperature_celsius += delta_T_altitude
        self.temperature_celsius = np.clip(self.temperature_celsius, -80.0, 50.0)

        # Normalize temperature for display
        self.temperature_normalized = self.normalize_data(self.temperature_celsius)

        # Initialize pressure after temperature
        self.initialize_pressure()

        # Initialize ocean temperatures separately
        is_ocean = self.elevation <= 0
        self.temperature_celsius[is_ocean] = np.clip(
            20 - (np.abs(self.latitude[is_ocean]) / 90) * 40,  # Temperature decreases with latitude
            -2,  # Minimum ocean temperature
            30   # Maximum ocean temperature
        )
        
        self.temperature_normalized = self.normalize_data(self.temperature_celsius)

    def initialize_pressure(self):
        """Initialize pressure field"""
        # Constants
        R = 287.05  # Gas constant for dry air (J/(kg·K))
        g = 9.81    # Gravitational acceleration (m/s²)

        # Convert temperature to Kelvin
        T = self.temperature_celsius + 273.15

        # Calculate pressure based on temperature and altitude
        # Using hypsometric equation: P = P0 * exp(-g*h/(R*T))
        self.pressure = self.P0 * np.exp(-g * self.elevation / (R * T))
        self.pressure_normalized = self.normalize_data(self.pressure)
        

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    app = SimulationApp()
    app.root.mainloop()  # Add this line to start the Tkinter event loop
