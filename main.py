import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time
import json
import multiprocessing
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
        # Initialize all variables
        self.map_size = 512
        self.map_width = None  # Will be set when image is loaded
        self.map_height = None # Will be set when image is loaded
        
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
        self.previous_temperature_celsius = 0
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

        # Initialize temperature field
        self.initialize_temperature()

        # Initialize wind field
        self.initialize_wind()

        # Initialize ocean currents
        self.initialize_ocean_currents()

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

                # Initialize grid spacing based on map dimensions
                earth_circumference = 40075000  # meters
                self.grid_spacing_x = earth_circumference / self.map_width
                self.grid_spacing_y = earth_circumference / self.map_height

                # Convert image to numpy array
                img_array = np.array(elevation_image)
                img_array = img_array.astype(int)
                if len(img_array.shape) > 2:  # If image is RGB/RGBA
                    img_array = img_array[:, :, 0]  # Take first channel

                # Initialize altitude array from image data
                self.altitude = img_array.copy()

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
        print("\nWind Update Debug:")
        
        # Calculate initial wind speed for debugging
        initial_wind_speed = np.sqrt(self.u**2 + self.v**2)
        print(f"Initial Wind Speed Range: {np.min(initial_wind_speed):.2f} to {np.max(initial_wind_speed):.2f} m/s")
        
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

        # Final wind speed for debugging
        final_wind_speed = np.sqrt(self.u**2 + self.v**2)
        print(f"Final Wind Speed Range: {np.min(final_wind_speed):.2f} to {np.max(final_wind_speed):.2f} m/s")

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
        """Update temperature fields for both land and ocean using energy balance."""
        print(f"Starting temperature update at step {self.time_step}")

        # Constants
        sigma = 5.670374419e-8  # Stefan-Boltzmann constant
        albedo_land = 0.3
        albedo_ocean = 0.06
        emissivity = 0.9
        heat_capacity_land = 1e6  # J/m²/K
        heat_capacity_ocean = 4e6  # J/m²/K

        # Solar radiation calculation
        S0 = 1361  # Solar constant in W/m²
        S_avg = S0 / 4
        solar_factor = 0.5 * (1 + np.cos(self.latitudes_rad))
        S_lat = S_avg * np.clip(solar_factor, 0, None)

        # Incoming radiation
        incoming_radiation = np.zeros_like(self.temperature_celsius)
        is_land = self.elevation > 0
        is_ocean = ~is_land

        incoming_radiation[is_land] = (1 - albedo_land) * S_lat[is_land]
        incoming_radiation[is_ocean] = (1 - albedo_ocean) * S_lat[is_ocean]

        # Outgoing longwave radiation
        T_kelvin = self.temperature_celsius + 273.15
        outgoing_radiation = emissivity * sigma * T_kelvin ** 4

        # Net radiation balance
        net_radiation = incoming_radiation - outgoing_radiation

        # Update temperature
        delta_temp_land = (net_radiation / heat_capacity_land) * self.time_step_seconds
        delta_temp_ocean = (net_radiation / heat_capacity_ocean) * self.time_step_seconds

        self.temperature_celsius[is_land] += delta_temp_land[is_land]
        self.temperature_celsius[is_ocean] += delta_temp_ocean[is_ocean]

        # Smooth temperature field for numerical stability
        self.temperature_celsius = gaussian_filter(self.temperature_celsius, sigma=1.0)

        # Normalize temperature for visualization
        self.temperature_normalized = self.normalize_data(self.temperature_celsius)

        print(f"\nTemperature Update Debug:")
        print(f"  - Land Cells Updated: {np.sum(is_land)}")
        print(f"  - Ocean Cells Updated: {np.sum(is_ocean)}")
        print(f"  - Temperature Change Range (Land): {np.min(delta_temp_land[is_land]):.4f} to {np.max(delta_temp_land[is_land]):.4f} °C")
        print(f"  - Temperature Change Range (Ocean): {np.min(delta_temp_ocean[is_ocean]):.4f} to {np.max(delta_temp_ocean[is_ocean]):.4f} °C")
        print(f"  - Temperature Range After Update: Min: {np.min(self.temperature_celsius):.2f} °C, Max: {np.max(self.temperature_celsius):.2f} °C")

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

    def update_pressure(self):
        """Update atmospheric pressure based on physical processes."""
        # Constants
        R = 287.05  # Specific gas constant for dry air (J/(kg·K))
        g = 9.81    # Gravitational acceleration (m/s²)

        print("\nPressure Update Debug:")

        # 1. Calculate Air Density
        T_kelvin = self.temperature_celsius + 273.15
        air_density = self.pressure / (R * T_kelvin)

        # 2. Pressure Tendency due to Wind Divergence (Continuity Equation)
        div_u = np.gradient(self.u, axis=1) / self.grid_spacing_x
        div_v = np.gradient(self.v, axis=0) / self.grid_spacing_y
        divergence = div_u + div_v

        # 3. Limit Divergence to Prevent Excessive Pressure Changes
        divergence = np.clip(divergence, -1e-3, 1e-3)  # Adjust as needed

        # 4. Pressure Tendency Calculation
        pressure_tendency = -air_density * R * T_kelvin * divergence

        # 5. Pressure Tendency due to Temperature Changes (Thermal Effects)
        temp_change = self.temperature_celsius - self.previous_temperature_celsius
        thermal_pressure_tendency = air_density * R * temp_change / self.time_step_seconds

        # 6. Update Pressure with Limited Gradients
        total_pressure_tendency = pressure_tendency + thermal_pressure_tendency
        self.pressure += total_pressure_tendency * self.time_step_seconds

        # 7. Apply Periodic Boundary Conditions for X (longitude) and Reflective for Y (latitude)
        self.pressure = gaussian_filter(self.pressure, sigma=2.0)  # Enhanced smoothing

        # 8. Advect Pressure with Correct Boundary Conditions
        self.pressure = self.advect_field(self.pressure, self.u, self.v)

        # 9. Store Current Temperature for Next Iteration
        self.previous_temperature_celsius = self.temperature_celsius.copy()

        # 10. Ensure Pressure Remains within Realistic Bounds
        self.pressure = np.clip(self.pressure, 50000, 110000)

        # 11. Update Normalized Pressure for Visualization
        self.pressure_normalized = self.normalize_data(self.pressure)

        # Debug Output
        print(f"Pressure Range: {np.min(self.pressure):.2f} to {np.max(self.pressure):.2f} Pa")
        print(f"Max Pressure Gradient: {np.max(np.abs(np.gradient(self.pressure))):.2f} Pa/m")

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
            if self.selected_layer.get() == "Pressure":
                # Create pressure visualization
                pressure_rgba = self.map_pressure_to_color(self.pressure)
                
                # Create altitude overlay
                altitude_rgba = self.map_altitude_to_overlay(self.elevation)
                
                # Blend the two RGBA arrays
                display_data = self.blend_rgba_arrays(pressure_rgba, altitude_rgba)
                
                # Convert to PIL Image and display
                image = Image.fromarray(display_data)
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

    def blend_rgba_arrays(self, base_rgba, overlay_rgba):
        """Blend two RGBA arrays using alpha compositing."""
        # Convert to float for calculations
        base = base_rgba.astype(float) / 255
        overlay = overlay_rgba.astype(float) / 255
        
        # Extract alpha channels
        base_alpha = base[..., 3]
        overlay_alpha = overlay[..., 3]
        
        # Calculate resulting alpha
        out_alpha = overlay_alpha + base_alpha * (1 - overlay_alpha)
        
        # Calculate resulting colors
        out_colors = np.zeros_like(base)
        for i in range(3):  # For RGB channels
            out_colors[..., i] = (overlay[..., i] * overlay_alpha + 
                                 base[..., i] * base_alpha * (1 - overlay_alpha)) / np.maximum(out_alpha, 1e-8)
        
        # Set alpha channel
        out_colors[..., 3] = out_alpha
        
        # Convert back to uint8
        return (out_colors * 255).astype(np.uint8)

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

        # Initialize ocean temperatures separately
        is_ocean = self.elevation <= 0
        self.temperature_celsius[is_ocean] = np.clip(
            20 - (np.abs(self.latitude[is_ocean]) / 90) * 40,  # Temperature decreases with latitude
            -2,  # Minimum ocean temperature
            30   # Maximum ocean temperature
        )
        
        self.temperature_normalized = self.normalize_data(self.temperature_celsius)

        # Initialize pressure after temperature
        self.initialize_pressure()

        # Print initial temperature values
        print(f"Initial Temperature Min: {np.min(self.temperature_celsius)}, Max: {np.max(self.temperature_celsius)}")

    def initialize_pressure(self):
        """Initialize pressure field using temperature gradients and altitude."""
        # Constants
        R = 287.05  # Gas constant for dry air (J/(kg·K))
        g = 9.81    # Gravitational acceleration (m/s²)

        # Convert temperature to Kelvin
        T = self.temperature_celsius + 273.15

        # Use elevation above sea level
        elevation_above_sea_level = np.maximum(self.elevation, 0)

        # Apply the hypsometric equation: P = P0 * exp(-g*h/(R*T))
        self.pressure = self.P0 * np.exp(-g * elevation_above_sea_level / (R * T))

        # Smooth pressure to simulate horizontal mixing
        self.pressure = gaussian_filter(self.pressure, sigma=2.0)

        # Normalize pressure for visualization
        self.pressure_normalized = self.normalize_data(self.pressure)

        print(f"Initialized pressure. Min: {np.min(self.pressure):.2f}, Max: {np.max(self.pressure):.2f} Pa")

    def map_pressure_to_color(self, pressure_data):
        """
        Map pressure to a red-blue color gradient with altitude overlay.
        Red indicates low pressure, blue indicates high pressure.
        """
        # Create RGBA array (including alpha channel for transparency)
        rgba_array = np.zeros((self.map_height, self.map_width, 4), dtype=np.uint8)
        
        # Normalize pressure data to [0, 1]
        pressure_normalized = self.normalize_data(pressure_data)
        
        # Create pressure color gradient (red to blue)
        rgba_array[..., 0] = ((1 - pressure_normalized) * 255).astype(np.uint8)  # Red
        rgba_array[..., 2] = (pressure_normalized * 255).astype(np.uint8)        # Blue
        
        # Add green component for better visualization
        rgba_array[..., 1] = (np.minimum(pressure_normalized, 1 - pressure_normalized) * 255).astype(np.uint8)
        
        # Calculate alpha channel based on elevation
        elevation_normalized = self.normalize_data(self.elevation)
        
        # Make water areas more transparent
        is_water = self.elevation <= 0
        alpha = np.ones_like(elevation_normalized) * 255  # Start with full opacity
        alpha[is_water] = 128  # 50% transparency for water
        
        # Add elevation-based transparency for land
        land_alpha = (elevation_normalized * 127 + 128)  # Scale from 128-255
        alpha[~is_water] = land_alpha[~is_water]
        
        rgba_array[..., 3] = alpha.astype(np.uint8)
        
        return rgba_array

    def map_altitude_to_overlay(self, elevation):
        """Create a semi-transparent altitude overlay."""
        # Create RGBA array
        rgba = np.zeros((self.map_height, self.map_width, 4), dtype=np.uint8)
        
        # Get the actual grayscale range from the image
        min_gray = -1098
        sea_level = 113
        max_gray = self.altitude.max()
        
        # Create masks based on raw grayscale values
        ocean = self.altitude <= sea_level
        land = self.altitude > sea_level
        
        # Ocean (transparent blue)
        rgba[ocean] = [0, 0, 255, 64]  # Light blue with 25% opacity
        
        # Land areas with varying colors and transparency
        if np.any(land):
            land_colors = np.array([
                [4, 97, 2, 128],      # Dark Green (lowest elevation)
                [43, 128, 42, 128],   # Light Green
                [94, 168, 93, 128],   # Light yellow
                [235, 227, 87, 128],  # Yellow
                [171, 143, 67, 128],  # Brown
                [107, 80, 0, 128],    # Dark Brown
                [227, 227, 227, 128]  # Gray (highest elevation)
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
            rgba[land] = (start_colors * (1 - ratios[:, np.newaxis]) + 
                         end_colors * ratios[:, np.newaxis]).astype(np.uint8)
        
        return rgba

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

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    app = SimulationApp()
    app.root.mainloop()  # Add this line to start the Tkinter event loop