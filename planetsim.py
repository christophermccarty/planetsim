import json
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tkinter as tk
from tqdm import tqdm
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import Menu
from scipy.ndimage import convolve
from generation import (generate_terrain, overlay_large_features, add_crater_optimized, generate_craters,
                        apply_minimal_smoothing, scale_to_grayscale, calculate_temperature, apply_atmospheric_model,
                        flood_fill)


class PlanetSim:
    def __init__(self):
        self.current_terrain = None
        self.temperatures = None
        self.TILE_SIZE = 20
        self.canvas_size = 500
        self.terrain_size = 500
        self.TILE_SIZE = self.canvas_size / self.terrain_size
        self.window = tk.Tk()
        self.window.title("Planet Simulator")
        self.canvas = tk.Canvas(self.window, width=500, height=500)
        self.info_label = tk.Label(self.window)
        self.info_label.pack()
        self.current_ocean_map = None
        self.min_elevation = -420
        self.max_elevation = 8848


    def create_sidebar(self, frame_update_function):
        sidebar = tk.Frame(self.window, width=200, bg='gray')

        params = {
            "width": 500,
            "height": 500,
            "sigma": 2
        }

        terrain_params = {
            "scale": 200.0,
            "octaves": 2,
            "persistence": 0.7,
            "lacunarity": 2.0,
            "strength": 0.5,
            "scale_factor": 4,
            "num_small_craters": 500,
            "num_large_craters": 10,
            "max_small_radius": 30,
            "max_large_radius": 120,
            "max_depth": 0.3
        }

        entries = {}
        for param, value in params.items():
            frame = tk.Frame(sidebar, bg='gray')
            frame.pack(fill='x', padx=5, pady=5)
            label = tk.Label(frame, text=param.capitalize(), bg='gray')
            label.pack(side='left')
            entry = tk.Entry(frame)
            entry.pack(side='right', expand=True)
            entry.insert(0, str(value))
            entries[param] = entry

        self.terrain_entries = {}
        for param, value in terrain_params.items():
            frame = tk.Frame(sidebar, bg='gray')
            frame.pack(fill='x', padx=5, pady=5)
            label = tk.Label(frame, text=param.capitalize(), bg='gray')
            label.pack(side='left')
            entry = tk.Entry(frame)
            entry.pack(side='right', expand=True)
            entry.insert(0, str(value))
            self.terrain_entries[param] = entry
            frame.pack_forget()  # Hide the terrain entries initially

        def on_generate_clicked(self):
            self.save_settings(entries)
            self.params = self.load_settings(entries)  # Load the updated settings
            frame_update_function(self.params)

        generate_button = tk.Button(sidebar, text="Generate Terrain", command=lambda: on_generate_clicked(self))
        generate_button.pack(pady=5)
        exit_button = tk.Button(sidebar, text="Exit", command=lambda: self.exit_program(self.window, entries))
        exit_button.pack(pady=5)

        return sidebar, entries

    def show_terrain_entries(self):
        self.sidebar.pack(fill='y', side='left', padx=5, pady=5)  # Pack the sidebar here
        for frame in self.terrain_entries.values():
            frame.pack()  # Show the terrain entries

        # Display the terrain map on the canvas
        if self.current_terrain is not None:
            ocean_map = self.classify_ocean(self.current_terrain)
            self.update_terrain_display(self.current_terrain, ocean_map)
        else:
            print("No terrain map available.")

    def generate_terrain_from_menu(self):
        # Generate the terrain
        print("Generating terrain...")
        self.current_terrain = generate_terrain(
            int(self.params['width']), int(self.params['height']), scale=self.params['scale'],
            octaves=int(self.params['octaves']), persistence=self.params['persistence'], lacunarity=self.params['lacunarity']
        )
        if self.current_terrain is None:
            print("Error: generate_terrain returned None")
            return

        # Add craters to the terrain
        print("Generating craters...")
        self.current_terrain = generate_craters(
            self.current_terrain, num_small_craters=int(self.params['num_small_craters']),
            num_large_craters=int(self.params['num_large_craters']),
            max_small_radius=self.params['max_small_radius'], max_large_radius=self.params['max_large_radius'],
            max_depth=self.params['max_depth']
        )
        if self.current_terrain is None:
            print("Error: generate_craters returned None")
            return

        # Update the terrain display
        self.update_terrain_display(self.current_terrain)

    def load_terrain(self, image_path, min_elevation, max_elevation):
        # Open the image file
        with rasterio.open(image_path) as src:
            # Read the data into a numpy array
            image_data = src.read(1)

        # Update the canvas size to match the image size
        self.canvas.config(width=image_data.shape[1], height=image_data.shape[0])

        # Store the original image data
        self.original_image = image_data

        # Print the range of grayscale values
        print(f"Grayscale range: {image_data.min()} - {image_data.max()}")

        # Calculate the scale factor to map pixel values to the elevation range
        scale_factor = (max_elevation - min_elevation) / (image_data.max() - image_data.min() + 1)

        # Initialize an elevation matrix
        elevation_matrix = np.zeros_like(image_data, dtype=np.float32)

        # Map the grayscale values to elevations
        total_pixels = image_data.shape[0] * image_data.shape[1]
        with tqdm(total=total_pixels, desc='Loading terrain') as pbar:
            for i in range(image_data.shape[0]):
                for j in range(image_data.shape[1]):
                    # Map the pixel value to the elevation range
                    elevation = (image_data[i][j] - image_data.min()) * scale_factor + min_elevation
                    # If the elevation is less than 1 and greater than -1, round it up to 1
                    if -2 < elevation < 1:
                        elevation = 1
                    # Truncate the elevation to 1 decimal place
                    elevation = np.around(elevation, 1)
                    elevation_matrix[i][j] = elevation
                    pbar.update(1)

        return elevation_matrix

    def update_terrain_display(self, terrain, ocean_map=None):
        try:
            # Get the 'terrain' colormap
            terrain_cmap = plt.get_cmap('terrain')

            # Create a custom colormap that excludes the bottom 25% of color values
            custom_cmap = mcolors.LinearSegmentedColormap.from_list(
                'custom_cmap',
                terrain_cmap(np.linspace(0.25, 1, 256))
            )

            # Normalize the terrain values to the range of the colormap
            norm = plt.Normalize(vmin=self.min_elevation, vmax=self.max_elevation)

            # Calculate the colors for the terrain using vectorized operations
            terrain_colors = terrain_cmap(norm(terrain))
            terrain_colors = np.array(terrain_colors)  # Convert terrain_colors to a numpy array
            terrain_colors[terrain > 0] = custom_cmap(norm(terrain[terrain > 0]))

            # Store the terrain colors
            self.terrain_colors = terrain_colors

            # If ocean_map is provided, set the color of ocean tiles to blue
            if ocean_map is not None:
                print(f"ocean_map shape: {ocean_map.shape}, terrain shape: {terrain.shape}")  # Debug print
                terrain_colors[ocean_map] = [0, 0, 1, 1]  # Blue color for ocean tiles

            # Convert the colors from RGBA to RGB and scale to [0, 255]
            colors = (terrain_colors[:, :, :3] * 255).astype(np.uint8)

            # Create an image from the colors array
            image = Image.fromarray(colors)

            # Convert the PIL.Image.Image object to a tkinter.PhotoImage object
            photo_image = ImageTk.PhotoImage(image)

            # Clear the canvas
            self.canvas.delete('all')

            # Draw the PhotoImage on the canvas
            self.canvas.create_image(0, 0, image=photo_image, anchor='nw')

            # Keep a reference to the PhotoImage to prevent it from being garbage collected
            self.photo_image = photo_image

            print("Canvas updated")

        except Exception as e:
            print(f"An error occurred: {e}")

    def classify_ocean(self, elevation_matrix, ocean_level=394.66, neighbor_threshold=4):
        # Initialize the ocean map
        ocean_map = np.zeros_like(elevation_matrix, dtype=bool)

        # Classify ocean tiles
        for i in tqdm(range(elevation_matrix.shape[0]), desc='Classifying ocean tiles'):
            for j in range(elevation_matrix.shape[1]):
                # Mark tiles as ocean if their elevation is at or below the ocean level
                ocean_map[i][j] = elevation_matrix[i][j] < ocean_level

        # Create a copy of the ocean map to modify
        new_ocean_map = ocean_map.copy()

        # Iterate over the ocean map again
        for i in range(1, ocean_map.shape[0] - 1):
            for j in range(1, ocean_map.shape[1] - 1):
                # If the current tile is not an ocean tile
                if not ocean_map[i][j]:
                    # Count the number of neighboring ocean tiles
                    neighbor_count = np.sum(ocean_map[i - 1:i + 2, j - 1:j + 2]) - ocean_map[i][j]

                    # If the number of neighboring ocean tiles is greater than or equal to the threshold
                    if neighbor_count >= neighbor_threshold:
                        # Mark the current tile as an ocean tile
                        new_ocean_map[i][j] = True

        return new_ocean_map

    def update_frame(self, params):
        # Generate the terrain
        self.current_terrain = generate_terrain(
            int(params['width']), int(params['height']), scale=params['scale'],
            octaves=int(params['octaves']), persistence=params['persistence'], lacunarity=params['lacunarity']
        )
        if self.current_terrain is None:
            print("Error: generate_terrain returned None")
            return

        # Add craters to the terrain
        self.current_terrain = generate_craters(
            self.current_terrain, num_small_craters=int(params['num_small_craters']),
            num_large_craters=int(params['num_large_craters']),
            max_small_radius=params['max_small_radius'], max_large_radius=params['max_large_radius'],
            max_depth=params['max_depth']
        )
        if self.current_terrain is None:
            print("Error: generate_craters returned None")
            return

        # Calculate the latitude for each row in the terrain
        latitudes = np.linspace(-90, 90, self.current_terrain.shape[0])

        # Calculate the temperature
        self.temperatures = calculate_temperature(self.current_terrain, latitudes[:, np.newaxis])
        if self.temperatures is None:
            print("Error: calculate_temperature returned None")
            return

        # Update the terrain display
        self.update_terrain_display(self.current_terrain)

        return self.temperatures

    def create_image(self, terrain):
        terrain_scaled = scale_to_grayscale(terrain).astype('uint8')
        img = Image.fromarray(terrain_scaled, 'L')
        return img

    def load_settings(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
        except FileNotFoundError:
            print("Settings file not found. Using default settings.")
            settings = {}

        # Ensure all necessary keys are present in settings
        for param, entry in self.entries.items():
            if param not in settings:
                print(f"Key '{param}' not found in settings. Using default value.")
                settings[param] = float(entry.get())
            elif entry.winfo_exists():
                entry.delete(0, tk.END)
                entry.insert(0, str(settings[param]))

        return settings

    def view_terrain(self):
        if self.current_terrain is not None and self.current_ocean_map is not None:
            self.update_terrain_display(self.current_terrain, self.current_ocean_map)
            print("Terrain viewed")
        else:
            print("No terrain map available.")

    def view_temperature(self, window, canvas):
        # Ensure self.temperatures is not None
        if self.temperatures is None:
            self.temperatures = self.update_frame(self.params)
            if self.temperatures is None:
                print("Error: Failed to generate temperatures")
                return

        # Create a color map from blue (cooler) to red (hotter)
        cmap = plt.get_cmap('coolwarm')

        # Clear the canvas
        canvas.delete('all')

        # Normalize the temperatures to the range [0, 1]
        normalized_temperatures = (self.temperatures - self.temperatures.min()) / (
                self.temperatures.max() - self.temperatures.min())

        # Convert the normalized temperatures to colors
        rgba_colors = cmap(normalized_temperatures)

        # Flatten the rgba_colors array
        flattened_rgba_colors = rgba_colors.reshape(-1, rgba_colors.shape[-1])

        # Convert the RGBA colors to hexadecimal color strings
        hex_colors = ['#%02x%02x%02x' % (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in
                      flattened_rgba_colors]

        # Reshape hex_colors to match the original shape of rgba_colors
        hex_colors = np.array(hex_colors).reshape(rgba_colors.shape[0], rgba_colors.shape[1])

        # Draw the tiles on the canvas
        for i in tqdm(range(self.current_terrain.shape[0]), desc='Drawing tiles'):
            for j in range(self.current_terrain.shape[1]):
                canvas.create_rectangle(j, i, j + 1, i + 1, fill=hex_colors[i][j], outline="")

    def on_mouse_move(self, event):
        # Check if self.current_terrain is None
        if self.current_terrain is None:
            print("Warning: current_terrain is None")
            return

        # Calculate the tile's index based on the mouse's position
        tile_x = int(event.x // self.TILE_SIZE)
        tile_y = int(event.y // self.TILE_SIZE)

        # Ensure tile_x and tile_y do not exceed the size of the terrain
        tile_x = min(tile_x, self.current_terrain.shape[1] - 1)
        tile_y = min(tile_y, self.current_terrain.shape[0] - 1)

        # Retrieve the tile's elevation and truncate to 1 decimal place
        elevation = np.around(self.current_terrain[tile_y][tile_x], 1)

        # Calculate the tile's latitude and longitude
        latitude = round(90 - tile_y / self.current_terrain.shape[0] * 180, 2)  # Adjusted latitude calculation
        longitude = round(tile_x / self.current_terrain.shape[1] * 360 - 180, 2)

        # Retrieve the grayscale pixel value
        grayscale_pixel_value = self.original_image[tile_y][tile_x]

        # Calculate the altitude and truncate to 1 decimal place
        altitude = np.around(max(0, (grayscale_pixel_value - 8068) / (self.original_image.max() - 8068) * 8848), 1)

        # If the altitude is greater than 0 and less than 1, round it up to 1
        if 0 < altitude < 1:
            altitude = 1

        # Update the window title with the tile's information
        self.window.title(
            f"Elevation: {elevation}, Altitude: {altitude}, Latitude: {latitude}, Longitude: {longitude}, "
            f"Grayscale Pixel Value: {grayscale_pixel_value}")

    def save_settings(self, entries):
        settings = {param: float(entry.get()) for param, entry in entries.items() if entry.winfo_exists()}
        print("Settings to save:", settings)  # Debug print
        with open('settings.json', 'w') as f:
            json.dump(settings, f, indent=4)

    def exit_program(self, window, entries):
        print("Exit button clicked")  # Debug print
        self.save_settings(entries)
        print("Settings saved")  # Debug print
        window.destroy()

    def main(self):
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.pack(side='right', expand=True)
        # Create a menu bar
        menubar = Menu(self.window)
        # Create a view menu
        view_menu = Menu(menubar, tearoff=0)
        view_menu.add_command(label="Terrain", command=self.show_terrain_entries)
        view_menu.add_command(label="New Terrain", command=self.generate_terrain_from_menu)
        view_menu.add_command(label="Temperature", command=lambda: self.view_temperature(self.window, self.canvas))
        # Add the view menu to the menu bar
        menubar.add_cascade(label="View", menu=view_menu)
        # Add the menu bar to the window
        self.window.config(menu=menubar)
        self.sidebar, self.entries = self.create_sidebar(self.update_frame)  # Initialize self.entries and self.sidebar
        self.params = self.load_settings()  # Load settings from the JSON file
        self.elevation_matrix = self.load_terrain(
            r"D:\dev\planetsim\images\16_bit_dem_small_1280.tif", -420, 8848)
        print(f"Terrain loaded with shape {self.elevation_matrix.shape}")
        self.current_terrain = self.elevation_matrix  # Set self.current_terrain to the result of load_terrain
        self.ocean_map = self.classify_ocean(self.elevation_matrix)
        print(f"Ocean map classified with shape {self.ocean_map.shape}")
        # Call the update_terrain_display function
        self.update_terrain_display(self.current_terrain, self.ocean_map)
        # Display the terrain map
        self.view_terrain()
        self.window.mainloop()


if __name__ == "__main__":
    sim = PlanetSim()
    sim.main()