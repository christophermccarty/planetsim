import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tkinter as tk
from tqdm import tqdm
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import Menu
from generation import (generate_terrain, overlay_large_features, add_crater_optimized, generate_craters,
                        apply_minimal_smoothing, scale_to_grayscale, calculate_temperature, apply_atmospheric_model)


class PlanetSim:
    def __init__(self):
        self.current_terrain = None
        self.temperatures = None
        self.TILE_SIZE = 10
        self.canvas_size = 500
        self.terrain_size = 500
        self.TILE_SIZE = self.canvas_size / self.terrain_size
        self.window = tk.Tk()
        self.window.title("Planet Simulator")
        self.canvas = tk.Canvas(self.window, width=500, height=500)
        self.info_label = tk.Label(self.window)
        self.info_label.pack()

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

    def generate_terrain_from_menu(self):
        # Generate the terrain
        self.current_terrain = generate_terrain(
            int(self.params['width']), int(self.params['height']), scale=self.params['scale'],
            octaves=int(self.params['octaves']), persistence=self.params['persistence'], lacunarity=self.params['lacunarity']
        )
        if self.current_terrain is None:
            print("Error: generate_terrain returned None")
            return

        # Add craters to the terrain
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

    def load_image_as_terrain(self, image_path):
        # Load the image
        image = Image.open(image_path).convert('L')
        image_np = np.array(image)

        # Print the grayscale range
        print("Grayscale range:", image_np.min(), image_np.max())

        # Resize the canvas to fit the image
        self.canvas.config(width=image.width, height=image.height)

        # Scale the grayscale values to the range [0, 247]
        image_np_scaled = (image_np - image_np.min()) * (247 / (image_np.max() - image_np.min()))

        # Convert the scaled grayscale values to elevations
        self.current_terrain = -10984 + image_np_scaled * (8848 - (-10984)) / 247

        print("Loading terrain...")
        print("Elevation range:", self.current_terrain.min(), self.current_terrain.max())

        # Display the terrain map on the canvas
        self.update_terrain_display(self.current_terrain)

        # Calculate the latitude for each row in the terrain
        latitudes = np.linspace(-90, 90, self.current_terrain.shape[0])

        # Generate the initial temperatures
        self.temperatures = calculate_temperature(self.current_terrain, latitudes[:, np.newaxis])

    def update_terrain_display(self, terrain):
        # Clear the canvas
        self.canvas.delete('all')

        # Create a colormap to map terrain heights to colors
        cmap = plt.get_cmap('terrain')

        # Normalize the terrain heights to the range [0, 1]
        terrain_normalized = (terrain - terrain.min()) / (terrain.max() - terrain.min())

        # Create a grid of rectangles on the canvas
        for i in tqdm(range(terrain.shape[0]), desc='Displaying terrain'):
            for j in range(terrain.shape[1]):
                # Get the color for this tile from the colormap
                color = cmap(terrain_normalized[i, j])

                # Convert the color from RGB to hexadecimal
                color = '#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255))

                # Create a rectangle on the canvas for this tile
                self.canvas.create_rectangle(j, i, j+1, i+1, fill=color, outline="")

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
        if self.current_terrain is not None:
            self.update_terrain_display(self.current_terrain)
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
        cmap = mcolors.LinearSegmentedColormap.from_list("temp_map", ["blue", "red"])

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
        if self.temperatures is None or self.current_terrain is None:
            self.temperatures = self.update_frame(self.params)

        # Calculate the tile's index based on the mouse's position
        tile_x = int(event.x // self.TILE_SIZE)
        tile_y = int(event.y // self.TILE_SIZE)

        # Ensure tile_x and tile_y do not exceed the size of the terrain
        tile_x = min(tile_x, self.current_terrain.shape[1] - 1)
        tile_y = min(tile_y, self.current_terrain.shape[0] - 1)

        # Retrieve the tile's temperature and elevation
        temperature = round(self.temperatures[tile_y][tile_x], 2)
        elevation = round(self.current_terrain[tile_y][tile_x], 2)

        # Calculate the tile's latitude and longitude
        latitude = round(tile_y / self.current_terrain.shape[0] * 180 - 90, 2)
        longitude = round(tile_x / self.current_terrain.shape[1] * 360 - 180, 2)

        # Update the window title with the tile's information
        self.window.title(
            f"Temperature: {temperature}, Elevation: {elevation}, Latitude: {latitude}, Longitude: {longitude}")

    def main(self):
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.pack(side='right', expand=True)
        # Create a menu bar
        menubar = Menu(self.window)
        # Create a view menu
        view_menu = Menu(menubar, tearoff=0)
        view_menu.add_command(label="Terrain", command=self.show_terrain_entries)
        view_menu.add_command(label="Temperature", command=lambda: self.view_temperature(self.window, self.canvas))
        # Add the view menu to the menu bar
        menubar.add_cascade(label="View", menu=view_menu)
        # Add the menu bar to the window
        self.window.config(menu=menubar)
        self.sidebar, self.entries = self.create_sidebar(self.update_frame)  # Initialize self.entries and self.sidebar
        self.params = self.load_settings()  # Load settings from the JSON file
        # Load the image as the tilemap
        self.load_image_as_terrain(r"C:\dev\planetsim\images\World_elevation_map.png")
        # Generate the initial temperatures
        self.view_temperature(self.window, self.canvas)
        self.window.mainloop()


if __name__ == "__main__":
    sim = PlanetSim()
    sim.main()