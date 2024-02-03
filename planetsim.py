import cProfile
import pstats
import tkinter as tk
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from opensimplex import OpenSimplex
from tkinter import ttk
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# Global variable to keep track of whether settings have been saved
settings_saved = False


def generate_terrain(width, height, seed=None, scale=200.0, octaves=1, persistence=0.7, lacunarity=2.0):
    if seed is None:
        seed = np.random.randint(0, 100)
    simplex = OpenSimplex(seed)

    max_amp = 0
    amp = 1
    freq = 3

    terrain = np.zeros((height, width))

    # Wrap the outer loop with tqdm for a progress bar
    for o in tqdm(range(octaves), desc='Generating terrain'):
        for i in range(height):
            for j in range(width):
                terrain[i][j] += simplex.noise2(i / scale * freq, j / scale * freq) * amp
        max_amp += amp
        amp *= persistence
        freq *= lacunarity

    # Normalizing the terrain
    terrain = (terrain + max_amp) / (2 * max_amp)

    return terrain

def overlay_large_features(terrain, seed=None, strength=0.5, scale_factor=4):
    if seed is None:
        seed = np.random.randint(0, 100)
    simplex = OpenSimplex(seed)
    height, width = terrain.shape
    large_scale = max(width, height) / scale_factor

    for i in range(height):
        for j in range(width):
            terrain[i][j] += simplex.noise2(i / large_scale, j / large_scale) * strength

    return terrain

def add_crater_optimized(terrain, cx, cy, radius, depth):
    height, width = terrain.shape
    y, x = np.ogrid[-cx:height-cx, -cy:width-cy]
    mask = x**2 + y**2 <= radius**2

    # Apply crater effect within the masked area
    distance_from_center = np.sqrt(x**2 + y**2)[mask]
    delta = depth * (1 - distance_from_center / radius)
    terrain[mask] -= delta
    terrain[terrain < 0] = 0  # Ensure terrain height doesn't go below 0

def generate_craters(terrain, num_small_craters=15, num_large_craters=3, max_small_radius=10, max_large_radius=40,
                     max_depth=0.1):
    total_craters = num_small_craters + num_large_craters
    with tqdm(total=total_craters, desc='Adding craters') as pbar:
        # Add small craters
        for _ in range(num_small_craters):
            height, width = terrain.shape
            # Random center, radius, and depth for each crater
            cx, cy = np.random.randint(0, height), np.random.randint(0, width)
            radius = np.random.uniform(0, max_small_radius)
            depth = np.random.uniform(0, max_depth)
            add_crater_optimized(terrain, cx, cy, radius, depth)
            pbar.update()

        # Add large craters
        for _ in range(num_large_craters):
            height, width = terrain.shape
            # Random center, radius, and depth for each crater
            cx, cy = np.random.randint(0, height), np.random.randint(0, width)
            radius = np.random.uniform(0, max_large_radius)
            depth = np.random.uniform(0, max_depth)
            add_crater_optimized(terrain, cx, cy, radius, depth)
            pbar.update()

    return terrain

def apply_minimal_smoothing(terrain, sigma=1):
    return gaussian_filter(terrain, sigma=sigma)

def scale_to_grayscale(arr):
    min_val, max_val = arr.min(), arr.max()
    return (arr - min_val) / (max_val - min_val) * 255


def create_sidebar(window, frame_update_function):
    sidebar = tk.Frame(window, width=200, bg='gray')
    sidebar.pack(fill='y', side='left', padx=5, pady=5)

    params = {
        "width": 500,
        "height": 500,
        "scale": 200.0,
        "octaves": 1,
        "persistence": 0.7,
        "lacunarity": 2.0,
        "strength": 0.5,
        "scale_factor": 4,
        "num_small_craters": 15,
        "num_large_craters": 3,
        "max_small_radius": 10,
        "max_large_radius": 40,
        "max_depth": 0.1,
        "sigma": 1
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

    def on_generate_clicked():
        save_settings(entries)
        params = load_settings(entries)  # Load the updated settings
        frame_update_function(window, canvas, params)

    generate_button = tk.Button(sidebar, text="Generate Terrain", command=on_generate_clicked)
    generate_button.pack(pady=5)
    exit_button = tk.Button(sidebar, text="Exit", command=lambda: exit_program(window, entries))
    exit_button.pack(pady=5)

    return sidebar, entries


def update_terrain_display(terrain, canvas):
    # Clear the canvas
    canvas.delete('all')

    # Create a colormap to map terrain heights to colors
    cmap = plt.get_cmap('terrain')

    # Normalize the terrain heights to the range [0, 1]
    terrain_normalized = (terrain - terrain.min()) / (terrain.max() - terrain.min())

    # Create a grid of rectangles on the canvas
    for i in range(terrain.shape[0]):
        for j in range(terrain.shape[1]):
            # Get the color for this tile from the colormap
            color = cmap(terrain_normalized[i, j])

            # Convert the color from RGB to hexadecimal
            color = '#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255))

            # Create a rectangle on the canvas for this tile
            canvas.create_rectangle(j, i, j+1, i+1, fill=color, outline="")

def update_frame(window, canvas, params):
    terrain = generate_terrain(
        int(params['width']), int(params['height']), scale=params['scale'],
        octaves=int(params['octaves']), persistence=params['persistence'], lacunarity=params['lacunarity']
    )
    if terrain is None:
        print("Error: generate_terrain returned None")
        return

    terrain = overlay_large_features(terrain, strength=params['strength'], scale_factor=params['scale_factor'])
    if terrain is None:
        print("Error: overlay_large_features returned None")
        return

    terrain = generate_craters(
        terrain, num_small_craters=int(params['num_small_craters']),
        num_large_craters=int(params['num_large_craters']),
        max_small_radius=params['max_small_radius'], max_large_radius=params['max_large_radius'],
        max_depth=params['max_depth']
    )
    if terrain is None:
        print("Error: generate_craters returned None")
        return

    terrain = apply_minimal_smoothing(terrain, sigma=params['sigma'])
    if terrain is None:
        print("Error: apply_minimal_smoothing returned None")
        return

    update_terrain_display(terrain, canvas)


def create_image(terrain):
    terrain_scaled = scale_to_grayscale(terrain).astype('uint8')
    img = Image.fromarray(terrain_scaled, 'L')
    return img


def save_settings(entries):
    global settings_saved
    if settings_saved:
        return

    settings = {param: float(entry.get()) for param, entry in entries.items() if entry.winfo_exists()}
    print("Settings to save:", settings)  # Debug print
    with open('settings.json', 'w') as f:
        json.dump(settings, f, indent=4)

    settings_saved = True

def exit_program(window, entries):
    print("Exit button clicked")  # Debug print
    save_settings(entries)
    print("Settings saved")  # Debug print
    window.destroy()


def load_settings(entries):
    try:
        with open('settings.json', 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        print("Settings file not found. Using default settings.")
        settings = {}

    # Ensure all necessary keys are present in settings
    for param, entry in entries.items():
        if param not in settings:
            print(f"Key '{param}' not found in settings. Using default value.")
            settings[param] = float(entry.get())
        elif entry.winfo_exists():
            entry.delete(0, tk.END)
            entry.insert(0, str(settings[param]))

    return settings


def main():
    window = tk.Tk()
    window.title("Planet Simulator")
    sidebar, entries = create_sidebar(window, update_frame)
    params = load_settings(entries)  # Load settings from the JSON file
    global label
    label = tk.Label(window)
    label.pack(side='right', expand=True)
    global canvas  # This global is necessary to prevent the canvas from being garbage collected
    canvas = tk.Canvas(window, width=params['width'], height=params['height'])  # Set initial size, it will be updated later
    canvas.pack(side='right', expand=True)
    update_frame(window, canvas, params)  # Generate initial terrain after loading settings
    window.bind('<Destroy>', lambda event: save_settings(entries))  # Save settings when window is about to close
    window.mainloop()


if __name__ == "__main__":
    main()