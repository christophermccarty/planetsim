import random
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from opensimplex import OpenSimplex

class Tile:
    def __init__(self, altitude, temperature=20.0, biome="Unknown", precipitation=0.0, plate_type="Unknown",
                 drift_rate=0.0):
        self.altitude = altitude
        self.temperature = temperature
        self.biome = biome
        self.precipitation = precipitation
        self.plate_type = plate_type
        self.drift_rate = drift_rate


def get_terrain_color(altitude, max_altitude, min_altitude):
    """Maps an altitude value to a terrain color."""
    color_mapping = {
        -1.0: "#0b3d91",  # Deep Sea
        -0.5: "#1e5fb5",  # Ocean
        -0.0: "#3e8ce3",  # Shallow Sea
        0.05: "#44a5e5",  # Beach
        0.1: "#00cc9c",  # Coastal Plain
        0.2: "#00cc58",  # Lowlands
        0.3: "#00cc00",  # Plateau
        0.35: "#7fff00",  # Foothills
        0.4: "#adff2f",  # Low Mountains
        0.45: "#ffff00",  # High Mountains
        0.5: "#ffcc00",  # Snowy Mountains
        0.55: "#ffa500",  # Alpine
        0.6: "#ff7f50",  # High Alpine
        0.65: "#ff6347",  # High Ice Fields
        0.7: "#ff4500",  # Glacial Peaks
        0.75: "#8b0000",  # Snowy Peaks
        0.8: "#8b4513",  # High Peaks
        0.85: "#a0522d",  # Summit Range
        0.9: "#d2691e",  # Summit Peaks
        0.95: "#f4a460",  # High Summit Peaks
        1.0: "#ffffff"  # Mountaintops
    }

    # Normalize the altitude value between -1 and 1
    normalized = 2 * (altitude - min_altitude) / (max_altitude - min_altitude) - 1

    # Find the appropriate color based on the altitude
    previous_key = None
    for key in sorted(color_mapping.keys()):
        if normalized <= key:
            if previous_key is None:
                return color_mapping[key]
            # Choose the color whose key is closest to the normalized altitude
            if abs(normalized - previous_key) < abs(normalized - key):
                return color_mapping[previous_key]
            return color_mapping[key]
        previous_key = key

def batch_noise3(simplex, x, y, z):
    # Create an empty array to store the noise values
    noise_values = np.empty_like(x)

    # Compute the noise values for each coordinate
    for i in np.ndindex(x.shape):
        noise_values[i] = simplex.noise3(x[i], y[i], z[i])

    return noise_values


def generate_terrain(canvas, width, height, tiles_count, warp_factor,
                     num_octaves, persistence, lacunarity, amplitude, frequency, scale):
    simplex = OpenSimplex(seed=random.randint(0, 1000000))
    frequency *= scale  # Scaling the frequency
    tiles_x = int((tiles_count // (height / width)) ** 0.5)
    tiles_y = tiles_count // tiles_x
    tile_width = width / tiles_x
    tile_height = height / tiles_y

    min_altitude, max_altitude = random.randint(-11000, 0), random.randint(0, 9000)

    # Create a grid of u and v coordinates
    u, v = np.meshgrid(np.linspace(0, 1, tiles_x), np.linspace(0, 1, tiles_y))

    # Convert u, v to theta, phi, and then to x, y, z
    theta = 2 * np.pi * u
    phi = np.pi * v
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Warp x, y, z
    warp_x = x + warp_factor * batch_noise3(simplex, x, y, z)
    warp_y = y + warp_factor * batch_noise3(simplex, y, z, x)
    warp_z = z + warp_factor * batch_noise3(simplex, z, x, y)

    # Generate noise values using octaves
    noise_value = np.zeros_like(u)
    # amplitude = 1.0
    # frequency = 0.005
    for k in range(num_octaves):
        noise_value += batch_noise3(simplex, warp_x * frequency, warp_y * frequency, warp_z * frequency) * amplitude
        frequency *= lacunarity
        amplitude *= persistence

    # Calculate altitude
    altitude = min_altitude + (noise_value + 1) * 0.5 * (max_altitude - min_altitude)

    # Create an offscreen image buffer and a drawing context
    buffer = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(buffer)

    # Create tiles and draw them onto the offscreen buffer
    tiles = []
    for i in range(tiles_x):
        row = []
        for j in range(tiles_y):
            tile_altitude = altitude[j, i]
            tile = Tile(tile_altitude)
            color = get_terrain_color(tile_altitude, max_altitude, min_altitude)
            x0, y0 = i * tile_width, j * tile_height
            x1, y1 = x0 + tile_width, y0 + tile_height
            draw.rectangle([(x0, y0), (x1, y1)], fill=color, outline=color)
            row.append(tile)
        tiles.append(row)

    # Update the last column of tiles based on the first column
    for j in range(tiles_y):
        tiles[tiles_x - 1][j].altitude = tiles[0][j].altitude
        color = get_terrain_color(tiles[0][j].altitude, max_altitude, min_altitude)
        x0, y0 = (tiles_x - 1) * tile_width, j * tile_height
        x1, y1 = x0 + tile_width, y0 + tile_height
        draw.rectangle([(x0, y0), (x1, y1)], fill=color, outline=color)

    # Convert the PIL image to a Tkinter-compatible image and draw it onto the canvas
    global tk_image
    tk_image = ImageTk.PhotoImage(buffer)
    canvas.create_image(0, 0, anchor="nw", image=tk_image)

    return tiles


def regenerate_map_with_slider_values(sliders, canvas, window_width, window_height, tiles_count):
    """Retrieve values from the sliders and regenerate the map."""
    values = {key: slider.get() for key, slider in sliders.items()}
    simplex = OpenSimplex(seed=random.randint(0, 1000000))
    generate_terrain(canvas, window_width, window_height, tiles_count,
                     num_octaves=values["num_octaves"],
                     persistence=values["persistence"], lacunarity=values["lacunarity"],
                     amplitude=values["amplitude"], frequency=values["frequency"], scale=values["scale"],
                     warp_factor=values["warp_factor"])


def generate_temperature_map(tiles):
    watts_per_square_meter = 1361.0
    for row in tiles:
        for tile in row:
            # Calculate temperature based on solar energy and tile's altitude
            tile.temperature = (watts_per_square_meter / 100) - (tile.altitude / 1000)

def get_temperature_color(temperature, max_temperature, min_temperature):
    # Normalize the temperature value between 0 and 1
    normalized_temp = (temperature - min_temperature) / (max_temperature - min_temperature)

    # Map the normalized temperature to a color (example: blue to red)
    r = int(normalized_temp * 255)
    g = 0
    b = int((1 - normalized_temp) * 255)

    return (r, g, b)


def render_terrain_map(tiles, canvas, width, height):
    tiles_x = len(tiles)
    tiles_y = len(tiles[0])
    tile_width = width / tiles_x
    tile_height = height / tiles_y

    # Find the maximum and minimum altitude values
    max_altitude = max(max(tile.altitude for tile in row) for row in tiles)
    min_altitude = min(min(tile.altitude for tile in row) for row in tiles)

    # Create an offscreen image buffer and a drawing context
    buffer = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(buffer)

    # Draw the tiles onto the offscreen buffer with colors based on altitude
    for i in range(tiles_x):
        for j in range(tiles_y):
            tile_altitude = tiles[i][j].altitude
            color = get_terrain_color(tile_altitude, max_altitude, min_altitude)
            x0, y0 = i * tile_width, j * tile_height
            x1, y1 = x0 + tile_width, y0 + tile_height
            draw.rectangle([(x0, y0), (x1, y1)], fill=color, outline=color)

    # Convert the PIL image to a Tkinter-compatible image and draw it onto the canvas
    global tk_image_terrain
    tk_image_terrain = ImageTk.PhotoImage(buffer)
    canvas.create_image(0, 0, anchor="nw", image=tk_image_terrain)


def render_temperature_map(tiles, canvas, width, height):
    tiles_x = len(tiles)
    tiles_y = len(tiles[0])
    tile_width = width / tiles_x
    tile_height = height / tiles_y

    # Find the maximum and minimum temperature values
    max_temperature = max(max(tile.temperature for tile in row) for row in tiles)
    min_temperature = min(min(tile.temperature for tile in row) for row in tiles)

    # Create an offscreen image buffer and a drawing context
    buffer = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(buffer)

    # Draw the tiles onto the offscreen buffer with colors based on temperature
    for i in range(tiles_x):
        for j in range(tiles_y):
            tile_temperature = tiles[i][j].temperature
            color = get_temperature_color(tile_temperature, max_temperature, min_temperature)
            x0, y0 = i * tile_width, j * tile_height
            x1, y1 = x0 + tile_width, y0 + tile_height
            draw.rectangle([(x0, y0), (x1, y1)], fill=color, outline=color)

    # Convert the PIL image to a Tkinter-compatible image and draw it onto the canvas
    global tk_image_temperature
    tk_image_temperature = ImageTk.PhotoImage(buffer)
    canvas.create_image(0, 0, anchor="nw", image=tk_image_temperature)
