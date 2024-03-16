import numpy as np
from opensimplex import OpenSimplex
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import multiprocessing


# Constants for atmospheric composition
NITROGEN = 0.78084
OXYGEN = 0.20946
ARGON = 0.00934
CARBON_DIOXIDE = 0.000413
NEON = 0.00001818
HELIUM = 0.00000524
METHANE = 0.00000179
KRYPTON = 0.00000114
HYDROGEN = 0.00000055
NITROUS_OXIDE = 0.00000032
XENON = 0.000000009
OZONE = 0.000000004
NITROGEN_DIOXIDE = 0.000000002
IODINE = 0.000000001
CARBON_MONOXIDE = 0.000000001

# Constants for radiative forcing of greenhouse gases (W/m^2)
WATER_VAPOR_RF = 0.15  # Estimated value, actual value varies greatly
CARBON_DIOXIDE_RF = 1.82
METHANE_RF = 0.48
NITROUS_OXIDE_RF = 0.17
OZONE_RF = 0.35  # Tropospheric Ozone

# Constants for albedo
ALBEDO_LAND = 0.3  # Albedo of land
ALBEDO_WATER = 0.06  # Albedo of water


def generate_noise_for_chunk(args):
    simplex, scale, octaves, persistence, lacunarity, chunk, start_i, start_j = args
    for i in range(chunk.shape[0]):
        for j in range(chunk.shape[1]):
            x = (i + start_i) * scale
            y = (j + start_j) * scale
            noise = 0
            amplitude = 1
            frequency = 1
            for _ in range(octaves):
                noise += amplitude * simplex.noise2(frequency * x, frequency * y)
                amplitude *= persistence
                frequency *= lacunarity
            chunk[i, j] = noise
    return chunk

def generate_terrain(width, height, scale, octaves, persistence, lacunarity):
    # Initialize an OpenSimplex noise generator with a seed
    simplex1 = OpenSimplex(seed=0)

    # Create an empty array for the terrain
    terrain = np.zeros((height, width))

    # Determine the number of CPU cores
    num_cores = multiprocessing.cpu_count() - 1

    # Split the terrain array into chunks
    chunks = np.array_split(terrain, num_cores)

    # Create a multiprocessing pool
    pool = multiprocessing.Pool(num_cores)

    # Generate the Perlin noise for each chunk
    results = pool.map(generate_noise_for_chunk, [(simplex1, scale, octaves, persistence, lacunarity, chunk,
                                                   i*chunk.shape[0], 0) for i, chunk in enumerate(chunks)])

    # Combine the chunks back into a single terrain array
    terrain = np.concatenate(results)

    # Normalize the terrain data to a range of 0 to 1
    terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))

    # Scale the terrain data to the desired range of 0 to 10000
    terrain = terrain * 10000

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


def add_crater_optimized(terrain, cx, cy, radius, depth, angle):
    height, width = terrain.shape
    y, x = np.ogrid[-cx:height-cx, -cy:width-cy]
    mask = x**2 + y**2 <= radius**2

    # Apply crater effect within the masked area
    distance_from_center = np.sqrt(x**2 + y**2)[mask]
    delta = depth * (1 - distance_from_center / radius)

    # Adjust the shape of the crater based on the angle of impact
    delta *= np.cos(np.radians(angle))  # Adjust the depth based on the angle

    terrain[mask] -= delta
    terrain[terrain < 0] = 0  # Ensure terrain height doesn't go below 0


def generate_craters(terrain, num_small_craters=15, num_large_craters=3, max_small_radius=10, max_large_radius=40,
                     max_depth=0.1, angle_range=(20, 90)):
    total_craters = num_small_craters + num_large_craters
    with tqdm(total=total_craters, desc='Adding craters') as pbar:
        # Add small craters
        for _ in range(num_small_craters):
            height, width = terrain.shape
            # Random center, radius, depth, and angle for each crater
            cx, cy = np.random.randint(0, height), np.random.randint(0, width)
            radius = np.random.uniform(0, max_small_radius)
            depth = np.random.uniform(0, max_depth)
            angle = np.random.uniform(*angle_range)  # Select a random angle from the range
            add_crater_optimized(terrain, cx, cy, radius, depth, angle)
            pbar.update()

        # Add large craters
        for _ in range(num_large_craters):
            height, width = terrain.shape
            # Random center, radius, depth, and angle for each crater
            cx, cy = np.random.randint(0, height), np.random.randint(0, width)
            radius = np.random.uniform(0, max_large_radius)
            depth = np.random.uniform(0, max_depth)
            angle = np.random.uniform(*angle_range)  # Select a random angle from the range
            add_crater_optimized(terrain, cx, cy, radius, depth, angle)
            pbar.update()

    return terrain


def apply_minimal_smoothing(terrain, sigma=1):
    return gaussian_filter(terrain, sigma=sigma)


def scale_to_grayscale(arr):
    min_val, max_val = arr.min(), arr.max()
    return (arr - min_val) / (max_val - min_val) * 255


# Create a function to calculate the latitude of each tile
def calculate_latitudes(terrain, step=10):
    # Constants
    R = 6371  # Radius of the Earth (km)

    # Calculate the latitude of each tile for a subset of points
    subset_indices = np.arange(0, terrain.shape[0], step)
    subset_latitudes = 90 - (subset_indices / terrain.shape[0]) * 180

    # Use interpolation to estimate the latitudes for the remaining points
    interpolate = interp1d(subset_indices, subset_latitudes, kind='linear', fill_value="extrapolate")
    latitudes_array = interpolate(np.arange(terrain.shape[0]))

    return latitudes_array


def calculate_temperature(args):
    # Eventually add back in atmospheric temperature generation with
    # atmospheric_temperatures = np.zeros_like(terrain)
    # Tatm_solution = f * Ts_solution
    # atmospheric_temperatures[i][j] = Tatm_solution
    # and return atmospheric_temperatures as well
    terrain, resized_shape, lat = args
    # Constants
    S0 = 1361  # Solar constant (W/m2)
    sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m2/K4)
    A = 0.3  # Albedo
    G = 0.85  # Greenhouse factor
    epsilon = 0.8  # Atmospheric emissivity
    f = 0.7  # Factor to relate Tatm to Ts
    lapse_rate = 0.0065  # Lapse rate (K/m)

    # Create a 2D array of latitudes
    latitudes_2d = np.tile(lat, (resized_shape[1], 1)).T

    # Linear model for temperature based on latitude
    Ts_solutions = 288 - 0.5 * np.abs(latitudes_2d)  # 288 is the average global temperature in K, 0.5 is a scaling factor

    # Adjust temperature based on elevation
    elevation = terrain.reshape(resized_shape)  # Reshape terrain to match the shape of Ts_solutions
    elevation_resized = np.resize(elevation, Ts_solutions.shape)  # Resize elevation to match the shape of Ts_solutions
    Ts_solutions -= lapse_rate * elevation_resized

    return Ts_solutions


def model_wind(terrain, latitudes):
    # Constants
    omega = 7.292e-5  # Earth's rotation rate (rad/s)
    wind_speed_at_ref = np.full_like(terrain, 10)  # Arbitrary wind speed at reference height
    ref_height = 100  # Reference height in meters
    roughness_length = 0.1  # Roughness length in meters (can be adjusted based on terrain)

    # Reshape latitudes to match the shape of terrain
    latitudes_2d = np.tile(latitudes, (terrain.shape[1], 1)).T

    # Calculate Coriolis force
    coriolis_force = 2 * omega * np.sin(np.radians(latitudes_2d))

    # Add a small constant to terrain to avoid division by zero or logarithm of zero
    terrain = np.abs(terrain) + 1e-7

    # Adjust wind speed based on Coriolis force and elevation
    wind_speed = (wind_speed_at_ref * (1 + coriolis_force) * np.log((terrain + terrain) / (roughness_length + terrain))
                  / np.log((ref_height + terrain) / (roughness_length + terrain)))

    # Adjust wind direction based on latitude to model Hadley, Ferrel, and Polar cells
    latitude_adjustment = np.zeros_like(latitudes_2d)
    latitude_adjustment[(latitudes_2d >= 0) & (latitudes_2d < 30)] = np.pi / 4  # Trade winds (Hadley Cell)
    latitude_adjustment[(latitudes_2d >= 30) & (latitudes_2d < 60)] = -np.pi / 4  # Westerlies (Ferrel Cell)
    latitude_adjustment[(latitudes_2d >= 60) & (latitudes_2d <= 90)] = np.pi / 4  # Polar easterlies (Polar Cell)
    latitude_adjustment[(latitudes_2d < 0) & (latitudes_2d > -30)] = -np.pi / 4  # Trade winds (Hadley Cell)
    latitude_adjustment[(latitudes_2d <= -30) & (latitudes_2d > -60)] = np.pi / 4  # Westerlies (Ferrel Cell)
    latitude_adjustment[(latitudes_2d <= -60) & (latitudes_2d >= -90)] = -np.pi / 4  # Polar easterlies (Polar Cell)

    # Ensure wind direction is within the range [-pi, pi]
    adjusted_wind_direction = (latitude_adjustment + np.pi) % (2 * np.pi) - np.pi

    return wind_speed, adjusted_wind_direction
