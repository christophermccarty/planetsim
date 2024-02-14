import numpy as np
from opensimplex import OpenSimplex
from tqdm import tqdm
from scipy.ndimage import gaussian_filter



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

# Constants for greenhouse gases
WATER_VAPOR = 0.25
CARBON_DIOXIDE_GHG = 0.65
METHANE = 0.06
NITROUS_OXIDE = 0.03
OZONE = 0.01


def generate_terrain(width, height, seed=None, scale=400.0, octaves=1, persistence=0.7, lacunarity=2.0):
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

    # Scale the normalized terrain values to the range [0, 20000]
    terrain *= 20000

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


def calculate_greenhouse_effect(temperature):
    # The greenhouse effect increases the temperature based on the concentration of greenhouse gases
    greenhouse_effect = WATER_VAPOR + CARBON_DIOXIDE_GHG + METHANE + NITROUS_OXIDE + OZONE
    return temperature * (1 + greenhouse_effect)

def apply_atmospheric_model(temperatures):
    # Apply the greenhouse effect to each temperature in the map
    return [[calculate_greenhouse_effect(temp) for temp in row] for row in temperatures]


from tqdm import tqdm

from tqdm import tqdm

def calculate_temperature(elevation, latitude):
    # Constants for solar radiation simulation
    SOLAR_CONSTANT = 1361  # Solar constant in W/m^2
    BOND_ALBEDO = 0.306  # Albedo of the planet
    STEFAN_BOLTZMANN_CONSTANT = 5.670374419e-8  # Stefan-Boltzmann constant in W/m^2/K^4
    # Constants for sunlight absorption
    SUNLIGHT_ABSORPTION = 0.7

    # Calculate equilibrium temperature, T_eq
    equilibrium_temperature = ((SOLAR_CONSTANT * (1 - BOND_ALBEDO)) / (4 * STEFAN_BOLTZMANN_CONSTANT)) ** 0.25

    # Initialize the temperature array
    temperature = np.zeros_like(elevation)

    # Iterate over the elevation array with a progress bar
    for i in tqdm(range(elevation.shape[0]), desc='Calculating temperatures'):
        for j in range(elevation.shape[1]):
            # Adjust the temperature based on the elevation and latitude of the tile
            if elevation[i][j] <= 11000:  # For the troposphere
                temp = equilibrium_temperature - (0.0065 * elevation[i][j])  # Temperature decreases by 6.5 degrees per km of elevation
            else:  # For the stratosphere and above
                temp = equilibrium_temperature - (0.0065 * 11000)  # Temperature is constant

            temp *= 1 - 0.0025 * np.abs(latitude[i])  # Temperature decreases by 2.5 degrees per degree of latitude from the equator
            temp = temp - 273.15  # Convert from Kelvin to Celsius

            # Adjust the temperature based on the greenhouse gas concentration and solar radiation
            greenhouse_effect = WATER_VAPOR + CARBON_DIOXIDE_GHG + METHANE + NITROUS_OXIDE + OZONE
            temp *= (1 + greenhouse_effect) * SUNLIGHT_ABSORPTION

            # Store the calculated temperature
            temperature[i][j] = temp

    return temperature