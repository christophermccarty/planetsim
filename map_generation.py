import numpy as np
from opensimplex import OpenSimplex
import multiprocessing
from scipy.ndimage import gaussian_filter

class MapGenerator:
    def __init__(self, seed):
        self.generator_seed = seed

    def generate_noise_map_multiprocessing(self, map_height, map_width, octaves, frequency, lacunarity, persistence,
                                         layer_name='', num_processes=1):
        print(f'Starting generation of {layer_name} with multiprocessing...')
        print(f'Using {num_processes} processes for {layer_name} generation.')
    
        # Latitude and Longitude arrays generated once
        latitude = np.pi * (np.linspace(0, 1, map_height) - 0.5)  # Range -π/2 to π/2
        longitude = 2 * np.pi * (np.linspace(0, 1, map_width) - 0.5)  # Range -π to π
        latitudes, longitudes = np.meshgrid(latitude, longitude, indexing="ij")
    
        # Prepare arguments for each process
        chunk_indices = np.array_split(np.arange(map_height), num_processes)
        args_list = []
        for idx, chunk in enumerate(chunk_indices):
            lat_chunk = latitudes[chunk, :]
            lon_chunk = longitudes[chunk, :]
            args = (
                self.generator_seed,
                octaves,
                frequency,
                lacunarity,
                persistence,
                lat_chunk,
                lon_chunk,
                layer_name,
                idx + 1,
                len(chunk_indices)
            )
            args_list.append(args)
    
        # Process the map in chunks
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(generate_noise_chunk, args_list)
    
        # Reassemble the results
        total = np.vstack(results)
        print(f'Finished generation of {layer_name}.')
        return total

    @staticmethod
    def normalize_data(data):
        return (data - data.min()) / (data.max() - data.min())

def generate_noise_chunk(args):
    (
        generator_seed,
        octaves,
        base_frequency,
        lacunarity,
        persistence,
        latitudes,
        longitudes,
        layer_name,
        chunk_number,
        total_chunks
    ) = args

    # Initialize OpenSimplex generator
    generator = OpenSimplex(generator_seed)

    # Initialize arrays for total noise
    total = np.zeros_like(latitudes, dtype=np.float32)

    # Precompute amplitude and frequency values
    amplitudes = np.array([persistence ** octave for octave in range(octaves)], dtype=np.float32)
    frequencies = np.array([base_frequency * (lacunarity ** octave) for octave in range(octaves)], dtype=np.float32)

    # Convert latitude and longitude to spherical coordinates
    x_coords = np.cos(latitudes) * np.cos(longitudes)
    y_coords = np.cos(latitudes) * np.sin(longitudes)
    z_coords = np.sin(latitudes)

    # Sum amplitudes for normalization
    max_amplitude = np.sum(amplitudes)

    # Loop over octaves and accumulate noise
    for octave in range(octaves):
        frequency = frequencies[octave]
        amplitude = amplitudes[octave]

        # Scale coordinates by frequency
        x = frequency * x_coords
        y = frequency * y_coords
        z = frequency * z_coords

        # Generate noise values
        noise_values = np.vectorize(generator.noise3)(x, y, z)

        # Accumulate weighted noise values
        total += noise_values * amplitude
        print(f'[{layer_name}] Process {chunk_number}: Octave {octave + 1}/{octaves} complete.')

    # Normalize the result
    total /= max_amplitude
    print(f'[{layer_name}] Process {chunk_number}/{total_chunks} finished.')

    return total