import numpy as np


def generalized_dot(vec_map_a, vec_map_b):
    return np.sum(vec_map_a * vec_map_b, axis=-1)


def generalized_norm(vecmap):
    return np.sqrt(generalized_dot(vecmap, vecmap))


def normalize_vector_map(vector_map):
    normalization = np.linalg.norm(vector_map, axis=-1)
    return vector_map / normalization[..., np.newaxis]


def reflect_light(light, normals):
    return light - 2 * generalized_dot(light, normals)[..., np.newaxis] * normals
