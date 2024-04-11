import numpy as np
from itertools import combinations

def calculate_overlap(field1, field2):
    intersection = len(field1 & field2)
    union = len(field1 | field2)
    return intersection / union

def identify_same_fields(place_fields_list, overlap_threshold=0.6):
    D = len(place_fields_list)
    field_ids = {}
    field_counter = 0

    # Assign unique IDs to place fields and convert areas to sets
    for day_fields in place_fields_list:
        for field_center, field_area in day_fields.items():
            field_ids[(field_center, frozenset(field_area))] = field_counter
            field_counter += 1

    N = len(field_ids)
    registration_matrix = np.full((D, N), np.nan)

    # Precompute overlaps between all pairs of fields
    overlaps = {}
    for (day1, day2) in combinations(range(D), 2):
        for field_center1, field_area1 in place_fields_list[day1].items():
            for field_center2, field_area2 in place_fields_list[day2].items():
                overlap = calculate_overlap(set(field_area1), set(field_area2))
                if overlap >= overlap_threshold:
                    field_id1 = field_ids[(field_center1, frozenset(field_area1))]
                    field_id2 = field_ids[(field_center2, frozenset(field_area2))]
                    overlaps[(field_id1, field_id2)] = overlap
                    overlaps[(field_id2, field_id1)] = overlap

    # Update registration matrix based on overlaps
    for day, day_fields in enumerate(place_fields_list):
        for field_center, field_area in day_fields.items():
            field_id = field_ids[(field_center, frozenset(field_area))]
            registration_matrix[day, field_id] = 1

            for other_field_id, overlap in overlaps.items():
                if other_field_id[0] == field_id:
                    registration_matrix[day, other_field_id[1]] = 1

    return registration_matrix

# Example usage
place_fields_list = [
    {1: np.array([1, 2, 3]), 49: np.array([49, 35, 34, 6])},
    {1: np.array([1, 2, 4]), 50: np.array([50, 36, 35, 7])},
    {2: np.array([2, 3, 5]), 51: np.array([51, 37, 36, 8])}
]

registration_matrix = identify_same_fields(place_fields_list)
print(registration_matrix)
