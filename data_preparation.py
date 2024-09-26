import os
import json
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import numpy as np

def list_to_pairs(lst):
    """Convert flat list of coordinates into list of (x, y) tuples."""
    return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]

def ensure_valid_polygon(polygon):
    """Attempt to correct a polygon if it's invalid."""
    if not polygon.is_valid:
        corrected_polygon = polygon.buffer(0)
        if corrected_polygon.is_valid:
            return corrected_polygon
        else:
            raise ValueError("Could not correct invalid polygon.")
    return polygon

def extract_points_from_polygon(polygon):
    """Extract all exterior coordinates from a Polygon or MultiPolygon."""
    if isinstance(polygon, Polygon):
        return list(polygon.exterior.coords)
    elif isinstance(polygon, MultiPolygon):
        points = []
        for poly in polygon.geoms:
            points.extend(list(poly.exterior.coords))
        return points
    else:
        raise ValueError("Unsupported geometry type")

def gift_wrapping_hull(points):
    """Compute the concave hull using the Gift Wrapping (Jarvis March) approach."""
    points = np.array(points)
    
    if len(points) < 3:
        raise ValueError("At least 3 points are required to form a hull")
    
    # Start with the leftmost point (smallest x-coordinate)
    hull = []
    point_on_hull = points[np.argmin(points[:, 0])]
    while True:
        hull.append(tuple(point_on_hull))
        endpoint = points[0]
        
        for point in points[1:]:
            if np.array_equal(endpoint, point_on_hull) or \
               np.cross(endpoint - point_on_hull, point - point_on_hull) > 0:
                endpoint = point
        
        point_on_hull = endpoint
        
        if np.array_equal(endpoint, hull[0]):
            break
    
    # Ensure the hull is closed
    hull.append(hull[0])
    
    concave_hull = Polygon(hull)
    return concave_hull

def calculate_shape_based_on_overlap(footprint, roof):
    """Calculate the shape based on the overlap between footprint and roof."""
    poly_footprint = ensure_valid_polygon(Polygon(footprint))
    poly_roof = ensure_valid_polygon(Polygon(roof))

    # Calculate the area of the intersection and the footprint
    intersection = poly_footprint.intersection(poly_roof)
    intersection_area = intersection.area
    footprint_area = poly_footprint.area

    if intersection_area > 0:
        overlap_ratio = intersection_area / footprint_area
        if overlap_ratio >= 0.1:
            # Condition 1: Use Unary Union for substantial overlap
            shape_polygon = unary_union([poly_footprint, poly_roof])
        else:
            # Condition 2: Use Jarvis March (Gift Wrapping) for small overlap
            combined_points = extract_points_from_polygon(poly_footprint) + extract_points_from_polygon(poly_roof)
            shape_polygon = gift_wrapping_hull(combined_points)
    else:
        # Condition 3: Use Jarvis March (Gift Wrapping) for no overlap
        combined_points = extract_points_from_polygon(poly_footprint) + extract_points_from_polygon(poly_roof)
        shape_polygon = gift_wrapping_hull(combined_points)

    return shape_polygon

def add_shape_label_to_annotations(data):
    """Add 'shape' label to each annotation in the data."""
    for annotation in data.get('annotations', []):
        footprint = list_to_pairs(annotation['footprint'])
        roof = list_to_pairs(annotation['roof'])
        
        try:
            # Calculate the shape based on overlap conditions
            shape_polygon = calculate_shape_based_on_overlap(footprint, roof)
            
            shape = []  # Initialize the shape variable

            if isinstance(shape_polygon, Polygon):
                shape = [coord for coord in shape_polygon.exterior.coords]
            elif isinstance(shape_polygon, MultiPolygon):
                for poly in shape_polygon.geoms:
                    shape.extend([coord for coord in poly.exterior.coords])

            flattened_shape = [coord for xy_pair in shape for coord in xy_pair]
            annotation['shape'] = flattened_shape
        
        except ValueError as e:
            print(f"Skipping annotation due to error: {e}")
            annotation['shape'] = []  # Optionally, leave this empty or handle as needed

def process_json_file(file_path):
    """Process a single JSON file to add 'shape' attribute to each annotation."""
    with open(file_path, 'r') as file:
        data = json.load(file)

    add_shape_label_to_annotations(data)

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def process_json_files_in_folder(folder_path):
    """Process all JSON files in the given folder to add 'shape' attributes."""
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            process_json_file(file_path)
            print(f'Processed {filename}')

# On val folder
folder_path = './data/BONAI-shape/val/label/'
process_json_files_in_folder(folder_path)

# On train folder
folder_path = './data/BONAI-shape/train/label/'
process_json_files_in_folder(folder_path)