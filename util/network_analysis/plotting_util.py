import numpy as np

def sum_distance_to_segment_endpoints(points, p1, p2):
    """
    Compute the sum of distances from each point to both endpoints of a segment.
    
    :param points: Nx2 array of (x, y) points.
    :param p1: (x, y) coordinates of the first endpoint.
    :param p2: (x, y) coordinates of the second endpoint.
    :return: Array of sum distances.
    """
    # Convert inputs to NumPy arrays
    p1, p2 = np.array(p1), np.array(p2)
    points = np.array(points)

    # Compute Euclidean distances to both endpoints
    d1 = np.linalg.norm(points - p1, axis=1)
    d2 = np.linalg.norm(points - p2, axis=1)

    # Sum the distances
    sum_distances = d1 + d2

    return sum_distances

def scan_edge_copies(edge_pts, p1, p2, L):
    min_dist = np.inf
    shifts = np.array([-L, 0, L])
    
    best_p1 = []
    best_p2 = []
    for i, xs in enumerate(shifts):
        for j, ys in enumerate(shifts):
            for z, zs in enumerate(shifts):
                shift = np.array([xs, ys, zs])        
                d = sum_distance_to_segment_endpoints(edge_pts, p1 + shift, p2) 
                d = np.mean(d)
                if d < min_dist:
                    best_p1 = p1 + shift
                    best_p2 = p2
                    min_dist = d
                
                d = sum_distance_to_segment_endpoints(edge_pts, p1, p2+shift) 
                d = np.mean(d)
                if d < min_dist:
                    best_p1 = p1 
                    best_p2 = p2 + shift
                    min_dist = d
                    
    return best_p1, best_p2, min_dist
    

if __name__ == '__main__':
    # Example usage:
    points = np.array([[1, 2], [3, 4], [5, 6], [8, 8]])
    segment_start = (0, 0)
    segment_end = (4, 4)
    
    distances = distance_points_to_segment(points, segment_start, segment_end)
    print(distances)
