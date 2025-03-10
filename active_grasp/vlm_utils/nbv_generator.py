import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

truncate=np.pi/4
# np.random.seed(42)

def generate_random_points_in_sphere(S, R_s, num_points=10):
    """
    Generates random points uniformly distributed on the sphere surface.
    """
    theta = np.random.uniform(0, np.pi/2, num_points)  # Polar angle
    phi = np.random.uniform(0, 2 * np.pi, num_points)  # Azimuthal angle
    R_s = np.random.uniform(0, 1, num_points)*R_s * 0.6 # Random radius

    X = S[0] + R_s * np.sin(theta) * np.cos(phi)
    Y = S[1] + R_s * np.sin(theta) * np.sin(phi)
    Z = S[2] + R_s * np.cos(theta)

    sphere_points = np.vstack((X, Y, Z)).T
    return sphere_points

    
def generate_random_points_on_sphere(S, R_s, num_points=10):
    """
    Generates uniform points uniformly distributed on the sphere surface.
    """
    theta = np.linspace(0, np.pi/2 - truncate, num_points)  # Polar angle
    phi = np.linspace(0, 2 * np.pi, num_points)  # Azimuthal angle

    sphere_points = []
    for t in theta:
        for p in phi:
            X = S[0] + R_s * np.sin(t) * np.cos(p)
            Y = S[1] + R_s * np.sin(t) * np.sin(p)
            Z = S[2] + R_s * np.cos(t)
            sphere_points.append([X, Y, Z])

    return np.array(sphere_points)

# 2. Compute the **intersection circle** between the plane and sphere
def compute_fixed_intersection_circle(S, R_s, P1, P2):
    """
    Computes the intersection circle between the sphere and the perpendicular plane at the midpoint of P1 and P2.
    The plane does NOT pass through the sphere center but through the midpoint of P1 and P2.
    """
    # Compute the midpoint (this is where the plane passes through)
    C = (P1 + P2) / 2

    # Compute the normal of the plane (vector from P1 to P2)
    N = P2 - P1
    N = N / np.linalg.norm(N)

    # Compute the distance from the sphere center to the plane
    d = abs(np.dot(S - C, N))

    # Ensure the plane intersects the sphere
    if d >= R_s:
        raise ValueError("The plane does not intersect the sphere (d >= R_s)")

    # Compute the correct radius of the intersection circle
    R_c = np.sqrt(R_s**2 - d**2)

    # Compute the projection of the sphere center onto the plane (correct circle center)
    C_circle = S - np.dot((S - C), N) * N

    return C, C_circle, R_c, N

# 4. Query **tangent vector at a specific point**
def query_tangent_vector(S, R_s, P1, P2, P_q):
    """
    Queries the tangent vector at a given point on the sphere.
    """
    if isinstance(P_q, list):
        P_q = np.array(P_q)

    # Ensure P_q is on the sphere (project if necessary)
    P_q = S + R_s * (P_q - S) / np.linalg.norm(P_q - S)

    # Compute the normal of the plane (vector from P1 to P2)
    N = P2 - P1
    N = N / np.linalg.norm(N)

    # Compute radial vector at P_q
    radial_vector = P_q - S
    radial_vector = radial_vector / np.linalg.norm(radial_vector)

    # Compute rejection of N onto the radial vector
    proj_N_on_radial = np.dot(N, radial_vector) * radial_vector
    tangent_vector = N - proj_N_on_radial  # Remove radial component

    # Normalize to maintain consistent vector lengths
    tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)

    # # Compute the projection of S onto the plane perpendicular to P1 P2
    # P_mid = (P1 + P2) / 2
    # dist_S_plane = np.dot(P_mid - S, N)
    # alpha = (dist_S_plane + R_s) / R_s 

    # # Get the starting pole of the field
    # P_pole = S - R_s * N

    # Compute the strength of the field
    # theta = np.arccos(np.dot(P_pole - S, P_q - S) / R_s**2) / np.pi # normalize to [0, 1]
    # strength = 1 - theta**alpha

    beta = np.dot(P2 - P1, P_q - P1) / (np.linalg.norm(P2 - P1) * np.linalg.norm(P_q - P1))
    beta = np.arccos(beta) / np.pi

    return tangent_vector * beta # + generate_upward_tangent_vector(S, P_q) 


def query_tangent_vector_sum_from_field_list(S, P_q, field_list):
    """sum up the tangent vectors"""
    tangent_vector = np.zeros(3)
    for field in field_list:
        tangent_vector += field(P_q = P_q)

    radial_vector = P_q - S
    radial_vector = radial_vector / np.linalg.norm(radial_vector)
    elevation = np.abs(np.arcsin(radial_vector[2]))
    upward = generate_upward_tangent_vector(S, P_q) 
    if elevation <= truncate and np.dot(tangent_vector, upward) < 0:
        # rejection on upward tangent vector
        tangent_vector = tangent_vector - np.dot(tangent_vector, upward) * upward
        # print(f"elevation: {elevation / np.pi * 180}")

    mag = np.linalg.norm(tangent_vector)
    if mag < 1e-2:
        tangent_vector*= 0
    return tangent_vector


def generate_upward_tangent_vector(S, P_q):
    """
    Generates a tangent vector that always points upwards relative to the sphere.
    """
    radial_vector = P_q - S
    radial_vector /= np.linalg.norm(radial_vector) + 1e-6
    
    up_vector = np.array([0, 0, 1])  # Global upward direction
    tangent_vector = np.cross(np.cross(radial_vector, up_vector), radial_vector)
    tangent_vector /= np.linalg.norm(tangent_vector) + 1e-6
    return tangent_vector

def generate_tangent_vector_towards_opposite(S, P_q):
    """
    Generates a tangent vector that always points towards the opposite side of the sphere.
    """
    radial_vector = P_q - S
    radial_vector /= np.linalg.norm(radial_vector) + 1e-6
    
    up_vector = np.array([0, 0, 1])  # Global upward direction
    tangent_vector = np.cross(np.cross(radial_vector, up_vector), radial_vector)
    tangent_vector /= np.linalg.norm(tangent_vector) + 1e-6
    return -tangent_vector


def animate_query_tangent_vectors(S, R_s, Ps_num=2):
    """
    Animates query points moving along the velocity field on the sphere with trajectory visualization.
    """
    # Generate reference points inside the sphere
    Ps = generate_random_points_in_sphere(S, R_s, Ps_num)

    query_point_cam = generate_random_points_on_sphere(S, R_s, 1)[0]

    animate_query_tangent_vector(S, 
                                 R_s, 
                                 query_point_cam,
                                 Ps[0],
                                 Ps[1:],
                                 field_list=[partial(query_tangent_vector, S, R_s, Ps[i], Ps[0]) for i in range(1, Ps_num)])
    
    

def animate_query_tangent_vector(S, 
                                R_s, 
                                query_point_cam, 
                                target_point,
                                occlusion_points,
                                field_list=None,
                                steps=100, dt=1e-2, num_queries=20):
    """
    Animates query points moving along the velocity field on the sphere with trajectory visualization.
    """
    # Generate multiple query points on the sphere
    query_point_cam = np.array(query_point_cam)
    
    # Store trajectories for all query points
    trajectory = []
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate multiple query points on the sphere
    query_points = generate_random_points_on_sphere(S, R_s, num_queries)
    query_points = np.array(query_points)
    
    # Compute tangent vectors at each query point
    tangent_vectors = np.zeros((num_queries**2, 3))
    for i, q in enumerate(query_points):
        tangent_vectors[i] = query_tangent_vector_sum_from_field_list(S, P_q=q, field_list=field_list)
    tangent_vectors /= np.linalg.norm(tangent_vectors, axis=1)[:, None] + 1e-6
    
    # Plot sphere surface
    u = np.linspace(0, np.pi / 2, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    X = S[0] + R_s * np.outer(np.sin(u), np.cos(v))
    Y = S[1] + R_s * np.outer(np.sin(u), np.sin(v))
    Z = S[2] + R_s * np.outer(np.cos(u), np.ones_like(v))
    ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.3)
    
    # Initialize scatter plot for moving points
    points_plot, = ax.plot([], [], [], 'ro', label="Query points")

    # Plot P1, P2, and the query points
    ax.scatter(*target_point, color='green', label="Target Object", s=200)
    for i, P in enumerate(occlusion_points):
        ax.scatter(*P, color='blue', label=f"Neighbour {i+1}", s=100)
    ax.scatter(*np.array(query_points).T, color='purple', label="Query Points", s=20)
    
    # Plot tangent vectors
    ax.quiver(query_points[:, 0], query_points[:, 1], query_points[:, 2],
              tangent_vectors[:, 0], tangent_vectors[:, 1], tangent_vectors[:, 2],
              length=.025, color='red', label="Tangent Vectors")
    
    # Initialize scatter plot for moving points
    point_plot, = ax.plot([], [], [], 'ro', label="Camera Viewpoint")
    trajectory_line = ax.plot([], [], [], 'b-', alpha=0.5)[0]
    
    def update(frame):
        nonlocal query_point_cam, trajectory
        
        # Recompute tangent vectors at current positions
        tangent_vector = query_tangent_vector_sum_from_field_list(S, P_q=query_point_cam, field_list=field_list)
        
        # Compute new positions
        query_point_cam += dt * tangent_vector
        
        # Project back to the sphere
        query_point_cam = S + R_s * (query_point_cam - S) / np.linalg.norm(query_point_cam - S)
        
        # Store new positions in trajectories for all query points
        trajectory.append(query_point_cam.copy())
            
        # Update scatter plot
        point_plot.set_data(query_point_cam[None,:][:,0], query_point_cam[None,:][:,1])
        point_plot.set_3d_properties(query_point_cam[None,:][:,2])
        
        # Update trajectory lines for all query points
        
        traj_arr = np.array(trajectory)
        trajectory_line.set_data(traj_arr[:,0], traj_arr[:,1]) 
        trajectory_line.set_3d_properties(traj_arr[:,2])
        
        return [points_plot] + [trajectory_line]
    
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_zticks([])  # Remove z-axis ticks

    ax.set_xticklabels([])  # Remove x-axis labels
    ax.set_yticklabels([])  # Remove y-axis labels
    ax.set_zticklabels([])  # Remove z-axis labels

    ax.grid(False)  # Disable grid

    # Hide the panes (background walls)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    # Hide the axis lines (spines)
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Transparent
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    ax.set_title("Query Tangent Vectors Motion on the Sphere")
    
    max_range = R_s * 1.2
    ax.set_xlim(S[0] - max_range, S[0] + max_range)
    ax.set_ylim(S[1] - max_range, S[1] + max_range)
    ax.set_zlim(S[2] - max_range, S[2] + max_range)
    
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    S = np.array([0, 0, 0])  # Sphere Center
    R_s = 1  # Sphere Radius
    animate_query_tangent_vectors(S, R_s)
