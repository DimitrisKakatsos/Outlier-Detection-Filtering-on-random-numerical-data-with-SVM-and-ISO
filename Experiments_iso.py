import numpy as np
from sklearn.ensemble import IsolationForest  
import random
import time
import matplotlib.pyplot as plt
import warnings
#ee
warnings.filterwarnings("ignore")

class OutlierBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
    
    def add(self, item):
        self.buffer.append(item)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
    
    def get_all(self):
        return self.buffer

def calculate_filter_bounds(sorted_outliers):
    sorted_outliers = np.array(sorted_outliers) 
    xmin_outlier = [np.min(sorted_outliers[:, 0]), np.argmin(sorted_outliers[:, 0])]
    xmax_outlier = [np.max(sorted_outliers[:, 0]), np.argmax(sorted_outliers[:, 0])]
    ymin_outlier = [np.min(sorted_outliers[:, 1]), np.argmin(sorted_outliers[:, 1])]
    ymax_outlier = [np.max(sorted_outliers[:, 1]), np.argmax(sorted_outliers[:, 1])]

    # Calculate w_prime vector (a_count, b_count, c_count, d_count)
    a_count = np.sum(np.isclose(sorted_outliers, [xmin_outlier[0], 0]))
    b_count = np.sum(np.isclose(sorted_outliers, [xmax_outlier[0], 0]))
    c_count = np.sum(np.isclose(sorted_outliers, [0, ymin_outlier[1]]))
    d_count = np.sum(np.isclose(sorted_outliers, [0, ymax_outlier[1]]))
    w_prime = np.array([a_count, b_count, c_count, d_count])
    #print(a_count)
    theta_prime = 5
    # Check and update filter bounds based on w_prime vector
    for sorted_outliers in w_prime:    
        if a_count > theta_prime:
            xmin_outlier[0] *= 0.8
        if b_count > theta_prime:
            xmax_outlier[0] *= 1.2
        if c_count > theta_prime:
            ymin_outlier[1] *= 0.8
        if d_count > theta_prime:
            ymax_outlier[1] *= 1.2
    return xmin_outlier, xmax_outlier, ymin_outlier, ymax_outlier

def sort_outliers(outliers_detected, good_data_points, centroid, gamma, delta, gamma_prime, delta_prime, threshold):
    count = 0
    cockoc = 0
    if len(good_data_points) == 0:
        return np.array([]),np.array([])
    sorted_outliers = []
    for outlier in outliers_detected:
        # Calculate distances
        distance_a = np.linalg.norm(outlier - good_data_points, axis=1)
        distance_b = np.linalg.norm(outlier - centroid)
        x1 = gamma * distance_a + delta
        x2 = gamma_prime * distance_b + delta_prime
        sort_score_a = 1 / (1 + np.exp(-x1))
        sort_score_b = 1 / (1 + np.exp(-x2))
        sort_score = sort_score_a * sort_score_b
        # Add outlier as a good data point if the sort score is below threshold
        if np.all(sort_score < threshold):
            good_data_points.append(outlier)
            count += 1
        else:
            sorted_outliers.append(outlier)
            cockoc += 1       
    return np.array(sorted_outliers),good_data_points

# Function to run the Isolation Forest algorithm
def run_algorithm_iforest(data_points, w):
    # Create an Isolation Forest model
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)

    # Fit the model to the data
    isolation_forest.fit(data_points)

    # Predict outlier labels (1 for inliers, -1 for outliers)
    outlier_labels = isolation_forest.predict(data_points)

    # Calculate accuracy (ratio of inliers to total data points)
    accuracy = np.sum(outlier_labels == 1) / len(outlier_labels)
    threshold = 0.99999999999781
    
    gamma = 8
    delta = 4
    gamma_prime = 2
    delta_prime = 1
    # store good data points in a list and outliers in the buffer
    good_data_points = data_points[outlier_labels == 1].tolist()
    outlier_buffer = OutlierBuffer(w)
    outliers_detected = data_points[outlier_labels== -1]
    for outlier in outliers_detected:
        outlier_buffer.add(outlier)
    outliers_detected = outlier_buffer.get_all() 

    # Create an OutlierBuffer for storing outliers
    outlier_buffer = OutlierBuffer(w)

    # Add outliers to the buffer
    for outlier in outliers_detected:
        outlier_buffer.add(outlier)
    outliers_detected = outlier_buffer.get_all()

    # Get sorted outliers
    sorted_outliers,good_data_points = sort_outliers(outliers_detected, good_data_points, np.mean(good_data_points, axis=0), gamma, delta, gamma_prime, delta_prime, threshold) 
    # Calculate filter bounds
    xmin_outlier, xmax_outlier, ymin_outlier, ymax_outlier = calculate_filter_bounds(sorted_outliers)
    # Get filtered outliers based on updated filter bounds
    filtered_outliers = sorted_outliers[
        (sorted_outliers[:, 0] >= xmin_outlier[0]) &
        (sorted_outliers[:, 0] <= xmax_outlier[0]) &
        (sorted_outliers[:, 1] >= ymin_outlier[1]) &
        (sorted_outliers[:, 1] <= ymax_outlier[1])
    ]
    #print("Number of outliers:",len(sorted_outliers))
    good_data_points.extend(filtered_outliers.tolist())
    sorted_outliers_iso_list = sorted_outliers.tolist()
    sorted_outliers_iso_list = [item for item in sorted_outliers_iso_list if item not in filtered_outliers.tolist()]
    return xmin_outlier, xmax_outlier, ymin_outlier, ymax_outlier, len(good_data_points), good_data_points,sorted_outliers_iso_list, accuracy

n = 2
low, high = 0, 30
w = 100
outlier_buffer = OutlierBuffer(w)

good_data = []
filter_outlier=[]
sizes = [100, 200, 500, 1000]
size=1
p=1
# Loop through different data sizes
for size in sizes:
    num_loops = size  # Set num_loops to match the data size
    sumT = 0
    sume = 0

    for p in range(num_loops):
        # Get the current time
        start = time.time()
        data_points = np.zeros((size, n))
        for i in range(data_points.shape[0]):
            low_val, high_val = sorted([random.randint(low, high), random.randint(low, high)])
            data_points[i, 0] = low_val
            data_points[i, 1] = high_val

        # Algorithm
        xmin_outlier, xmax_outlier, ymin_outlier, ymax_outlier, num_matching_points, good_data_points, sorted_outliers, accuracy = run_algorithm_iforest(data_points, w)
        good_data.extend(good_data_points)
        filter_outlier.extend(sorted_outliers)
        end = time.time()
        elapsed_time = end - start

        # Update sums
        sumT += elapsed_time
        sume += num_matching_points
    #print("Number of outliers:",len(sorted_outliers))
    avgT = sumT / num_loops
    avge = sume / num_loops
    total_data_points=len(good_data)+len(filter_outlier)
    print("Total Data points:",total_data_points)
    print("Number of Good data points",len(good_data))
    print("Number of Outliers:",len(filter_outlier))
    print("Data Size:", size)
    print("Average Time:", avgT)
    print("Average Data Points Matching Filter:", avge)
    print("Accuracy:",accuracy)
    print()
    plt.figure()
    plt.title(f"Data Size: {size}")
    plt.scatter(np.array(good_data)[:, 0], np.array(good_data)[:, 1], label="Good Data Points", color='green')
    plt.scatter(np.array(filter_outlier)[:, 0], np.array(filter_outlier)[:, 1], label="Outliers", color='red')
    plt.legend()
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    good_data.clear()
    filter_outlier.clear()