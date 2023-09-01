import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest  
import random
import time
import matplotlib.pyplot as plt
import warnings

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
    #print("data points before the sorting function",len(good_data_points))
    #print("outliers before sorting",outliers_detected)
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
    good_data_points.extend(filtered_outliers.tolist())

    sorted_outliers_iso_list = sorted_outliers.tolist()

    # Create a new list without the elements in filtered_outliers
    sorted_outliers_iso_list = [item for item in sorted_outliers_iso_list if item not in filtered_outliers.tolist()]


    return xmin_outlier, xmax_outlier, ymin_outlier, ymax_outlier, len(good_data_points), good_data_points,sorted_outliers_iso_list, accuracy


def run_algorithm_svm(data_points, w):
    # fit the One-Class SVM model
    svm = OneClassSVM(kernel='linear', nu=0.05)
    svm.fit(data_points)

    y_pred = svm.predict(data_points)

    accuracy = np.sum(y_pred == 1) / len(y_pred)
    # store good data points in a list and outliers in the buffer
    good_data_points_svm= data_points[y_pred == 1].tolist()
    outlier_buffer_svm = OutlierBuffer(w)
    outliers_detected_svm = data_points[y_pred == -1]
    for outlier in outliers_detected_svm:
        outlier_buffer_svm.add(outlier)
    outliers_detected_svm = outlier_buffer_svm.get_all()    
    threshold = 0.99999999999781  # threshold

    gamma = 8
    delta = 4
    gamma_prime = 2
    delta_prime = 1
    # Get sorted outliers
    sorted_outliers_svm,good_data_points_svm = sort_outliers(outliers_detected_svm, good_data_points_svm, np.mean(good_data_points_svm, axis=0), gamma, delta, gamma_prime, delta_prime, threshold) 

    # Calculate filter bounds
    xmin_outlier, xmax_outlier, ymin_outlier, ymax_outlier = calculate_filter_bounds(sorted_outliers_svm)

    # Get filtered outliers based on updated filter bounds
    filtered_outliers = sorted_outliers_svm[
        (sorted_outliers_svm[:, 0] >= xmin_outlier[0]) &
        (sorted_outliers_svm[:, 0] <= xmax_outlier[0]) &
        (sorted_outliers_svm[:, 1] >= ymin_outlier[1]) &
        (sorted_outliers_svm[:, 1] <= ymax_outlier[1])
    ]

    good_data_points_svm.extend(filtered_outliers.tolist())
    # Convert NumPy array to a Python list
    sorted_outliers_svm_list = sorted_outliers_svm.tolist()
    sorted_outliers_svm_list = [item for item in sorted_outliers_svm_list if item not in filtered_outliers.tolist()]
    return xmin_outlier, xmax_outlier, ymin_outlier, ymax_outlier, len(good_data_points_svm), good_data_points_svm, sorted_outliers_svm_list, accuracy

n = 2
low, high = 0, 30
w = 100
#outlier_buffer = OutlierBuffer(w)

good_data_svm = []
filter_outlier_svm=[]
periergo_svm=[]
good_data_iso = []
filter_outlier_iso=[]
periergo_iso=[]
sizes = [100, 200,500,1000]
size=1
p=1
avgT_total_svm = 0
avge_total_svm = 0
accuracy_total_svm = 0
overall_accuracy_svm=0

avgT_total_iso = 0
avge_total_iso = 0
accuracy_total_iso = 0
overall_accuracy_iso=0
# Lists to store results for each algorithm
avgT_svm_list = []
avge_svm_list = []
accuracy_svm_list = []

avgT_iso_list = []
avge_iso_list = []
accuracy_iso_list = []
# Loop through different data sizes
for size in sizes:
    num_loops = size  # Set num_loops to match the data size
    sumT_svm = 0
    sume_svm = 0
    sumacc_svm=0
    sumT_iso = 0
    sume_iso = 0
    sumacc_iso=0
    for p in range(num_loops):
        # Get the current time
        start = time.time()
        data_points = np.zeros((size, n))
        for i in range(data_points.shape[0]):
            low_val, high_val = sorted([random.randint(low, high), random.randint(low, high)])
            data_points[i, 0] = low_val
            data_points[i, 1] = high_val

        #Algorithm
        # Run the SVM algorithm and get required values
        xmin_outlier, xmax_outlier, ymin_outlier, ymax_outlier, num_matching_points_svm, good_data_points_svm, sorted_outliers_svm ,accuracy_svm= run_algorithm_svm(data_points, w)
        # Get the new time and calculate elapsed time
        end = time.time()
        elapsed_time_svm = end - start
        
        # Update sums for SVM
        sumT_svm += elapsed_time_svm
        sume_svm += num_matching_points_svm
        good_data_svm.extend(good_data_points_svm)
        filter_outlier_svm.extend(sorted_outliers_svm)
        

        # Run the isolation forest algorithm and get required values
        xmin_outlier, xmax_outlier, ymin_outlier, ymax_outlier, num_matching_points_iso, good_data_points_iso, sorted_outliers_iso,accuracy_iso= run_algorithm_iforest(data_points, w)
        sumacc_svm += accuracy_svm
        sumacc_iso+= accuracy_iso
        # Get the new time and calculate elapsed time
        end = time.time()
        elapsed_time_iso = end - start

        # Update sums for iso
        sumT_iso += elapsed_time_iso
        sume_iso += num_matching_points_iso
        good_data_iso.extend(good_data_points_iso)
        filter_outlier_iso.extend(sorted_outliers_iso)
        


    # Calculate averages for SVM
    avgT_svm = sumT_svm / num_loops
    avge_svm = sume_svm / num_loops 
    
    # Calculate averages for iso
    avgT_iso = sumT_iso / num_loops
    avge_iso = sume_iso / num_loops 
    
    # Update overall totals for SVM
    avgT_total_svm += avgT_svm
    avge_total_svm += avge_svm
    accuracy_total_svm += sumacc_svm/num_loops
    
    # Update overall totals for iso
    avgT_total_iso += avgT_iso
    avge_total_iso += avge_iso
    accuracy_total_iso += sumacc_iso/num_loops

    # Append results to respective lists for SVM
    avgT_svm_list.append(avgT_svm)
    avge_svm_list.append(avge_svm)
    accuracy_svm_list.append(accuracy_total_svm)
    
    # Append results to respective lists for iso
    avgT_iso_list.append(avgT_iso)
    avge_iso_list.append(avge_iso)
    accuracy_iso_list.append(accuracy_total_iso)

    # Print results for the current data size
    print("Data Size:", size)
    print("SVM:")
    print("Number of Good data points",len(good_data_svm))
    print("Number of Outliers:",len(filter_outlier_svm))
    print("Average Time:", avgT_svm)
    print("Average Data Points Matching Filter:", "{:.2f}".format(avge_svm))
    print("Accuracy:", "{:.4f}".format(accuracy_total_svm))
    print()
    print("IForest:")
    print("Number of Good data points",len(good_data_iso))
    print("Number of Outliers:",len(filter_outlier_iso))
    print("Average Time:", avgT_iso)
    print("Average Data Points Matching Filter:", "{:.2f}".format(avge_iso))
    print("Accuracy:","{:.4f}".format(accuracy_total_iso))
    print()
    overall_accuracy_svm+=accuracy_total_svm
    overall_accuracy_iso+=accuracy_total_iso
    accuracy_total_iso = 0
    accuracy_total_svm = 0
# Calculate overall averages for SVM
avgT_total_svm /= len(sizes)
avge_total_svm /= len(sizes)
overall_accuracy_svm /= len(sizes)

# Calculate overall averages for iso
avgT_total_iso /= len(sizes)
avge_total_iso /= len(sizes)
overall_accuracy_iso /= len(sizes)

# Print overall results
print("SVM:")
print("Overall Average Time:",avgT_total_svm)
print("Overall Average Data Points Matching Filter:", "{:.2f}".format(avge_total_svm))
print("Overall Accuracy:", "{:.4f}".format(overall_accuracy_svm))
print()
print("IForest:")
print("Overall Average Time:", avgT_total_iso)
print("Overall Average Data Points Matching Filter:", "{:.2f}".format(avge_total_iso))
print("Overall Accuracy:", "{:.4f}".format(overall_accuracy_iso))
plt.figure(figsize=(10, 6))
plt.plot(sizes, avgT_svm_list, label='SVM', marker='o')
plt.plot(sizes, avgT_iso_list, label='IForest', marker='o')
plt.xlabel('Data Size')
plt.ylabel('Average Time')
plt.title('Average Time Comparison: SVM vs IForest')
plt.legend()
plt.grid()
plt.show()

# Plot results for Average Data Points Matching Filter
plt.figure(figsize=(10, 6))
plt.plot(sizes, avge_svm_list, label='SVM', marker='o')
plt.plot(sizes, avge_iso_list, label='IForest', marker='o')
plt.xlabel('Data Size')
plt.ylabel('Average Data Points Matching Filter')
plt.title('Average Data Points Matching Filter Comparison: SVM vs IForest')
plt.legend()
plt.grid()
plt.show()

# Plot results for Accuracy
plt.figure(figsize=(10, 6))
plt.plot(sizes, accuracy_svm_list, label='SVM', marker='o')
plt.plot(sizes, accuracy_iso_list, label='IForest', marker='o')
plt.xlabel('Data Size')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: SVM vs IForest')
plt.legend()
plt.grid()
plt.show()

# Calculate the overall number of good data points and outliers for SVM and Isolation Forest
overall_good_data_svm = len(good_data_svm)
overall_outliers_svm = len(filter_outlier_svm)
overall_good_data_iso = len(good_data_iso)
overall_outliers_iso = len(filter_outlier_iso)

# Create a bar plot to compare the overall good data points and outliers
plt.figure(figsize=(10, 6))
labels = ['SVM', 'Isolation Forest']
good_data_counts = [overall_good_data_svm, overall_good_data_iso]
outlier_counts = [overall_outliers_svm, overall_outliers_iso]
width = 0.35
x = range(len(labels))

fig, ax = plt.subplots()
rects1 = ax.bar(x, good_data_counts, width, label='Good Data Points', color='green')
rects2 = ax.bar([i + width for i in x], outlier_counts, width, label='Outliers', color='red')

ax.set_xlabel('Algorithm')
ax.set_ylabel('Count')
ax.set_title('Overall Good Data Points and Outliers Comparison: SVM vs Isolation Forest')
ax.set_xticks([i + width / 2 for i in x])
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# Create a scatter plot for SVM good data points and outliers
plt.figure(figsize=(10, 6))

# Plot SVM good data points in green
plt.scatter(np.array(good_data_svm)[:, 0], np.array(good_data_svm)[:, 1], label="SVM Good Data Points", color='green')

# Plot SVM outliers in red
plt.scatter(np.array(filter_outlier_svm)[:, 0], np.array(filter_outlier_svm)[:, 1], label="SVM Outliers", color='red')

plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title('SVM Good Data Points and Outliers')
plt.grid(True)
plt.show()

# Create a scatter plot for Isolation Forest good data points and outliers
plt.figure(figsize=(10, 6))

# Plot Isolation Forest good data points in blue
plt.scatter(np.array(good_data_iso)[:, 0], np.array(good_data_iso)[:, 1], label="Isolation Forest Good Data Points", color='blue')

# Plot Isolation Forest outliers in orange
plt.scatter(np.array(filter_outlier_iso)[:, 0], np.array(filter_outlier_iso)[:, 1], label="Isolation Forest Outliers", color='orange')

plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title('Isolation Forest Good Data Points and Outliers')
plt.grid(True)
plt.show()
