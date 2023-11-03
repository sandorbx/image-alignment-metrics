import tkinter
from tkinter import filedialog
from tkinter import simpledialog
from threading import Lock
import csv
import os
import nrrd
import numpy as np
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor

# Initialize Tkinter
root = tkinter.Tk()
root.withdraw()

parallel_limit = simpledialog.askinteger("Parallel Processing", "Enter the number of images to process in parallel:", minvalue=1)
if not parallel_limit:
    raise ValueError("You must specify a valid number for parallel processing. Exiting...")



template_file_path = filedialog.askopenfilename(title="Select the template image file", filetypes=[("NRRD files", "*.nrrd")])
if not template_file_path:
    raise ValueError("No template image file selected! Exiting...")


template, _  = nrrd.read(template_file_path)


nrrd_dir = filedialog.askdirectory(title="Select the directory containing the NRRD files")
if not nrrd_dir:
    raise ValueError("No directory selected! Exiting...")

nrrd_dir = os.path.join(nrrd_dir, '')

# metrics and append to metrics_list

def process_image(future, image_file, template, metrics_list):
    data2, _ = future.result()
    metrics = calc_metrics(template, data2)
    metrics["Image"] = image_file  # Add image file name to metrics dictionary
    with metrics_lock:  # Ensure thread-safe append operation
        metrics_list.append(metrics)
        
def write_to_csv(metrics_list, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(["Image", "NMI", "MSE", "Correlation Coefficient", "NCC"])
        # Write rows
        for metrics in metrics_list:
            csv_writer.writerow([metrics["Image"], metrics["NMI"], metrics["MSE"], metrics["Correlation Coefficient"], metrics["NCC"]])

# entropy
def calc_entropy(hist, total_pixels):
    prob = hist / total_pixels
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log(prob))
    return entropy

# joint entropy
def calc_joint_entropy(joint_hist, total_pixels):
    joint_prob = joint_hist / total_pixels
    joint_prob = joint_prob[joint_prob > 0]
    joint_entropy = -np.sum(joint_prob * np.log(joint_prob))
    return joint_entropy

# metrics

# normalized cross correlation (NCC) between two images
def calc_ncc(template, data2):
    mean_template = np.mean(template)
    mean_data2 = np.mean(data2)

    numerator = np.sum((template - mean_template) * (data2 - mean_data2))
    denominator = np.sqrt(np.sum((template - mean_template) ** 2) * np.sum((data2 - mean_data2) ** 2))

    if denominator == 0:
        return 0  # Avoid division by zero
    else:
        return numerator / denominator
def calc_metrics(template, data2):
    results = {}
    if template.shape != data2.shape:
        results["error"] = "The images must have the same dimensions."
        return results

    
    # NMI calculation
    hist1, _ = np.histogram(template, bins=256, range=(0, 256))
    hist2, _ = np.histogram(data2, bins=256, range=(0, 256))
    joint_hist, _, _ = np.histogram2d(template.flatten(), data2.flatten(), bins=256, range=[[0, 256], [0, 256]])
    total_pixels = template.size
    entropy1 = calc_entropy(hist1, total_pixels)
    entropy2 = calc_entropy(hist2, total_pixels)
    joint_entropy = calc_joint_entropy(joint_hist, total_pixels)
    mutual_information = entropy1 + entropy2 - joint_entropy
    normalized_mutual_information = 2 * mutual_information / (entropy1 + entropy2)
    
    # MSE
    mse = mean_squared_error(template.flatten(), data2.flatten())
    
    # Correlation Coefficient
    corr_coeff = np.corrcoef(template.flatten(), data2.flatten())[0, 1]
    
    # NCC
    ncc = calc_ncc(template, data2)

    results["NMI"] = normalized_mutual_information
    results["MSE"] = mse
    results["Correlation Coefficient"] = corr_coeff
    results["NCC"] = ncc
    return results

metrics_list = []
metrics_lock = Lock()



nrrd_files = [f for f in os.listdir(nrrd_dir) if f.endswith('.nrrd')]

    
# Use multi-threading 
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(nrrd.read, os.path.join(nrrd_dir, f)) for f in nrrd_files]
    with ThreadPoolExecutor() as metrics_executor:
            
        metrics_futures = [metrics_executor.submit(process_image, future, image_file, template, metrics_list) 
                           for future, image_file in zip(futures, nrrd_files)]
            
        
        # Wait for all threads
        for future in metrics_futures:
                future.result()

csv_file_path = os.path.join(nrrd_dir, f"{os.path.basename(template_file_path).split('.')[0]}+correlation-metrics.csv")
write_to_csv(metrics_list, csv_file_path)


