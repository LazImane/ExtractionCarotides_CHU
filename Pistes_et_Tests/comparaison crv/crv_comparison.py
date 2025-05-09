import numpy as np
import matplotlib.pyplot as plt

def read_activity_file(filename):
    values = []
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            _, val = line.strip().split('\t')
            val = float(val)
            if val > 0:
                values.append(val)
    return np.array(values)

def compare_activity_files(file1, file2):
    data1 = read_activity_file(file1)
    data2 = read_activity_file(file2)
    
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]
    
    diff = data1 - data2
    abs_diff = np.abs(diff)
    rel_diff = np.abs(diff) / ((data1 + data2)/2) * 100  
    
    print(f"\nComparing {file1} vs {file2}")
    print(f"Number of comparable points: {min_len}")
    print(f"Mean absolute difference: {np.mean(abs_diff):.4f} kBq/cc")
    print(f"Mean relative difference: {np.mean(rel_diff):.2f}%")
    
    plt.figure(figsize=(12, 6))
    plt.plot(data1, 'b-', label=file1)
    plt.plot(data2, 'r-', label=file2)
    plt.xlabel("Time index (non-zero points)")
    plt.ylabel("Activity (kBq/cc)")
    plt.title("Activity Curves Comparison")
    plt.legend()
    plt.grid()
    plt.show()
    
    return {
        'data1': data1,
        'data2': data2,
        'abs_diff': abs_diff,
        'rel_diff': rel_diff
    }

if __name__ == "__main__":
    # Example usage - replace with your actual files
    file1 = "P04.txt"
    file2 = "P04_carot.crv"
    
    results = compare_activity_files(file1, file2)