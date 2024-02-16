import numpy as np

def compare_npz_files(file1, file2):
    # Load data from the first npz file
    data1 = np.load(file1)

    # Load data from the second npz file
    data2 = np.load(file2)

    # Check if the files have the same keys
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    if keys1 != keys2:
        print("The npz files have different keys.")
        print(f"Keys in {file1}: {keys1}")
        print(f"Keys in {file2}: {keys2}")
        return

    # Compare the contents of the files
    for key in keys1:
        if not np.array_equal(data1[key], data2[key]):
            print(f"Difference found in key: {key}")
            print(f"Shape in {file1}: {data1[key].shape}")
            print(f"Shape in {file2}: {data2[key].shape}")

            num_different_points = np.sum(data1[key] != data2[key])
            print(f"Number of different points: {num_different_points}")

            print(f"Data in {file1}:")
            print(data1[key])
            print(f"Data in {file2}:")
            print(data2[key])
        else:
            print(f"Data for key {key} is identical in both files.")

# Example usage:
if __name__ == "__main__":
    file1 = "/root/2TreeLearn/data/pipeline/RTC_dense/results/pointwise_results/pointwise_results.npz"  # Path to the first npz file
    file2 = "/root/TreeLearn/data/pipeline/RTC_dense/results/pointwise_results/pointwise_results.npz"  # Path to the second npz file

    compare_npz_files(file1, file2)
