import os
import numpy as np
import pandas as pd
import nibabel as nib
import networkx as nx

###
# 1. From connectomes matrices get the degree of each node in every file. Store node index and degree in a new folder.
###

# Paths
folder_path = '/Users/gabrieledele/Desktop/Assignment/gabriele_sc_connectomes'
cleaned_output_folder = '/Users/gabrieledele/Desktop/Assignment/code&output/1_outputs_cleaned'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Iterate through each CSV file
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)

    # Read CSV file into a NumPy array
    mydata = np.genfromtxt(file_path, delimiter=',')

    # Get degree from graph
    G = nx.Graph(mydata)
    w = G.degree()

    # Post-processing - cleanup
    output_data = np.array(list(w))

    # Remove decimals
    output_data[:, 0] = np.round(output_data[:, 0])

    # Format the second column to keep only the first 3 digits
    output_data[:, 1] = output_data[:, 1]

    # Save results to a new CSV file in cleaned output folder
    cleaned_output_path = os.path.join(cleaned_output_folder, f'cleaned_{csv_file}')
    np.savetxt(cleaned_output_path, output_data, delimiter=',', fmt='%d', comments='')

print("Degrees computed")

###
# 2. Get those nodes that are eligible to be hubs (with degree greater than 1 SD from mean)
###

# Folder paths
cleaned_folder = '/Users/gabrieledele/Desktop/Assignment/code&output/1_outputs_cleaned'
hub_folder = '/Users/gabrieledele/Desktop/Assignment/code&output/2_hub'

# List all CSV files
cleaned_files = [f for f in os.listdir(cleaned_folder) if f.endswith('.csv')]

for cleaned_file in cleaned_files:
    cleaned_file_path = os.path.join(cleaned_folder, cleaned_file)

    # Load data from CSV file
    cleaned_data = pd.read_csv(cleaned_file_path, header=None)

    # Calculate mean and standard deviation of the second column
    mean_value = cleaned_data.iloc[:, 1].mean()
    std_dev = cleaned_data.iloc[:, 1].std()

    # Create a new dataframe with values greater than 1 SD from the mean
    hub_data = cleaned_data[cleaned_data.iloc[:, 1] > (mean_value + std_dev)]

    # Save to a CSV file in the hub folder
    hub_file_path = os.path.join(hub_folder, cleaned_file)
    hub_data.to_csv(hub_file_path, index=False, header=False)

print("Hub files created successfully.")

###
# 3. Get unique hub file
###

# Folder paths
hub_folder = '/Users/gabrieledele/Desktop/Assignment/code&output/2_hub'
unique_hub_file_path = '/Users/gabrieledele/Desktop/Assignment/code&output/3_unique_hub/unique_hub.csv'

# List all CSV files hub folder
hub_files = [f for f in os.listdir(hub_folder) if f.endswith('.csv')]

# Dictionary to store data for each index
index_data = {}

# Track how many files each index is present in
index_presence_counter = {}

# Iterate through each hub file
for hub_file in hub_files:
    hub_file_path = os.path.join(hub_folder, hub_file)

    # Load the data from hub CSV file
    hub_data = pd.read_csv(hub_file_path, header=None)

    # Iterate through each row in the current file
    for _, row in hub_data.iterrows():
        index = row.iloc[0]
        value = float(row.iloc[1])

        # Update the data dictionary for the current index
        if index in index_data:
            index_data[index].append(value)
            index_presence_counter[index] += 1
        else:
            index_data[index] = [value]
            index_presence_counter[index] = 1

# Filter indices that are present in more than n% of files
common_indices = [index for index, count in index_presence_counter.items() if count >= 0.95 * len(hub_files)]

# Create a df with the average values for each common index
average_data = {'Index': [], 'AverageValue': []}

for index in common_indices:
    # Get average value for the current index
    average_value = np.mean(index_data[index])
    average_data['Index'].append(index)
    average_data['AverageValue'].append(average_value)

# Convert dictionary to a DataFrame
unique_hub_data = pd.DataFrame(average_data)
unique_hub_data = unique_hub_data.round({'AverageDegreeVal': 1})

# Save to CSV file
unique_hub_data.to_csv(unique_hub_file_path, index=False)

print("Unique hub file created")

###
# 4. Use any adjacency matrix to extract the coordinates
###

# Load the brain image and brain atlas
t1_image = nib.load('/Users/gabrieledele/Desktop/Assignment/templates_atlas/MNI152_T1_2mm.nii')
brain_atlas = nib.load('/Users/gabrieledele/Desktop/Assignment/templates_atlas/yeo_tian_2mm.nii')

# Load the adjacency matrix
adjacency_matrix = pd.read_csv('/Users/gabrieledele/Desktop/Assignment/gabriele_sc_connectomes/connectome_sub-100206.csv', header=None)

# Extract the data from the memmap
atlas_data = brain_atlas.get_fdata()

# Extract the unique labels from the brain atlas
unique_labels = np.unique(brain_atlas.get_fdata())

# Convert voxel coordinates to world coordinates
def voxel_to_world_coordinates(voxel_coords, affine_matrix):
    world_coords = [affine_matrix[:3, :3].dot(voxel) + affine_matrix[:3, 3] for voxel in voxel_coords]
    return np.array(world_coords)

# Iterate through unique labels in the atlas
node_coordinates = []
for label in unique_labels:
    # Find voxel indices with the current label
    indices = np.where(brain_atlas.get_fdata() == label)
    
    # Use the centroid of the voxels as node coordinates
    centroid = np.mean(np.array(indices), axis=1)
    node_coordinates.append(centroid)

# Convert voxel coordinates to world coordinates
world_coordinates = voxel_to_world_coordinates(node_coordinates, t1_image.affine)

w_c2 = pd.DataFrame(world_coordinates)
#w_c2.head()
w_c2 = w_c2.drop([0])
#w_c2.head()
adjacency_matrix.head()
adjacency_matrix.reset_index(drop=True, inplace=True)
w_c2.reset_index(drop=True, inplace=True)

adjacency_matrix_with_coordinates = pd.concat([adjacency_matrix, w_c2], axis=1)

adjacency_matrix_with_coordinates.to_csv('/Users/gabrieledele/Desktop/Assignment/code&output/4_adj_coord/adjacency_matrix_with_coordinates.csv')
print("Adjacency with coordinates created")

###
# 5. Get hub coordinates
###

# Load unique hub file
unique_hub_data = pd.read_csv('/Users/gabrieledele/Desktop/Assignment/code&output/3_unique_hub/unique_hub.csv', index_col=0)

# Load the CSV file containing X, Y, Z coordinates
coordinates_data = pd.read_csv('/Users/gabrieledele/Desktop/Assignment/code&output/4_adj_coord/adjacency_matrix_with_coordinates.csv', index_col=0)

new_coord = coordinates_data.iloc[:,-3:]
new_coord.columns = ["X", "Y", "Z"]
new_coord.index.name = 'Index'

new_coord.index += 1

# Merge dataframes based on the common 'Index' column
merged_data = unique_hub_data.merge(new_coord, left_on='Index', right_on='Index')
merged_data = merged_data.sort_values(by=['Index'])

# Save the merged DataFrame to a new CSV file
merged_data.to_csv('/Users/gabrieledele/Desktop/Assignment/code&output/5_hub_coord/hub_with_coordinates.csv', index=True)

print("Hub with coordinates created")

###
# 6. Get .node file
###

# Load hub with coordinates file
hub_coord_path = '/Users/gabrieledele/Desktop/Assignment/code&output/5_hub_coord/hub_with_coordinates.csv'
hub_coord = pd.read_csv(hub_coord_path, delimiter=',', skipinitialspace=True)

# Extract x y z columns
coordinates = hub_coord[['X', 'Y', 'Z']]

# Prepare data frame
node_data = pd.DataFrame({
    'X': coordinates['X'],
    'Y': coordinates['Y'],
    'Z': coordinates['Z'],
})

# Concatenate columns with space and 1s
node_data['Coordinates'] = node_data['X'].astype(str) + " " + node_data['Y'].astype(str) + " " + node_data['Z'].astype(str) + " " + "1" + " " + "1"
node_data = node_data['Coordinates']

# Save .node file
output_path = '/Users/gabrieledele/Desktop/Assignment/code&output/6_node_file/file.node'
node_data.to_csv(output_path, sep='\t', header=False, index=False)

print(".node file created")
