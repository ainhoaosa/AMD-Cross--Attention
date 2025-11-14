import os
import pandas as pd
import numpy as np
from faker import Faker

# Initialize the fake data generator
fake = Faker()

# Function to generate synthetic metadata based on image folders
def generate_synthetic_metadata_from_folders(base_folder):
    metadata = []
    
    # List main folders (each folder represents an image class)
    classes = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
    print(f"Found classes: {classes}")
    for class_name in classes:
        class_folder = os.path.join(base_folder, class_name)
        
        # Use os.walk to iterate through all subfolders and image files
        image_files = []
        for root, _, files in os.walk(class_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):  # Include common extensions
                    image_files.append(os.path.join(root, file))
        
        print(f"Found {len(image_files)} images in class '{class_name}'")
        for image_file in image_files:
            # Generate synthetic data for each image
            patient_id = fake.unique.uuid4()
            age = np.random.randint(50, 90)  # Age between 50 and 90 years
            sex = np.random.choice(['M', 'F'])  # Random sex
            treatment = np.random.choice(['ranibizumab', 'bevacizumab'])  # Random treatment
            dosing_regimen = np.random.choice(['monthly', 'quarterly', 'as_needed'])  # Dosing regimen
            clinical_center = np.random.randint(1, 44)  # Random clinical center ID (1 to 43)
            
            # Append data to the list, including the image filename
            metadata.append({
                'amd': class_name,  # Class name (AMD or DR)
                'image_file': image_file,  # Image file path
                'age': age,
                'sex': sex,
                'treatment': treatment,
                'dosing_regimen': dosing_regimen,
                'clinical_center': clinical_center
            })
    
    # Convert the list of dictionaries into a DataFrame
    metadata_df = pd.DataFrame(metadata)
    return metadata_df

# Base path where the image folders are organized
base_folder = 'fundus/Fundus Dataset A and B'  # Update this path to your main folder

# Generate synthetic metadata based on the folders
synthetic_metadata = generate_synthetic_metadata_from_folders(base_folder)

# Display the first rows of the DataFrame
print(synthetic_metadata.head())

# Save the synthetic metadata to a CSV file
synthetic_metadata.to_csv('synthetic_metadata_A_and_B.csv', index=False)
