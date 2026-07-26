import os
import pandas as pd

# Folder containing the CSV files
input_folder = r"E:\Mahmoud\Exams\46\462\New-papers\Paper6\NewDataSets\Data1\Eye-tracking Output"

# List to hold the results
results = []

# Loop through each CSV file in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        try:
            df = pd.read_csv(file_path, low_memory=False)
            if "Participant" in df.columns:
                value_counts = df["Participant"].value_counts(dropna=False)
                for value, count in value_counts.items():
                    results.append({
                        "file_name": filename,
                        "value": value,
                        "frequency": count
                    })
            else:
                print(f"'Participant' column not found in {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Convert the results to a DataFrame and save or display
output_df = pd.DataFrame(results)
output_path = os.path.join(input_folder, "participant_value_frequencies.csv")
output_df.to_csv(output_path, index=False)

print(f"Done. Output saved to: {output_path}")
