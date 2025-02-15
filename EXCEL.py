import pandas as pd

# Define the path to your text file
txt_file = "HFL1_RVE_CON_12.txt"  # Replace with your file path

# Read the text file into a DataFrame.
# In this example, we assume the first two lines are headers that we want to skip.
df = pd.read_csv(txt_file, delim_whitespace=True, skiprows=2, header=None)

# Optionally, set column names if needed.
df.columns = ["Part_Instance", "Node_ID", "Attached_elements", "HFL_HFL1"]

# Define the output Excel file name
excel_file = "output.xlsx"

# Write the DataFrame to an Excel file.
df.to_excel(excel_file, index=False)

print("Data has been extracted and saved to", excel_file)     