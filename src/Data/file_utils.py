import os
import re

def change_comma(frame):
  new_frame = frame.copy()
  for col in frame.columns:
    new_frame[col] = frame[col].astype(str)
    new_frame[col] = frame[col].str.replace(',','.',regex=False)
  return new_frame

#================================================================
#================================================================
#================================================================

def format_path(path):
    """""Formats the path string in order to avoid conflicts."""
    if path[-1]!='/':
        path = path + '/'

    if not os.path.exists(path):
        os.makedirs(path)
    return path



def create_next_experiment_folder(base_path):
    os.makedirs(base_path, exist_ok=True)
    
    existing = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and d.startswith("exp_")
    ]
    
    numbers = []
    for folder in existing:
        match = re.match(r"exp_(\d+)", folder)
        if match:
            numbers.append(int(match.group(1)))
            
    next_number = max(numbers)+1 if numbers else 1
    new_folder_name = f"exp_{next_number:03d}"
    
    full_path = os.path.join(base_path, new_folder_name)
    os.makedirs(full_path)
    
    return full_path