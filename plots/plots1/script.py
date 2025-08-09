# script to update png file names
import os

plots_dir = "plots/plots3" 
files = os.listdir(plots_dir)
png_files = [f for f in files if f.endswith('.png') and not f.startswith('p')]

for old_name in png_files:
    new_name = 'p' + old_name
    old_path = os.path.join(plots_dir, old_name)
    new_path = os.path.join(plots_dir, new_name)
    os.rename(old_path, new_path)
    print(f"Renamed: {old_name} → {new_name}")

print("All files renamed successfully!")