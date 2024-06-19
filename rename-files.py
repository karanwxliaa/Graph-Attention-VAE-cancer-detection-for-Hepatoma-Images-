import os

def rename_files(root_dir):
  """Renames all files in subdirectories of root_dir with format '<categoryName>_image_<index>'."""
  for subdir, _, files in os.walk(root_dir):
    category_name = os.path.basename(subdir)  # Get category name from subdir name
    index = 0
    for filename in files:
      new_filename = f"{category_name}_image_{index}.jpg"
      os.rename(os.path.join(subdir, filename), os.path.join(subdir, new_filename))
      index += 1

# Replace 'path/to/your/root/directory' with your actual root directory path
root_dir = 'D:\VIT Material\VIT material\Hepatoma Research Project\Histopathology-Images'
rename_files(root_dir)
print(f"Files renamed in all subdirectories of {root_dir}")
