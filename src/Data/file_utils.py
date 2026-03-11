import os

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


