import os, shutil

folder = '1/'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    if '4_' in file_path:
        if os.path.isfile(file_path):
            if os.path.isfile(file_path):
                shutil.rmtree(file_path)


                
