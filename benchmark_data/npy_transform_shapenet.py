import os

file_path = "\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\ShapeNet-34\train.txt"

"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\ShapeNet-34\test.txt"



with open(file_path, 'r') as file:
    for line in file:
        # Process each line
        print(line.strip())  # Use .strip() to remove any leading/trailing whitespace


def list_all_files(directory_path):
    items = os.listdir(directory_path)
    files = [os.path.join(directory_path, item) for item in items if os.path.isfile(os.path.join(directory_path, item))]
    return files

directory_path = r"\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\shapenet_pc" 
files = list_all_files(directory_path)
print(len(files))





# "\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\ShapeNet-55"
# "\\datanasop3mech\ProjectData\3_phd\Stuti\PCC&PSS\Code\ODGNet\data\ShapeNet55-34\ShapeNet-Unseen21"
