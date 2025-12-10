import glob
import os
import numpy as np
import joblib
from tqdm import tqdm


amass_root = "/mnt/lustre/work/ponsmoll/pba936/AMASS_272"
# amass_root = "/home/miku/Documents/Dataset/AMASS_part_272"

all_pkls = glob.glob(f"{amass_root}/**/*.npy", recursive=True)
amass_occlusion = joblib.load("./data/amass_copycat_occlusion_v3.pkl")
amass_full_motion_dict = {}
filter_list = []
use_list = []
for data_path in tqdm(all_pkls):
    print("Processing", data_path)
    bound = 0

    path_parts = data_path.split("/")
    for _, part in enumerate(path_parts):
        if 'amass' in path_parts or 'AMASS' in part:
            amass_parts = part  # ['AMASS_processed']
    if amass_parts in path_parts:
        amass_index = path_parts.index(amass_parts)
        key_name = path_parts[amass_index + 1:]
    else:
        print("AMASS not found in path:", data_path)
        continue

    key_name_dump = "0-" + "_".join(key_name).replace(".npy", "")


    if key_name_dump in amass_occlusion:
        issue = amass_occlusion[key_name_dump]["issue"]
        if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[key_name_dump]:
            bound = amass_occlusion[key_name_dump]["idxes"][0]  # This bounded is calucaled assuming 30 FPS.....
            filter_list.append(key_name_dump)
            if bound < 10:
                print("bound too small", key_name_dump, bound)
                continue
        else:
            filter_list.append(key_name_dump)
            print("issue irrecoverable", key_name_dump, issue)
            continue

    entry_data = np.load(open(data_path, "rb"), allow_pickle=True)
    N = entry_data.shape[0]

    if "0-KIT_442_PizzaDelivery02_poses" == key_name_dump:
        bound = -2
        filter_list.append(key_name_dump)

    if bound == 0:
        bound = N

    if N < 10:
        continue

    use_list.append(key_name_dump)

output_txt = "success_keys.txt"
with open(output_txt, "w", encoding="utf-8") as f:
    for item in use_list:
        f.write(f"{item}\n")

output_txt = "failed_keys.txt"
with open(output_txt, "w", encoding="utf-8") as f:
    for item in filter_list:
        f.write(f"{item}\n")

print(f"use_list saved to {output_txt}, total {len(use_list)} items.")