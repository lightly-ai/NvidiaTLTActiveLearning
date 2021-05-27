# Mapping up the local directories to the TLT docker.
import os
import json
mounts_file = os.path.expanduser("~/.tlt_mounts.json")

# Define the dictionary with the mapped drives
drive_map = {
    "Mounts": [
        # Mapping the data directory
        {
            "source": os.getcwd(),
            "destination": "/workspace/tlt-experiments"
        },
        # Mapping the specs directory.
        {
            "source": f"{os.getcwd()}/yolo_v4/specs",
            "destination": "/workspace/tlt-experiments/yolo_v4/specs"
        },
    ]
}

# Writing the mounts file.
with open(mounts_file, "w") as mfile:
    json.dump(drive_map, mfile, indent=4)
