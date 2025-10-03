
import os
import pandas as pd
import numpy as np
import torch

data_dir = "./data"
theta_list = []
x_obs_list = []

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        energy = float(file.replace(".csv", ""))  # extract energy from filename
        path = os.path.join(data_dir, file)
        df = pd.read_csv(path)

        # Group by ion#
        for ion_id, group in df.groupby("ion#"):
            coords = group[["x", "y", "z"]].values

            # Summarize this ion's vacancy cloud
            mean_xyz = coords.mean(axis=0)     # [mean_x, mean_y, mean_z]
            count = coords.shape[0]            # number of displaced atoms
            x_obs = np.concatenate([mean_xyz, [count]])  # shape: [4]

            theta_list.append([energy])        # shape: [1]
            x_obs_list.append(x_obs)           # shape: [4]

# Convert to tensors
theta_tensor = torch.tensor(theta_list, dtype=torch.float32)
x_obs_tensor = torch.tensor(x_obs_list, dtype=torch.float32)

# Save to disk
torch.save(theta_tensor, "theta_tensor.pt")
torch.save(x_obs_tensor, "x_obs_tensor.pt")



def preprocess():
   
   
   
   
   
   
    return 