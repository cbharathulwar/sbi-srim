import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Suppress MPS fallback warnings
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import your loader
from src.utils.data_utils import load_and_pad_3d_tracks

def plot_3d_track(points, true_vec, pred_vec, error_deg, ion_idx, energy, num_vacs):
    """Creates an interactive 3D plot of the track and the vectors."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the point cloud (vacancies)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c='blue', alpha=0.3, s=10, label='Vacancies')
    
    # Scale vectors for visual clarity (make them span the track length)
    scale = np.max(np.abs(points)) if len(points) > 0 else 50
    
    # True Vector (Green)
    ax.quiver(0, 0, 0, true_vec[0]*scale, true_vec[1]*scale, true_vec[2]*scale, 
              color='green', arrow_length_ratio=0.15, linewidth=3, label='True Vector')
    
    # Predicted Vector (Red)
    ax.quiver(0, 0, 0, pred_vec[0]*scale, pred_vec[1]*scale, pred_vec[2]*scale, 
              color='red', arrow_length_ratio=0.15, linewidth=3, label='Predicted Vector')
    
    # UPDATED TITLE to include Energy and Vacancies
    ax.set_title(f"Track {ion_idx} | {energy:.1f} keV | {num_vacs} Vacancies | Error: {error_deg:.1f}°")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([points[:,0].max()-points[:,0].min(), 
                          points[:,1].max()-points[:,1].min(), 
                          points[:,2].max()-points[:,2].min()]).max() / 2.0
    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    posterior = torch.load("results/pipeline_b/pointnet_3d_posterior.pt", map_location=device)
    eval_csv = "data/mcpe-3d/mcpe_3d_eval_10k.csv"
    
    print("Loading data...")
    features, targets_tensor = load_and_pad_3d_tracks(eval_csv, max_points="auto")
    targets = targets_tensor.numpy()
    
    # Get original ion numbers to map back to the raw DataFrame
    df = pd.read_csv(eval_csv)
    ion_numbers = list(df.groupby('ion_number').groups.keys())

    print("Running quick inference in batches...")
    flow = posterior.posterior_estimator
    
    all_pred_means = []
    batch_size = 500  # Safe chunk size for Mac VRAM
    
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            print(f"   Processing track {i} to {min(i+batch_size, len(features))}...")
            batch_ctx = features[i:i+batch_size].to(device)
            
            try:
                samples = flow.sample((10,), condition=batch_ctx)
            except TypeError:
                samples = flow.sample((10,), cond_inputs=batch_ctx)
                
            if samples.shape[0] != 10:
                samples = samples.permute(1, 0, 2)
                
            samples = samples.cpu()
            batch_means = torch.mean(samples, dim=0).numpy()
            all_pred_means.append(batch_means)

    # Combine all batches into one big array
    pred_means = np.concatenate(all_pred_means, axis=0)

    # Calculate angles
    pred_v = pred_means[:, 1:4]
    true_v = targets[:, 1:4]
    
    pred_v = pred_v / (np.linalg.norm(pred_v, axis=1, keepdims=True) + 1e-9)
    dot_prods = np.sum(pred_v * true_v, axis=1)
    angles = np.degrees(np.arccos(np.clip(dot_prods, -1.0, 1.0)))
    
    # Hunt for the ricochets!
    ricochet_indices = np.where((angles >= 70) & (angles <= 100))[0]
    print(f"\nFound {len(ricochet_indices)} tracks with 70°-100° error.")
    
    for idx in ricochet_indices:
        error = angles[idx]
        ion_idx = ion_numbers[idx]
        
        # EXTRACT ENERGY AND VACANCY COUNT
        true_energy = targets[idx, 0] # Energy is the 0th column in your targets tensor
        
        # Get the raw unpadded points for plotting
        raw_points = df[df['ion_number'] == ion_idx][['x', 'y', 'z']].values
        num_vacancies = len(raw_points)
        
        # Center them so the vectors align perfectly
        raw_points = raw_points - np.mean(raw_points, axis=0)
        
        # UPDATED PRINT STATEMENT
        print(f"Plotting Ion {ion_idx} | Energy: {true_energy:.2f} keV | Vacancies: {num_vacancies} | Error: {error:.1f}°")
        
        # Pass the new info to the plotter
        plot_3d_track(raw_points, true_v[idx], pred_v[idx], error, ion_idx, true_energy, num_vacancies)

if __name__ == "__main__":
    main()