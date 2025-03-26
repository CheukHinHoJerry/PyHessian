import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pyhessian import hessian  # Hessian computation
from mace import data
from mace.tools import torch_geometric, torch_tools, utils
from mace.modules import WeightedEnergyForcesLoss
from density_plot import get_esd_plot # ESD plot
import ase.io

import random


def get_params(model_orig, model_perb, direction, alpha):
    """Perturb model parameters along a given direction."""
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

def parse_args():
    parser = argparse.ArgumentParser(description="Hessian Analysis of MACE Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration xyz file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run the computation on")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for data loading")
    parser.add_argument("--Ndata", type=int, default=10, help="Number of data to include, just ascending order of xyz")
    parser.add_argument("--shuffle_seed", type=int, default=10, help="seed for data shuffle")
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    
    oooutput_dir = os.path.join(args.output_dir, f"Ndata={args.Ndata}/bs={args.batch_size}/seed={args.shuffle_seed}")
    # Ensure output directory exists
    os.makedirs(oooutput_dir, exist_ok=True)
    plot_path = os.path.join(oooutput_dir, "eigen_density.png")
    data_path = os.path.join(oooutput_dir, "eigen_density.npz")
    
    torch_tools.set_default_dtype("float32")
    
    # Load model
    model = torch.load(args.model_path, map_location=device).to(device)
    
    # Define loss function
    criterion = WeightedEnergyForcesLoss(energy_weight=10, forces_weight=1000)
    
    # Load dataset
    atomic_energys_dict, all_configs = data.load_from_xyz(
        file_path=args.config_path,
        config_type_weights={"Default": 1.0},
        #energy_key="energy",
        energy_key = "TotEnergy",
        forces_key="force"
    )
    configs = all_configs[:args.Ndata]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    
    dataset = [
        data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max))
        for config in configs
    ]

    random.seed(args.shuffle_seed)
    random.shuffle(dataset)  # Shuffle dataset manually

    # Create DataLoader (without shuffle=True)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle here since we shuffled manually
        drop_last=False,
    )

    # 
    model = model.to(torch.float32)
    #data_loader = data_loader.to(torch.float32)
    
    batch = next(iter(data_loader))  # Use first batch only
    batch.to(device)
    #batch = batch.to()
    batch_dict = batch.to_dict()
    
    print("Testing model convergence...")
    output = model(batch_dict, training=True, compute_force=True)
    print("Predicted Energy:", output['energy'])
    print("Reference Energy:", batch['energy'])
    print("Loss:", criterion(pred=output, ref=batch))
    
    # Compute Hessian eigenvalues and eigenvectors
    hessian_comp = hessian(model, criterion, dataloader=data_loader, cuda=(device == "cuda"))
    trace = hessian_comp.trace()
    print("The trace of this model is: %.4f"%(np.mean(trace)))

    density_eigen, density_weight = hessian_comp.density()

    get_esd_plot(density_eigen, density_weight, path=plot_path)
    print(f"eigen density plot saved at {plot_path}")
    # Save data
    np.savez(data_path, density_eigen=density_eigen, density_weight=density_weight)
    print(f"Loss data saved at {data_path}")
    
if __name__ == "__main__":
    main()