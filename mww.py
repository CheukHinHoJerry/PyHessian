import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pyhessian import hessian  # Hessian computation
from mace import data
from mace.tools import torch_geometric, torch_tools, utils
from mace.modules import WeightedEnergyForcesLoss
import ase.io
import gc

import random

def print_cudamem(step=""):
    torch.cuda.synchronize()
    print(f"{step}: Allocated: {torch.cuda.memory_allocated()/1e9:.3f} GB, "
          f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.3f} GB, "
          f"Cached: {torch.cuda.memory_reserved()/1e9:.3f} GB")

def get_params_grad(model, loss):
    params = [p for p in model.parameters() if p.requires_grad]
    assert all([p.grad == None for p in params])
    gradsH = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, only_inputs=True)
    return params, gradsH

def dataloader_hv_product(model, dataloader, criterion, v, device):
    model.to(device)
    model.train()
    num_data = 0
    for (idx, inputs) in enumerate(dataloader):
        print_cudamem(f"idx = {idx}")
        model.zero_grad()

        # outputs = model(inputs.to(device), training=True, compute_force=True)
        # loss = criterion(pred=outputs, ref=inputs.to(device))
        # loss.backward(create_graph=True, retain_graph=False)        
        # params, gradsH = get_params_grad(model)
        outputs = model(inputs.to(device), training=True, compute_force=True)
        loss = criterion(pred=outputs, ref=inputs.to(device))
        
        # Compute gradients directly with torch.autograd.grad
        params, gradsH = get_params_grad(model, loss)
        Hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False)
        
        model.zero_grad(set_to_none=True)
        
        for tensor in [gradsH, Hv, params, loss, outputs, inputs]:
            if isinstance(tensor, list):
                for t in tensor:
                    del t
            del tensor
        
        gc.collect()
        torch.cuda.empty_cache()

device = "cuda"
model_path = "/scratch/st-ortner-1/jerry528/ETN/train_benchmark_results/results_MACEwater_firstlayer_new/water-non_symmetric_cp-L0-seed-4-c64/MACEwater_baseline_run-4_lastepoch.model"
batch_size = 20
config_path = "/scratch/st-ortner-1/jerry528/ETN/data/dataset_water/water_test.xyz"

torch_tools.set_default_dtype("float32")

# Load model
model = torch.load(model_path, map_location=device).to(device)

# Define loss function
criterion = WeightedEnergyForcesLoss(energy_weight=10, forces_weight=1000)

# Load dataset
atomic_energys_dict, configs = data.load_from_xyz(
    file_path=config_path,
    config_type_weights={"Default": 1.0},
    #energy_key="energy",
    energy_key = "TotEnergy",
    forces_key="force"
)

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

dataset = [
    data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max))
    for config in configs
]


# Create DataLoader (without shuffle=True)
data_loader = torch_geometric.dataloader.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,  # No need to shuffle here since we shuffled manually
    drop_last=False,
)

# 
model = model.to(torch.float32)
params = [p for p in model.parameters() if p.requires_grad]


v = [torch.randn(p.size()).to(device) for p in params
                ]  # generate random vector

dataloader_hv_product(model, data_loader, criterion, v, device)