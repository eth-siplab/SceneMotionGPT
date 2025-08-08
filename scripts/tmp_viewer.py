#%%
import pickle
from pathlib import Path

import torch

from mGPT.data.nymeria.scripts.generate_motion_representation_nymeria import recover_from_ric
from scripts.check_data_format import kinematic_chain, rerun_log_joint_animation

#%%
pckl_file = Path("/home/dominik/Documents/repos/MotionGPT/datasets/Nymeria/tmp/train_data.pkl")

with open(pckl_file, "rb") as f:
    data = pickle.load(f)


#%%
for key, value in data.items():
    motion = value['motion']

    joints = recover_from_ric(torch.from_numpy(motion), 34).numpy()

    rerun_log_joint_animation([joints], kinematic_chain)

    input("Press Enter to continue...")