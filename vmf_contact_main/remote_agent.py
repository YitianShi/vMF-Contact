import argparse
import logging
import os
from typing import Optional, cast

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from wandb.wandb_run import Run
import sys, os, copy, time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '.'))

import torch
print(torch.__version__)
print("Cuda available: ", torch.cuda.is_available())
print("Cuda device number: ", torch.cuda.device_count())

data_path = os.environ.get("LSDFPROJECTS", "..")
if not os.path.exists(data_path):
    data_path = ".."
print(f"Current data path: {data_path}")

from vmf_contact import vmfContactModule
from vmf_contact import DATASET_REGISTRY
from openpoints.utils import EasyConfig
import glob
import warnings
import socket
import pickle
import numpy as np
import struct
import open3d as o3d

def suppress_pytorch_lightning_logs():
    """
    Suppresses annoying PyTorch Lightning logs.
    """
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers`.*")
    warnings.filterwarnings("ignore", ".*this may lead to large memory footprint.*")
    warnings.filterwarnings("ignore", ".*DataModule.setup has already been called.*")
    warnings.filterwarnings("ignore", ".*DataModule.teardown has already been called.*")
    warnings.filterwarnings("ignore", ".*Set the gpus flag in your trainer.*")
    warnings.filterwarnings("ignore", ".*It is recommended to use.*")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def get_args_parser(
    description: Optional[str] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        add_help=add_help,
    )
    # Training
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Distributed training device number",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--learning_rate_flow",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--learning_rate_decay",
        type=float,
        default=5e-4,
        help="Learning rate decay",
    )
    parser.add_argument(
        "--run-finetuning",
        action="store_true",
        help="Whether to run finetuning",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=40000,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--flow_finetune",
        type=int,
        default=0,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--camera-num",
        type=int,
        help="Number of cameras",
        default=2,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Whether to evaluate the model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint to validate, if None then training",
    )
    # Data module
    parser.add_argument(
        "--data-root-dir",
        type=str,
        help="Root directory of the data",
        default="data_all",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Embedding dimension vmfContact",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(480, 640),
        help="Image size",
    )
    parser.add_argument(
        "--pcd_with_rgb",
        action="store_true",
        help="Whether to use RGB with PCD",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7 / 8,
        help="Image size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mgn",
        help="Dataset name",
    )
    ## Uncertainty
    parser.add_argument(
        "--prob_baseline",
        type=str,
        default=None,
        choices=["post", "lh", None],
        help="Baseline vector modeled as a constant or a learnable parameter",
    )
    parser.add_argument(
        "--certainty_budget",
        type=str,
        default="constant",
        help="Certainty budget",
    )
    parser.add_argument(
        "--entropy_weight",
        type=float,
        default=1e-6,
        help="The weight for the entropy regularizer.",
    )

    parser.add_argument(
        "--flow_layers",
        type=int,
        default=4,
        help="Number of flow layers",
    )

    parser.add_argument(
        "--point_backbone",
        type=str,
        default=None,
        choices=["pointnet++", "pointnext-s", "pointnext-b", "spotr", "dgcnn"],
    )
    # Flow
    parser.add_argument(
        "--hidden_feat_flow",
        type=int,
        default=512,
        help="Hidden feature size for the flow",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=256,
        help="Embedding dimension vmfContact",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="clip",
        choices=["clip", "vits", "vitb", "vitl", "vitg", "resnet"],
    )

    parser.add_argument(
        "--flow_type",
        type=str,
        default="resflow",
        choices=["resflow", "glow"],
    )

    parser.set_defaults(
        epochs=10,
        num_workers=10,
        epoch_length=1250,
        learning_rates=[
            1e-5,
            2e-5,
            5e-5,
            1e-4,
            2e-4,
            5e-4,
            1e-3,
            2e-3,
            5e-3,
            1e-2,
            2e-2,
            5e-2,
            0.1,
        ],
        #data_root_dir=f"{data_path}/data_all/data_debug",
        data_root_dir=glob.glob(f"{data_path}/data_all/data*"),
        data_root_dir_test=[f"{data_path}/data_all/data4"],
        data_root_dir_debug=[f"{data_path}/data_all/data_debug"],
    )
    return parser


import yaml
print(f"{data_path}/data_all/data*")
def parse_args_from_yaml(config_path: str):
    # Load default configurations from YAML
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Create the argument parser
    parser = get_args_parser()

    # Set defaults from YAML file
    parser.set_defaults(**yaml_config)

    # Parse command-line arguments (they will override YAML defaults)
    args = parser.parse_args()

    print(args)

    return args
    

def main_module(args: argparse.Namespace):
    logging.getLogger("vmf_contact").setLevel(logging.INFO)
    suppress_pytorch_lightning_logs()

    # Fix randomness
    pl.seed_everything(args.seed)
    logger.info("Using seed %s.", os.getenv("PL_GLOBAL_SEED"))
    point_backbone_cfgs = EasyConfig()
    current_file_path = os.path.abspath(__file__)
    directory_path = os.path.dirname(current_file_path)
    print(f"{directory_path}/../vmf_contact_main/cfgs/vmfcontact/*.yaml")
    cfgs = glob.glob(f"{directory_path}/../vmf_contact_main/cfgs/vmfcontact/*.yaml")
    for cfg in cfgs:
        if args.point_backbone in cfg:
            print(f"Loading {cfg}")
            setattr(args, "point_backbone_cfgs", cfg)
            break
    else:
        raise ValueError(f"Point backbone {args.point_backbone} not found.")
    point_backbone_cfgs.load(args.point_backbone_cfgs, recursive=True)
    args.point_backbone_cfgs = point_backbone_cfgs

    # Initialize logger if needed
    if args.experiment is not None:
        remote_logger = WandbLogger()
        cast(Run, remote_logger.experiment).config.update(
            {
                "seed": os.getenv("PL_GLOBAL_SEED"),
                "dataset": args.dataset,
                "flow_type": "residual",
                "flow_layers": args.flow_layers,
                "certainty_budget": args.certainty_budget,
                "learning_rate": args.learning_rate,
                "learning_rate_decay": args.learning_rate_decay,
                "max_epochs": args.max_epochs,
                "entropy_weight": args.entropy_weight,
                "flow_finetune": args.flow_finetune,
                "run_finetuning": args.run_finetuning,
            }
        )
    else:
        remote_logger = None

    dm = DATASET_REGISTRY[args.dataset](
        args, seed=int(os.getenv("PL_GLOBAL_SEED") or 0)
    )

    estimator = vmfContactModule(
        args,
        finetune=args.run_finetuning,
        user_params=dict(
            max_epochs=args.max_epochs,
            logger=remote_logger,
            accelerator="gpu",
            default_root_dir="logs",
            devices= 1 if args.debug or args.eval else args.devices,
            strategy="ddp_find_unused_parameters_true"
        ),
    )
    main_module, ckpt_loaded = estimator.module_loader(args.ckpt)
    if ckpt_loaded:
        import torch
        logger.info("Loaded checkpoint")
        return main_module
        pcd = torch.load(f"{args.data_root_dir_test[0]}/env_0_epi_1_step_0_data.pt", map_location="cpu")["camera_3"]["pcd"]
        prediction = main_module.inference(pcd)
        print(prediction)
    else:
        logger.info("No checkpoint loaded")
        return None

def main():
    current_file_folder = os.path.dirname(os.path.abspath(__file__))

    agent = main_module(parse_args_from_yaml(current_file_folder + "/config.yaml"))
    # Agent setup (client)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    pcd_resize=np.array(1)
    chunk_size = 4096
    range = 0.5
    
    while True:
        print("Connecting to the server ...")
        while True:
            try:
                client.connect(('localhost', 8081))
                break
            except:
                print("Server is not available, retrying ...")
                time.sleep(2)
                continue
        print("Connected to the server successfully")
        while True:
            # Receive the point cloud
            data = b""
            data_length_bin = None
            while not data_length_bin:
                data_length_bin = client.recv(4)
            data_length = struct.unpack('>I', data_length_bin)[0]
            while len(data) < data_length:
                packet = client.recv(chunk_size)
                if not packet:
                    break
                data += packet
            pcds_raw = pickle.loads(data)
            num_poses = len(pcds_raw)
            pcds_raw = pcds_raw.astype(np.float32) / 1000
            pcd_mean = np.mean(pcds_raw[..., :2], axis=1, keepdims=True)
            
            # Process the point cloud
            pcds_raw[..., :2] = (pcds_raw[..., :2] - pcd_mean) / pcd_resize
            pcds = copy.deepcopy(pcds_raw)
            try:
                pcds_processed = []
                for pcd in pcds:
                    pcd = pcd[(pcd[:, 0] > -range / pcd_resize) & (pcd[:, 0] < range / pcd_resize)]
                    pcd = pcd[(pcd[:, 1] > -range / pcd_resize) & (pcd[:, 1] < range / pcd_resize)]
                    pcd = pcd[(pcd[:, 2] > -0.02) & (pcd[:, 2] < 0.45)]
                    assert pcd.shape[0] > 0
                    pcds_processed.append(pcd)
            except:
                print("No points left, report error to the server") 
                pcds_processed = None
            # print("Processed point cloud: ", pcd.shape)

            # Viszualize the point cloud
            # o3d_pcd = o3d.geometry.PointCloud(
            #     o3d.utility.Vector3dVector(pcd)
            # )
            # # draw  the origin as a red sphere
            # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            #     size=0.1, origin=[0, 0, 0]
            # )
            # o3d.visualization.draw_geometries([o3d_pcd, mesh_frame])

            # Process the prompt point cloud
            pcd_from_prompt = None

            # draw the point cloud and prompt point cloud
            # o3d_pcd = o3d.geometry.PointCloud(
            #     o3d.utility.Vector3dVector(pcd)
            # )
            # o3d_pcd_from_prompt = o3d.geometry.PointCloud(
            #     o3d.utility.Vector3dVector(pcd_from_prompt)
            # )
            # # draw  the origin as a red sphere
            # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            #     size=0.1, origin=[0, 0, 0]
            # )
            # o3d.visualization.draw_geometries([o3d_pcd_from_prompt, mesh_frame])


            poses_chosen = np.zeros((num_poses, 4, 4))  # Pre-allocate array for efficiency
            # inference
            if pcds_processed is not None:
                # Assuming each pose_chosen is a 4x4 matrix
                for i, pcd_processed in enumerate(pcds_processed):
                    pose_chosen = agent.inference(pcd_processed, 
                                                  pcd_from_prompt=pcd_from_prompt,
                                                  grasp_height_th=-0.02, 
                                                  convention = "zyx")
                    if pose_chosen is None:
                        continue
                    pose_chosen[:3, 3] = pose_chosen[:3, 3] * pcd_resize
                    pose_chosen[:2, 3] += pcd_mean[i][0]
                    poses_chosen[i] = pose_chosen  # Assign to the pre-allocated array
            poses_chosen = torch.tensor(poses_chosen, dtype=torch.float32)
                    
            # Visualize the poses
            # frames = []

            # for i in range(poses.shape[0]):
            #     # Extract the transformation matrix for the ith pose
            #     pose_matrix = poses[i]

            #     # Create a coordinate frame at this pose
            #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)  # Size can be adjusted
            #     frame.transform(pose_matrix)  # Apply the transformation matrix to the frame

            #     # Add the frame to the list of frames
            #     frames.append(frame)
            #     break

            # pcds_raw = o3d.geometry.PointCloud(
            #     o3d.utility.Vector3dVector(pcds_raw)
            # )stop_eventation matrix: ", pose_chosen[:3, :3])

            # Convert the rotation matrix to a quaternion
            
            # Send pose to the server
            data = pickle.dumps(poses_chosen)
            data_length = len(data)
            client.sendall(struct.pack('>I', data_length))
            client.sendall(data)   
             
if __name__ == "__main__":
    main()
        
        
        
        
 