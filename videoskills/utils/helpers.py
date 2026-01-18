import os
import copy
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
import torch
import joblib
import yaml
from datetime import datetime
from typing import Any, Dict, Type

from videoskills import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
# LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def class_to_dict(obj) -> dict:
    if not hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def dict_to_class(data: Any) -> Any:
    """
    Recursively convert a dictionary or list into a simple object with attributes.
    - Dicts become instances of a dynamically created class with attributes for each key.
    - Lists have each element converted in the same way.
    - Primitive values are returned unchanged.
    """
    # Handle dictionaries by creating a dynamic class instance
    if isinstance(data, dict):
        # Create a new class for this dict
        cls = type('ConfigObject', (), {})
        obj = cls()
        for key, value in data.items():
            setattr(obj, key, dict_to_class(value))
        return obj

    # Handle lists by converting each element
    if isinstance(data, list):
        return [dict_to_class(item) for item in data]

    # Primitives are returned as-is
    return data

# def dict_to_class(data: Any) -> Any:
#     """
#     Convert a dictionary to a class instance.
#     :param dict: Dictionary to convert.
#     :return: Class instance with attributes set from the dictionary.
#     """
#     if isinstance(data, dict):
#         # Convert each value and create a SimpleNamespace
#         return SimpleNamespace(**{k: dict_to_class(v) for k, v in data.items()})
#     elif isinstance(data, list):
#         # Convert each item in the list
#         return [dict_to_class(item) for item in data]
#     else:
#         # Return primitives unchanged
#         return data


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'pt' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.dev:
            env_cfg.env.test = True
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint
        if args.dev:
            cfg_train.runner.max_iterations = 100
            cfg_train.runner.eval_interval = 10

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--load_config", "action" : "store_true",  "help": "Load config file from the task directory. If not provided, will use the default config file."},
        {"name": "--load_motion_sampling_state", "action": "store_true", "help": "Load motion sampling state from the task directory. "},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        {"name": "--motion_file", "type": str, "help": "motion file to use for training/evaluation. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--use_wandb", "action": "store_true", "default": False, "help": "Enable logging to Weights & Biases"},
        {"name": "--wandb_project", "type": str, "default": "VideoSkills", "help": "Weights & Biases project name"},
        {"name": "--dev", "action": "store_true", "default": False, "help": "development mode, use smaller envs"},
        # {"name": "--load_run", "type": str, "default": False, "help": "logging path of resume experiment"},

        # GVHMR
        {"name": "--folder", "type": str, "default": None, "help": "folder of videos for demo"},
        {"name": "--static_cam", "action": "store_true", "default": False, "help": "If true, skip DPVO"},
        {"name": "--gvhmr_output", "type": str, "default": None, "help": "folder of gvhmr outputs"},

        # Refinepipeline
        {"name": "--accelerate", "action": "store_true", "default": False, "help": "Use batched accelerated refining for easy motions (default: True)."},
        {"name": "--render_run", "type": str, "default": None, "help": "Only render the motion from certain run without refining (default: False)."},

        # Distill/DAgger
        {"name": "--teacher_ckpt", "type": str, "default": None,
         "help": "Path to teacher checkpoint for distillation/DAgger."},
        {"name": "--teacher_config", "type": str, "default": None,
         "help": "Path to teacher config for distillation/DAgger."},
        {"name": "--distill_value", "action": "store_true", "default": False},

        # VAE resume
        {"name": '--vae_ckpt', "type": str, "default":None, "help": 'Path to the distilled VAE checkpoint (dagger_student_*.pt)'}
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"

    if args.dev:
        args.num_envs = 32
        # args.headless = False
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

def parse_motion_file_path(env_cfg, cfg, only_failed_key = False, ext = '.npy', max_files = None):
    if isinstance(env_cfg.motion.file, list):
        motion_file = env_cfg.motion.file
    else:
        motion_file = env_cfg.motion.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    if only_failed_key:  # for AMASS
        failed_key_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', cfg.runner.experiment_name, cfg.runner.load_run
                                      , 'eval_outputs')
        pkl_files = [file for file in os.listdir(failed_key_dir) if 'failed_keys' in file]

        def extract_iter(filename):
            # 假设格式始终为 'failed_keys_iterXXXXX.pkl'
            prefix = "failed_keys_iter"
            suffix = ".pkl"
            if filename.startswith(prefix) and filename.endswith(suffix):
                num_str = filename[len(prefix):-len(suffix)]  # 提取中间部分
                return int(num_str)
            return -1  # fallback

        pkl_files.sort(key=extract_iter)
        failed_keys = joblib.load(os.path.join(failed_key_dir, pkl_files[-1]))
        npy_paths = [os.path.join(motion_file, key.split('-')[-1] + ext) for key in failed_keys]
        # AMASS 结构特殊，直接存相对路径
        # for key in failed_keys:
        #     parts = key.split("-")
        #     if len(parts) >= 2:
        #         dataset = parts[0]
        #         subset = parts[1]
        #         filename = "-".join(parts[2:])
        #         rel_path = os.path.join(motion_file, dataset, subset, filename + ext)
        #         npy_paths.append(rel_path)
        return npy_paths
    else:
        import glob
        if not isinstance(env_cfg.motion.file, list):
            motion_file = glob.glob(os.path.join(motion_file, f"**/*{ext}"), recursive=True)
            if max_files is not None:
                motion_file = motion_file[:max_files]
        return motion_file


def print_and_save_cfg(env_cfg, train_cfg, filename="config.yaml", eval_mode=False):
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    if eval_mode or train_cfg.runner.resume:
        log_dir = os.path.join(log_root, train_cfg.runner.load_run)
    else:
        log_dir = os.path.join(log_root, train_cfg.runner.run_name + '_' + datetime.now().strftime('%b%d_%H-%M-%S'))
    env_cfg_dict = class_to_dict(env_cfg)
    train_cfg_dict = class_to_dict(train_cfg)
    class SmartListDumper(yaml.SafeDumper):
        pass
    SmartListDumper.add_representer(
        list,
        lambda dumper, data:
            dumper.represent_sequence("tag:yaml.org,2002:seq",
                                      data,
                                      flow_style=True)
    )
    cfg = {"env_cfg": env_cfg_dict, "train_cfg": train_cfg_dict}
    # print(yaml.dump(cfg, Dumper=SmartListDumper,
    #                 sort_keys=False, allow_unicode=True, width=120))

    def sanitize(o):
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if hasattr(o, 'cpu') and hasattr(o, 'numpy'):  # torch.Tensor
            return o.cpu().numpy().tolist()
        if isinstance(o, np.generic):  # 新增
            return o.item()
        return o
    def recursive_map(d):
        if isinstance(d, dict):
            return {k: recursive_map(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [recursive_map(v) for v in d]
        else:
            return sanitize(d)

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, filename), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return log_dir


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

