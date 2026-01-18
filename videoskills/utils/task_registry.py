import os
from typing import Tuple
from datetime import datetime
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.runners.runner_eval import OnPolicyRunnerEval
from videoskills import LEGGED_GYM_ROOT_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params, \
    dict_to_class
from videoskills.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import yaml

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, args) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        if args.load_config:
            # load config from the path
            return self.load_cfg(args)
        train_cfg = self.train_cfgs[args.task]
        env_cfg = self.env_cfgs[args.task]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def load_cfg(self, args) -> LeggedRobotCfg:
        """ Loads a config file from the given path.

        Args:
            cfg_path (str): Path to the config file.

        Returns:
            Dict: The loaded config file.
        """
        cfg_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', f'{args.task}_ppo', args.load_run, 'config.yaml')
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        # train_cfg = cfg
        # convert to LeggedRobotCfg
        env_cfg = cfg.get('env_cfg', {})
        train_cfg = cfg.get('train_cfg', {})
        if args.motion_file is not None:
            motion_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'dataset', f'{args.task}_motion', args.motion_file)
            env_cfg['motion']['file'] = motion_file_path

        train_cfg['runner']['load_run'] = args.load_run

        env_cfg = dict_to_class(env_cfg)
        train_cfg = dict_to_class(train_cfg)

        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)   # Hacky edited it, to ens
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)

        # if sim params are provided in env_cfg, parse them and update/override above:
        # create the environment
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)


        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_dir=None, eval_mode=False) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example).
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        if log_dir is None:
            if eval_mode:
                log_dir = os.path.join(log_root, args.load_run)
            else:
                log_dir = os.path.join(log_root, train_cfg.runner.run_name + '_' + datetime.now().strftime('%b%d_%H-%M-%S'))

        use_amp_runner = train_cfg.runner.use_amp_runner  # ✅ 从 cfg 中读取
        if train_cfg.runner.run_name.split('_')[-1] == 'test':
            train_cfg.runner.save_interval = 500
            train_cfg.runner.eval_interval = 500

        train_cfg_dict = class_to_dict(train_cfg)

        runner = OnPolicyRunnerEval(env, train_cfg_dict, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir

        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.resume_path = resume_path
            runner.load(resume_path)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()