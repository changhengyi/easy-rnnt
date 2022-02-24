from omegaconf import OmegaConf


def get_conf(cli_args):
    config_path = cli_args.config
    conf_args = OmegaConf.load(config_path)

    default_args = get_default_config()

    conf = OmegaConf.merge(default_args, conf_args, cli_args)

    return conf
    

def get_default_config():
    args_dict = {}

    args_dict['dist_backend'] = 'nccl'

    args = OmegaConf.create(args_dict)
    return args