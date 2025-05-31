import importlib

def import_class_from_module(module_name, class_name):
    module = importlib.import_module(module_name)
    class_obj = getattr(module, class_name)
    return class_obj


def parse_dataset_config(config):
    """
    Parse the configuration dictionary to extract relevant parameters.
    """
    return {
        'root': config['root'],
        'clip_length': config.get('clip_length', 30),
        'clip_overlap': config.get('clip_overlap', 0),
        'input_size': (config['h'], config['w']),
        'target_size': (config['h'], config['w']),
    }


def parse_metric_config(config):
    """
    Parse the configuration dictionary to extract metric names.
    """
    metric_names = []
    if 'eval_depth' in config:
        metric_names.extend(config['eval_depth']['metric_names'])
    if 'eval_pcd' in config:
        metric_names.extend(config['eval_pcd']['metric_names'])
    if 'eval_camera' in config:
        metric_names.extend(config['eval_camera']['metric_names'])
    return metric_names