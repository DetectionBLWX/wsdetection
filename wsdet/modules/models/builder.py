'''initialize'''
from .oicr import OICR


'''build model'''
def BuildModel(cfg, mode, **kwargs):
    supported_models = {
        'oicr': OICR,
    }
    model_type = cfg['type']
    assert model_type in supported_models, 'unsupport model_type %s...' % model_type
    return supported_models[model_type](cfg, mode=mode)