'''
Function:
    build sampler
Author:
    Zhenchao Jin
'''
from .groupsampler import GroupSampler


'''build sampler'''
def BuildSampler(sampler_type, **kwargs):
    supported_samplers = {
        'groupsampler': GroupSampler
    }
    assert sampler_type in supported_samplers, 'unsupport sampler_type %s...' % sampler_type
    return supported_samplers[sampler_type](**kwargs)