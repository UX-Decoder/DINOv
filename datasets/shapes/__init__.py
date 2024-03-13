from .sampler import ShapeSampler
from .simpleclick_sampler import SimpleClickSampler


def build_shape_sampler(cfg, **kwargs):
    sampler_name = cfg['STROKE_SAMPLER']['EVAL']['MODE']
    if sampler_name == 'random':
        return ShapeSampler(cfg, **kwargs)
    elif sampler_name == 'best':
        return SimpleClickSampler(cfg, **kwargs)
    else:
        assert False, "not implemented"