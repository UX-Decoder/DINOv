"""
placeholder for language open-set or grounding
"""


def build_language_encoder(config, **kwargs):
    model_name = config['MODEL']['TEXT']['ARCH']
    if model_name=='noencoder':
        return None
    else:
        raise NotImplementedError