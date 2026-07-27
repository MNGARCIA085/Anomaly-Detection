MODEL_REGISTRY = {}


def register(name):

    def deco(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return deco