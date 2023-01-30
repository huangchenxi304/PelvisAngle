from .gupen import gupen


def get_dataset(s):
    return {
        'gupen': gupen
    }[s.lower()]
