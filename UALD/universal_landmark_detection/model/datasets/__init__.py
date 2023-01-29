
from .chest import Chest

def get_dataset(s):
    return {
            'chest':Chest
           }[s.lower()]


