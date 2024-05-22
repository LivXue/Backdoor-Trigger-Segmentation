from .plms import PLMVictim


Victim_List = {
    'plm': PLMVictim,
}


def load_victim(config):
    victim = Victim_List[config["type"]](**config)
    return victim

