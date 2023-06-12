import random
from ..utils import split_price, load_pattern

__all__ = [
    'ComplexGeneratorBase',
    'ComplexGeneratorMenu',
    'ComplexGeneratorMain',
    'CfgValue'
]


class CfgValue:   
    def __init__(self, input: dict):
        self.column = input['field']
        self.prob = input.get('prob', 1.0)
        self.prepr_func = split_price if input.get('split_price', False) else None
        self.values = input.get('values')


class ComplexGeneratorBase:
    def __iter__(self):
        raise NotImplementedError


class ComplexGeneratorMenu(ComplexGeneratorBase):
    def __init__(self, pattern):
        self.cfg = [CfgValue(p) for p in pattern]
        
    def __iter__(self):
        for cfg_value in self.cfg:
            if cfg_value.prob > random.random():
                yield cfg_value.column, cfg_value.prepr_func, cfg_value.values
    
    @classmethod
    def load_patterns(cls, pattern_name: str):
        return [cls(pattern) for pattern in load_pattern(pattern_name)]


class ComplexGeneratorMain(ComplexGeneratorBase):
    def __init__(self, pattern_name: str):
        self.cfg: list = load_pattern(pattern_name)

    def __generate_keys(self):
        keys = []

        for pack in self.cfg['packs']:
            if pack['shuffle']:
                random.shuffle(pack['pack'])

        for pack in self.cfg['packs']:
            keys.extend([(val['field'], val['prob']) for val in pack['pack']])

        begin_or_end = self.cfg['begin_or_end']
        begin_or_end_key = (begin_or_end['field'], begin_or_end['field_prob'])
        keys.insert(0 if random.random() < begin_or_end['begin_prob'] else -1, begin_or_end_key)

        return keys

    def __iter__(self):
        for col, chance in self.__generate_keys():
            if chance > random.random():
                yield col
