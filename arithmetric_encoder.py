from typing import List, Dict, NamedTuple
from collections import OrderedDict

class ACModel():
    def __init__(self, min_base_exp: int, base: int, model: Dict[str, float], text: str):
        base_exp = max(min_base_exp, len(text))
        self.power = base ** base_exp
        
        min_power = base ** min_base_exp
        self.model = OrderedDict()
        for k in model:
            self.model[k] = int(model[k] * min_power) * self.power // min_power

class EncodedAC(NamedTuple):
    code: List[int]
    text_len: int
    ac_model: ACModel

def interval_to_code(interval: List[int], power: int)-> List[int]:
    def are_overlap(interv1: List[int], interv2: List[int])-> bool:
        return interv1[0] < interv2[1] and interv2[0] < interv1[1]
        
    # 表すコードの文字が2種類なので、2で割っていく
    code_float = 0
    code = []
    while True:
        if code_float < interval[0] or code_float >= interval[1]:
            lower = are_overlap([code_float, code_float + power // 2], interval)
            upper = are_overlap([code_float + power // 2, code_float + power * 2], interval)
            
            if upper:
                code_float += int(power // 2)
                code.append(1)
                if lower:
                    break
            else:
                code.append(0)
        else:
            break
        
        power = int(power // 2)

    return code
    
def encode(ac_model: ACModel, text: str)-> EncodedAC:
    interval = [0, 1 * ac_model.power]
    
    for c in text:
        lower = interval[0]
        found = False
        for k, v in ac_model.model.items():
            upper = lower + (interval[1] - interval[0]) * v // ac_model.power
            
            if c == k:
                interval[0] = lower
                interval[1] = upper
                found = True
                break
                
            lower = upper
            
        if not found:
            raise ValueError()
    
    code = interval_to_code(interval, ac_model.power)

    return EncodedAC(code, len(text), ac_model)

# decode
def decode(encoded_ac: EncodedAC):
    ac_model = encoded_ac.ac_model
    output = []
        
    code_float = 0
    power_ = ac_model.power
    
    for c in encoded_ac.code:
        code_float += c * power_ // 2
        power_ //= 2
    
    interval = [0, 1 * ac_model.power]
    
    # print(code_float)
    # print(code_float / ac_model.power)
        
    for _ in range(encoded_ac.text_len):
        lower = interval[0]
        # print([interval[0], interval[1]])
        found = False
        for k, v in ac_model.model.items():
            upper = lower + (interval[1] - interval[0]) * v // ac_model.power

            if code_float < upper:
                interval[0] = lower
                interval[1] = upper
                output.append(k)
                found = True
                break
            
            lower = upper
        
        if not found:
            raise ValueError()
        
        # print([interval[0], interval[1]])
    return output
