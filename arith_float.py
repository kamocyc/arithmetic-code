from typing import List, Dict

# 浮動小数点を使うバージョン
# 文字列長が長くなると、誤差で動かない
# 現在使っていない

model = {"a": 0.625, "b": 0.4}
text = "ababaaa"

interval = [0.0, 1.0]

for c in text:
    # a側 -> 下
    if c == 'a':
        interval[1] = (interval[1] - interval[0]) * 0.625 + interval[0]
    else:
        interval[0] = (interval[1] - interval[0]) * 0.625 + interval[0]
    
    print(interval)

def are_overlap(interv1: List[float], interv2: List[float])-> bool:
    return interv1[0] <= interv2[1] and interv2[0] <= interv1[1]
        
code_float = 0.0
power = 0.5
code = []

while True:
    if code_float < interval[0] or code_float > interval[1]:
        lower = are_overlap([code_float, code_float + power], interval)
        upper = are_overlap([code_float + power, code_float + power * 2], interval)
        
        if lower and upper:
            code_float += power
            code.append(1)
            break
        
        if lower:
            code.append(0)
        else:
            code.append(1)
            code_float += power
    else:
        break
        
    power *= 0.5
    # print(code_float)
    # print(code)
    # print(power)

print(code)
print(code_float)

# decode
def decode(model: Dict[str, float], code: List[int], code_len: int):
    interval = [0.0, 1.0]
    output = []
    
    code_float = 0.0
    power = 0.5
    for c in code:
        code_float += c * power
        power *= 0.5
        
    for _ in range(code_len):
        if code_float < (interval[1] - interval[0]) * 0.625 + interval[0]:
            interval[1] = (interval[1] - interval[0]) * 0.625 + interval[0]
            # lower
            output.append('a')
        else:
            interval[0] = (interval[1] - interval[0]) * 0.625 + interval[0]
            output.append('b')
    
        print(interval)
    return output

code_len = len(text)
decoded = "".join(decode(model, code, code_len))
print(decoded)
print(decoded == text)