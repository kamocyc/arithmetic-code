from hmm import *
from collections import OrderedDict
from arithmetric_encoder import *
from typing import Dict, Any, Union
import math

def get_bit(p: float)-> float:
    return -math.log2(p)

def get_entropy(model: Dict[str, float])-> float:
    return sum([-v * math.log2(v) for _, v in model.items()])

def get_uniform_entropy(num: int)-> float:
    return get_entropy(dict([(str(i), 1/num) for i in range(num)]))
    
    
def generate_hmm_a(num: int)-> List[str]:
    states = ("Rainy", "Sunny")
    observations = ("w", "s", "c")

    start_prob = {"Rainy": 0.6, "Sunny": 0.4}

    transition_prob = {
        "Rainy": {"Rainy": 0.7, "Sunny": 0.3},
        "Sunny": {"Rainy": 0.4, "Sunny": 0.6},
    }

    emission_prob = {
        "Rainy": {"w": 0.1, "s": 0.4, "c": 0.5},
        "Sunny": {"w": 0.6, "s": 0.3, "c": 0.1},
    }
    
    emitted = []
    states_count = {"Rainy": 0, "Sunny": 0}
    current_state = choose_randomly(start_prob)
    states_count[current_state] += 1

    for i in range(num):
        emitted.append(choose_randomly(emission_prob[current_state]))
        current_state = choose_randomly(transition_prob[current_state])
        states_count[current_state] += 1
    
    # print(states_count)
    # print(states_count["Rainy"] / (states_count["Rainy"] + states_count["Sunny"]))
    # print(states_count["Sunny"] / (states_count["Rainy"] + states_count["Sunny"]))
    
    return emitted

def generate_hmm_b(num: int)-> List[str]:
    states = ("Rainy", "Sunny")
    observations = ("w", "s", "c")

    start_prob = {"Rainy": 0.1, "Sunny": 0.9}

    transition_prob = {
        "Rainy": {"Rainy": 0.9, "Sunny": 0.1},
        "Sunny": {"Rainy": 0.8, "Sunny": 0.2},
    }

    emission_prob = {
        "Rainy": {"w": 0.1, "s": 0.2, "c": 0.7},
        "Sunny": {"w": 0.7, "s": 0.1, "c": 0.2},
    }
    
    emitted = []
    states_count = {"Rainy": 0, "Sunny": 0}
    current_state = choose_randomly(start_prob)
    states_count[current_state] += 1

    for i in range(num):
        emitted.append(choose_randomly(emission_prob[current_state]))
        current_state = choose_randomly(transition_prob[current_state])
        states_count[current_state] += 1
    
    # print(states_count)
    # print(states_count["Rainy"] / (states_count["Rainy"] + states_count["Sunny"])))
    # print(states_count["Sunny"] / (states_count["Rainy"] + states_count["Sunny"]))
    
    return emitted

def kl_divergence(pd1: Dict[Any, float], pd2: Dict[Any, float])-> float:
    assert(len(pd1) == len(pd2))
    
    total = 0.0
    for k in pd1:
        total += pd1[k] * math.log2(pd1[k] / pd2[k])
    
    return total

def count_chars(text: str, chars: List[str])-> Dict[str, int]:
    counts = OrderedDict()
    for c in chars:
        counts[c] = 0
        
    for c in text:
        counts[c] += 1
    
    return counts

def to_ratio(dic: Dict[str, Union[int, int]])-> Dict[str, float]:
    total = sum([v for _, v in dic.items()])
    
    for k, v in dic.items():
        dic[k] = v / total # type: ignore
    
    return dic # type: ignore
    
def do_test(model: Dict[Any, float], text: str):
    case_num = 3
    min_base_exp = 10
    base = 8
    
    # 圧縮して、展開してみる
    ac_model = ACModel(min_base_exp, base, model, text)
    encoded_ac = encode(ac_model, text)
    compressed = len(encoded_ac.code)
    original = len(text) * get_bit(1 / case_num)
    
    decoded = "".join(decode(encoded_ac))
    
    # print(decoded)
    
    data = {
        "compressed": compressed,
        "original": original,
        "comp_ratio": compressed / original,
        "is_preserved": decoded == text
    }
    
    # print(data)
    
    return data

def calc_limits(model: Dict[Any, float]):
    case_num = len(model)
    
    return {
        "model_entropy": get_entropy(model),
        "org_entropy": get_uniform_entropy(case_num),
        "lower_limit": get_entropy(model) / get_uniform_entropy(case_num)
    }
    
def main():
    length = 1000
    
    prob_a = get_stationary_distribution(
        [[0.7, 0.3], [0.4, 0.6]],
        [[0.1, 0.4, 0.5],
        [0.6, 0.3, 0.1]]
    )

    prob_b = get_stationary_distribution(
        [[0.9, 0.1], [0.8, 0.2]],
        [[0.1, 0.2, 0.7],
        [0.7, 0.1, 0.2]]
    )
    
    theory_model_a = OrderedDict(zip(["w", "s", "c"], prob_a))
    theory_model_b = OrderedDict(zip(["w", "s", "c"], prob_b))
    
    print(theory_model_a)
    print(theory_model_b)
    
    model_a: OrderedDict[str, float] = OrderedDict([("w", 0.314453125), ("s", 0.357421875), ("c", 0.328125)])
    model_b: OrderedDict[str, float] = OrderedDict([("w", 0.166015625), ("s", 0.189453125), ("c", 0.64453125)])
    
    string_a = ''.join(generate_hmm_a(length))
    string_b = ''.join(generate_hmm_b(length))
    
    generated_model_a = to_ratio(count_chars(string_a, ["w", "s", "c"]))
    generated_model_b = to_ratio(count_chars(string_b, ["w", "s", "c"]))

    # compressed: 圧縮後のバイト数
    # original: 元のバイト数
    # comp_ratio: 圧縮率
    # is_preserved: 圧縮・展開後にデータが同じならTrue
    # kl: KLダイバージェンス。
    # kl2: KLダイバージェンス（逆向き）
    
    result_aa = do_test(model_a, string_a)
    result_bb = do_test(model_b, string_b)
    result_ab = do_test(model_a, string_b)
    result_ba = do_test(model_b, string_a)
    
    result_aa["kl"] = kl_divergence(model_a, generated_model_a)
    result_bb["kl"] = kl_divergence(model_b, generated_model_b)
    result_ab["kl"] = kl_divergence(model_a, generated_model_b)
    result_ba["kl"] = kl_divergence(model_b, generated_model_a)
    
    result_aa["kl2"] = kl_divergence(generated_model_a, model_a)
    result_bb["kl2"] = kl_divergence(generated_model_b, model_b)
    result_ab["kl2"] = kl_divergence(generated_model_b, model_a)
    result_ba["kl2"] = kl_divergence(generated_model_a, model_b)
    
    limit_a = calc_limits(model_a)
    limit_b = calc_limits(model_b)
    
    result_aa["limit"] = limit_a
    result_ba["limit"] = limit_a
    result_bb["limit"] = limit_b
    result_ab["limit"] = limit_b
    
    result_aa["ratio_p_kl"] = result_aa["limit"]["lower_limit"] + result_aa["kl"]
    result_ba["ratio_p_kl"] = result_ba["limit"]["lower_limit"] + result_ba["kl"]
    result_bb["ratio_p_kl"] = result_bb["limit"]["lower_limit"] + result_bb["kl"]
    result_ab["ratio_p_kl"] = result_ab["limit"]["lower_limit"] + result_ab["kl"]
    
    # 0.15くらいは違ってくる...
    # n回やって差分とか
    print(result_aa["ratio_p_kl"])
    print(result_aa["comp_ratio"])
    print(result_bb["ratio_p_kl"])
    print(result_bb["comp_ratio"])
    print(result_ab["ratio_p_kl"])
    print(result_ab["comp_ratio"])
    print(result_ba["ratio_p_kl"])
    print(result_ba["comp_ratio"])
    
    
    print(result_aa)
    print(result_bb)
    print(result_ab)
    print(result_ba)
    
    
    # 実際はp2なのにp1とみなしたロス
    
    
    # if True:   
    #     ws = generate_hmm_a(1000)
    #     # print(ws)
    #     cw = sum([1 for w in ws if w == 'w'])
    #     cs = sum([1 for w in ws if w == 's'])
    #     cc = sum([1 for w in ws if w == 'c'])
    #     print([cw / 1000, cs / 1000, cc / 1000])
    #     print(prob_a)


    # if True:   
    #     ws = generate_hmm_b(1000)
    #     # print(ws)
    #     cw = sum([1 for w in ws if w == 'w'])
    #     cs = sum([1 for w in ws if w == 's'])
    #     cc = sum([1 for w in ws if w == 'c'])
    #     print([cw / 1000, cs / 1000, cc / 1000])
    #     print(prob_b)

main()
