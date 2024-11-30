import re

def get_relevant_filters(f_name):
    with open(f_name, "r") as f:
        rules = f.readlines()
    rel_filters = []
    for rule in rules:
        x = re.findall(' [0-9]+', rule)
        rel_filters = rel_filters + x
    rel_filters = [int(fil.strip()) for fil in rel_filters]
    rel_filters = list(set(rel_filters))
    return rel_filters
# def replace_filter(f_name):
#     with open(f_name, "r") as f:
#         rules = f.read
#     rules
# l = get_relevant_filters("rules2.txt")
# len(l)