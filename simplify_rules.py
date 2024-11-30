import re

def simplify_rule(rule):
    # mod_rules = []
    # for i in range(len(rules)):
    #     rule = rules[i]
    # writing regex to modify the rule
    #remove the second argument of the ab type predicates
    rule = re.sub(r"ab(\d+)\((X),\s*'\w+'\)", r"ab\1(\2)", rule)
    #next find the predicates of the form not number(X, '1') and replace them with not number(X)
    rule = re.sub(r"not\s+(\d+)\((X),\s*'1'\)", r"not \1(X)", rule)
    #next find the predicates of the form not number(X, '0') and replace them with number(X)
    rule = re.sub(r"not\s+(\d+)\((X),\s*'0'\)", r"\1(X)", rule)
    #next find the predicates of the form number(X, '1') and replace them with number(X)
    rule = re.sub(r"\s+(\d+)\((X),\s*'1'\)", r" \1(X)", rule)
    #next find the predicates of the form number(X, '0') and replace them with not number(X)
    rule = re.sub(r"\s+(\d+)\((X),\s*'0'\)", r" not \1(X)", rule)

    # add the new rule to the modified rules list
    # mod_rules.append(rule)
    return rule



