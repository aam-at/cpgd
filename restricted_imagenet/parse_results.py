from functools import partial

import numpy as np
from lib.parse_results import output_excel, parse_results

base_dir = "../results_imagenet"
parse_results = partial(parse_results, base_dir=base_dir)
output_excel = partial(output_excel, base_dir=base_dir)

# parse df-li
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "li", "df", sort_column="li_corr")
    output_excel(df, model_type, "li", "df")

# parse bethge-li
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "li", "bethge", sort_column="avg_acc_adv")
    output_excel(df, model_type, "li", "bethge")

# parse daa-li
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "li", "daa", sort_column="acc_adv", group_by="attack_eps")
    output_excel(df, model_type, "li", "daa", group_by="attack_eps")

# parse pgd-li
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "li", "pgd", sort_column="acc_adv", group_by="attack_eps")
    output_excel(df, model_type, "li", "pgd", group_by="attack_eps")

# parse fab-li
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "li", "fab")
    output_excel(df, model_type, "li", "fab")

# parse our-li
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "li", "our_li", group_by="attack_loop_number_restarts")
    output_excel(df, model_type, "li", "our_li", group_by="attack_loop_number_restarts")

# parse df-l2
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "df")
    output_excel(df, model_type, "l2", "df")

# parse cw-l2
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "cw")
    output_excel(df, model_type, "l2", "cw")

# parse ddn-l2
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "ddn")
    output_excel(df, model_type, "l2", "ddn")

# parse bethge-l2
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "bethge", sort_column="avg_acc_adv")
    output_excel(df, model_type, "l2", "bethge")

# parse pgd-l2
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "pgd", sort_column="acc_adv", group_by="attack_eps")
    output_excel(df, model_type, "l2", "pgd", group_by="attack_eps")

# parse fab-l2
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "fab")
    output_excel(df, model_type, "l2", "fab")

# parse our-l2
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "our_l2", group_by="attack_loop_number_restarts")
    output_excel(df, model_type, "l2", "our_l2", group_by="attack_loop_number_restarts")

# parse sparsefool-l1
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "sparsefool")
    output_excel(df, model_type, "l1", "sparsefool")

# parse ead-l1
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "ead")
    output_excel(df, model_type, "l1", "ead")

# parse bethge-l1
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "bethge", sort_column="avg_acc_adv")
    output_excel(df, model_type, "l1", "bethge")

# parse pgd-l1
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "pgd", sort_column="acc_adv", group_by="attack_eps")
    output_excel(df, model_type, "l1", "pgd", group_by="attack_eps")

# parse fab-l1
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "fab")
    output_excel(df, model_type, "l1", "fab")

# parse our-l1
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "our_l1", group_by="attack_loop_number_restarts")
    output_excel(df, model_type, "l1", "our_l1", group_by="attack_loop_number_restarts")

# parse sparsefool-l0
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "sparsefool", sort_column="l0_corr")
    output_excel(df, model_type, "l0", "sparsefool")

# parse jsma-l0
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l0", "jsma")
    output_excel(df, model_type, "l0", "jsma")

# parse bethge-l0
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l0", "bethge", sort_column="avg_acc_adv")
    output_excel(df, model_type, "l0", "bethge")

# parse pgd-l0
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l0", "pgd", sort_column="acc_adv", group_by="attack_eps")
    output_excel(df, model_type, "l0", "pgd", group_by="attack_eps")

# parse our-l0
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l0", "our_l0", group_by="attack_loop_number_restarts")
    output_excel(df, model_type, "l0", "our_l0", group_by="attack_loop_number_restarts")
