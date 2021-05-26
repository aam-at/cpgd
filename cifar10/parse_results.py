from functools import partial

import numpy as np
from lib.parse_results import output_excel, parse_results

base_dir = "../results_cifar10"
parse_results = partial(parse_results, base_dir=base_dir)
output_excel = partial(output_excel, base_dir=base_dir)
for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "li", "df", sort_column="li_corr")
    output_excel(df, model_type, "li", "df")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "li", "bethge", sort_column="li_corr")
    output_excel(df, model_type, "li", "bethge")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "li", "daa", sort_column="acc_adv", group_by="attack_eps")
    output_excel(df, model_type, "li", "daa", group_by="attack_eps")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "li", "pgd", sort_column="acc_adv", group_by="attack_eps")
    output_excel(df, model_type, "li", "pgd", group_by="attack_eps")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "li", "fab")
    output_excel(df, model_type, "li", "fab")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "li", "our_li", group_by="attack_loop_number_restarts")
    output_excel(df, model_type, "li", "our_li", group_by="attack_loop_number_restarts")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "df")
    output_excel(df, model_type, "l2", "df")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "cw")
    output_excel(df, model_type, "l2", "cw")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "ddn")
    output_excel(df, model_type, "l2", "ddn")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "bethge")
    output_excel(df, model_type, "l2", "bethge")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "pgd", sort_column="acc_adv", group_by="attack_eps")
    output_excel(df, model_type, "l2", "pgd", group_by="attack_eps")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "fab")
    output_excel(df, model_type, "l2", "fab")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l2", "our_l2", group_by="attack_loop_number_restarts")
    output_excel(df, model_type, "l2", "our_l2", group_by="attack_loop_number_restarts")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "sparsefool")
    output_excel(df, model_type, "l1", "sparsefool")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "ead")
    output_excel(df, model_type, "l1", "ead")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "bethge")
    output_excel(df, model_type, "l1", "bethge")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "pgd", sort_column="acc_adv", group_by="attack_eps")
    output_excel(df, model_type, "l1", "pgd", group_by="attack_eps")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "fab")
    output_excel(df, model_type, "l1", "fab")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "our_l1", group_by="attack_loop_number_restarts")
    output_excel(df, model_type, "l1", "our_l1", group_by="attack_loop_number_restarts")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l1", "sparsefool", sort_column="l0_corr")
    output_excel(df, model_type, "l0", "sparsefool")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l0", "jsma")
    output_excel(df, model_type, "l0", "jsma")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l0", "one_pixel")
    output_excel(df, model_type, "l0", "one_pixel")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l0", "bethge")
    output_excel(df, model_type, "l0", "bethge")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l0", "cornersearch")
    output_excel(df, model_type, "l0", "cornersearch")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l0", "pgd", sort_column="acc_adv", group_by="attack_eps")
    output_excel(df, model_type, "l0", "pgd", group_by="attack_eps")

for model_type in ["plain", "linf", "l2"]:
    df = parse_results(model_type, "l0", "our_l0", group_by="attack_loop_number_restarts")
    output_excel(df, model_type, "l0", "our_l0", group_by="attack_loop_number_restarts")
