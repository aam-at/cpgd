# -*- coding: utf-8 -*-
import ast
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from absl import flags

FLOAT_REGEXP = "[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"

flags.DEFINE_string("wildcard", None, "directories to parse logs")

FLAGS = flags.FLAGS


def load_params(str):
    try:
        return ast.literal_eval(str)
    except:
        return json.loads(str)


def load_training_params(path):
    with open(os.path.join(path, "tensorflow.log"), "r") as f:
        return load_params(f.readline())


def parse_float_value(id, text):
    try:
        match = re.findall("(?<=\s%s:)\s+%s" % (id, FLOAT_REGEXP), text)[0]
        val = float(match[0] + match[2])
        return val
    except:
        return np.nan


def round(l, decimals=6):
    rl = []
    for e in l:
        if isinstance(e, list):
            rl.append(round(e, decimals))
        elif isinstance(e, float):
            rl.append(np.round_(e, decimals=decimals))
        else:
            rl.append(e)
    return rl


def parse_log(load_dir, exclude=None, export_test_params=None):
    if exclude is None:
        exclude = []
    if export_test_params is None:
        export_test_params = []

    with open(os.path.join(load_dir, "tensorflow.log"), "r") as f:
        test_param_str = f.readline()
        test_params = load_params(test_param_str)
        text = "\n".join(f.readlines())
    name = os.path.basename(load_dir)

    try:
        test_results_str = re.findall("(?<=Test results).*", text)[-1]
        test_results_str = test_results_str[test_results_str.find(":") + 1:]
        test_values_with_name = test_results_str.split(",")
        test_result = {"name": name}

        for test_value_with_name in test_values_with_name:
            test_name, test_value = test_value_with_name.split(":")
            test_name = test_name.strip()
            test_value = float(test_value)
            if test_name not in exclude:
                test_result[test_name] = float(test_value)

        for param_name in export_test_params:
            test_result[param_name] = test_params[param_name]
        df = pd.DataFrame.from_dict([test_result])
    except:
        print(f"Failed to parse directory: {load_dir}")
        df = None
    return df


def to_ascii(v):
    nv = []
    for e in v:
        if sys.version_info >= (3, 0, 0):
            if isinstance(e, bytes):
                e = e.encode("ascii", "ignore")
            elif isinstance(e, list):
                e = to_ascii(e)
        else:
            if isinstance(e, unicode):
                e = e.encode("ascii", "ignore")
            elif isinstance(e, list):
                e = to_ascii(e)
        nv.append(e)
    return nv


def org_table(df, compute_stat=False, decimals=6):
    org_table = [None] + [list(df)] + [None] + to_ascii(
        df.values.tolist()) + [None]
    if compute_stat:
        df_stat = df.agg([np.mean, np.std])
        df_stat_list = df_stat.values.tolist()
        df_stat_list[0].insert(0, "mean")
        df_stat_list[1].insert(0, "std")
        org_table += df_stat_list + [None]
    return round(org_table, decimals)


def org_group_and_summarize(
    df,
    group_by=None,
    sort_groups=False,
    sort_keys=("err", "l2_df"),
    top_k=None,
    decimals=6,
):
    if group_by is None:

        def get_unique_name(name):
            try:
                unique_name = re.findall(".+(?=_.+)", name)[0]
            except:
                unique_name = re.findall(".+(?=\d+)", name)[0]
            return unique_name

    df["group_name"] = [group_by(n) for n in df["name"]]
    groups = df.groupby("group_name", axis=0)
    if isinstance(group_by, (list, tuple)):
        df.drop(group_by, axis=1, inplace=True)

    if sort_groups:

        def sort_key(item):
            name, group = item
            group_stat = group.agg([np.mean])
            if isinstance(sort_keys, (list, tuple)):
                assert len(sort_keys) == 2
                return (group_stat[sort_keys[0]] / group_stat[sort_keys[1]])[0]
            else:
                return group_stat[sort_keys][0]

        groups = sorted(groups, key=sort_key, reverse=False)

    if top_k is not None:
        groups = groups[:top_k]

    org_table = [None] + [list(df)]
    org_table[1].insert(1, "statistic")
    # append group size
    org_table[1].append("group_size")
    for name, group in groups:
        org_table.append(None)
        group_stat = group.agg([np.mean, np.std, np.min, np.max])
        group_stat.drop(["name"], axis=1, inplace=True)
        group_stat.drop(["group_name"], axis=1, inplace=True)
        group_stat_list = group_stat.values.tolist()
        group_stat_list[0].insert(0, "mean")
        group_stat_list[1].insert(0, "std")
        group_stat_list[2].insert(0, "min")
        group_stat_list[3].insert(0, "max")
        group_stat_list[0].insert(0, name)
        group_stat_list[1].insert(0, "")
        group_stat_list[2].insert(0, "")
        group_stat_list[3].insert(0, "")
        # append group size
        group_stat_list[0].append(len(group))
        org_table.append(group_stat_list[0])
        org_table.append(group_stat_list[1])
        org_table.append(group_stat_list[2])
        org_table.append(group_stat_list[3])
    org_table.append(None)
    return round(org_table, decimals)


if __name__ == "__main__":
    pass
