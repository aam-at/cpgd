# -*- coding: utf-8 -*-
import glob
import os
import re
import sys
from pathlib import Path

import absl
import numpy as np
import pandas as pd
from absl import flags

from utils import load_params

FLOAT_REGEXP = '[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'

flags.DEFINE_string("wildcard", None, "directories to parse logs")

FLAGS = flags.FLAGS


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


def to_ascii(v):
    nv = []
    for e in v:
        if sys.version_info >= (3, 0, 0):
            if isinstance(e, bytes):
                e = e.encode('ascii', 'ignore')
            elif isinstance(e, list):
                e = to_ascii(e)
        else:
            if isinstance(e, unicode):
                e = e.encode('ascii', 'ignore')
            elif isinstance(e, list):
                e = to_ascii(e)
        nv.append(e)
    return nv


def org_table(df, compute_stat=True, decimals=6):
    org_table = ([None] + [list(df)] + [None] + to_ascii(df.values.tolist()) + [None])
    if compute_stat:
        df_stat = df.agg([np.mean, np.std])
        df_stat_list = df_stat.values.tolist()
        df_stat_list[0].insert(0, 'mean')
        df_stat_list[1].insert(0, 'std')
        org_table += df_stat_list + [None]
    return round(org_table, decimals)


def org_group_and_summarize(df, group_by=None, sort_groups=False,
                            sort_keys=('err', 'l2_df'), decimals=6):
    if group_by is None:
        def group_by(id):
            name = df.get_value(id, "name")
            try:
                unique_name = re.findall(".+(?=_.+)", name)[0]
            except:
                unique_name = re.findall(".+(?=\d+)", name)[0]
            return unique_name

    groups = df.groupby(group_by, axis=0)
    if isinstance(group_by, (list, tuple)):
        df.drop(group_by, axis=1, inplace=True)

    if sort_groups:
        def sort_key(item):
            name, group = item
            group_stat = group.agg([np.mean])
            key = group_stat[sort_keys[0]] / group_stat[sort_keys[1]]
            return key[0]
        groups = sorted(groups, key=sort_key, reverse=False)

    org_table = [None] + [list(df)]
    org_table[1].insert(1, 'statistic')
    # append group size
    org_table[1].append('group_size')
    for name, group in groups:
        org_table.append(None)
        group_stat = group.agg([np.min, np.max, np.mean, np.std])
        group_stat.drop(["name"], axis=1, inplace=True)
        group_stat_list = group_stat.values.tolist()
        group_stat_list[0].insert(0, 'min')
        group_stat_list[1].insert(0, 'max')
        group_stat_list[2].insert(0, 'mean')
        group_stat_list[3].insert(0, 'std')
        group_stat_list[0].insert(0, name)
        group_stat_list[1].insert(0, '')
        group_stat_list[2].insert(0, '')
        group_stat_list[3].insert(0, '')
        # append group size
        group_stat_list[0].append(len(group))
        org_table.append(group_stat_list[0])
        org_table.append(group_stat_list[1])
        org_table.append(group_stat_list[2])
        org_table.append(group_stat_list[3])
    org_table.append(None)
    return round(org_table, decimals)


def parse_test_log(wildcard, exclude=None, export_org=False,
                   export_params=None, sort_rows=True):
    fields = [
        'nll', 'acc', 'conf', 'acc_at1', 'acc_at2', 'acc_df', 'l2_df',
        'l2_df_norm', 'conf_df', 'psnr_df', 'ssim_df'
    ]
    if exclude is not None:
        for f in exclude:
            fields.remove(f)
    if export_params is None:
        export_params = []
    if export_params is None:
        export_params = []

    failed = []
    values = []
    for index, load_dir in enumerate(sorted(glob.glob(wildcard))):
        log_path = os.path.join(load_dir, 'tensorflow.log')
        if not os.path.exists(log_path):
            print("Path %s does not contain log" % load_dir)
            continue
        name = os.path.basename(load_dir)

        try:
            with open(log_path, 'r') as f:
                params = load_params(f.readline())
                text = "\n".join(f.readlines())
                test_results = re.findall("(?=Test results).*", text)[0]

            curr_value = [name]
            for field in fields:
                curr_value.append(parse_float_value(field, test_results))
            for param_name in export_params:
                curr_value.append(params[param_name])
            values.append(tuple(curr_value))
        except:
            failed.append(load_dir)

    if len(failed) > 0:
        print("Failed to parse directories: %s" % failed)

    columns = ['name'] + fields + export_params
    df = pd.DataFrame(values, columns=columns)
    if sort_rows:
        df['performance_metric'] = df['acc'] / df['l2_df']
        df.sort_values('performance_metric', ascending=True, inplace=True)
        df.drop('performance_metric', axis=1, inplace=True)
    if export_org:
        return org_table(df)
    else:
        return df


def parse_test_corruptions_log(wildcard, exclude=None, export_org=False,
                               export_params=None, sort_rows=True):
    fields = ['acc_corr']
    sub_fields = ['acc', 'conf']
    if exclude is not None:
        for f in exclude:
            fields.remove(f)
    if export_params is None:
        export_params = []
    if export_params is None:
        export_params = []

    failed = []
    values = []
    first_dir = True
    for index, load_dir in enumerate(sorted(glob.glob(wildcard))):
        log_path = os.path.join(load_dir, 'tensorflow.log')
        if not os.path.exists(log_path):
            print("Path %s does not contain log" % load_dir)
            continue
        name = os.path.basename(load_dir)

        try:
            with open(log_path, 'r') as f:
                params = load_params(f.readline())
                text = "\n".join(f.readlines())
                test_results = re.findall("(?=Test results).*", text)
                final_results = re.findall("(?=Mean accuracy: ).*", text)[0]

            curr_value = [name]
            curr_value.append(float(re.findall(FLOAT_REGEXP, final_results)[0][0]))
            for test_result in test_results:
                name = re.findall("\((.*)\)", test_result)[0]
                for f in sub_fields:
                    curr_value.append(parse_float_value(f, test_result))
                    if first_dir:
                        fields.append(f + "_" + name)
            for param_name in export_params:
                curr_value.append(params[param_name])
            if len(curr_value) - 1 != len(fields):
                print("Failed to parse path %s" % load_dir)
            values.append(tuple(curr_value))
            first_dir = False
        except:
            failed.append(load_dir)

    if len(failed) > 0:
        print("Failed to parse directories: %s" % failed)

    columns = ['name'] + fields + export_params
    df = pd.DataFrame(values, columns=columns)
    if sort_rows:
        df['performance_metric'] = df['acc_clean'] / df['acc_corr']
        df.sort_values('performance_metric', ascending=True, inplace=True)
        df.drop('performance_metric', axis=1, inplace=True)
    if export_org:
        return org_table(df)
    else:
        return df


def parse_test_optimizer_l2_log(root, exclude=None, export_org=False,
                                export_test_params=None):
    fields = [
        'nll_loss', 'acc', 'acc_l2', 'acc_l2_0.5', 'acc_l2_1.0', 'acc_l2_1.5',
        'acc_l2_2.0', 'acc_l2_2.5', 'acc_l2_3.0', 'l2', 'l2_norm', 'l2_corr'
    ]
    if exclude is not None:
        for f in exclude:
            fields.remove(f)
    if export_test_params is None:
        export_test_params = []

    failed = []
    values = []
    for index, load_dir in enumerate(sorted(Path(root).glob('*'))):
        load_dir = str(load_dir)
        with open(os.path.join(load_dir, 'tensorflow.log'), 'r') as f:
            test_param_str = f.readline()
            test_params = load_params(test_param_str)
            text = "\n".join(f.readlines())
        name = os.path.basename(load_dir)

        try:
            test_results = re.findall("(?<=Test results).*", text)[-1]

            def parse_values(s):
                for field in fields:
                    curr_value.append(parse_float_value(field, s))

            curr_value = [name]
            parse_values(test_results)
            for param_name in export_test_params:
                curr_value.append(test_params[param_name])
            values.append(tuple(curr_value))
        except:
            failed.append(load_dir)

    if len(failed) > 0:
        print("Failed to parse directories: %s" % failed)

    columns = (['name'] + fields + export_test_params)
    df = pd.DataFrame(values, columns=columns)
    if export_org:
        return org_table(df)
    else:
        return df


def parse_test_optimizer_l1_log(root, exclude=None, export_org=False,
                                export_test_params=None):
    fields = [
        'nll_loss', 'acc', 'acc_l1', 'acc_l1_2.0', 'acc_l1_2.5', 'acc_l1_4.0',
        'acc_l1_5.0', 'acc_l1_6.0', 'acc_l1_7.5', 'acc_l1_8.0', 'acc_l1_8.75',
        'acc_l1_10.0', 'acc_l1_12.5', 'acc_l1_16.25', 'acc_l1_20.0', 'l1',
        'l1_corr'
    ]
    if exclude is not None:
        for f in exclude:
            fields.remove(f)
    if export_test_params is None:
        export_test_params = []

    failed = []
    values = []
    for index, load_dir in enumerate(sorted(Path(root).glob('*'))):
        load_dir = str(load_dir)
        with open(os.path.join(load_dir, 'tensorflow.log'), 'r') as f:
            test_param_str = f.readline()
            test_params = load_params(test_param_str)
            text = "\n".join(f.readlines())
        name = os.path.basename(load_dir)

        try:
            test_results = re.findall("(?<=Test results).*", text)[-1]

            def parse_values(s):
                for field in fields:
                    curr_value.append(parse_float_value(field, s))

            curr_value = [name]
            parse_values(test_results)
            for param_name in export_test_params:
                curr_value.append(test_params[param_name])
            values.append(tuple(curr_value))
        except:
            failed.append(load_dir)

    if len(failed) > 0:
        print("Failed to parse directories: %s" % failed)

    columns = (['name'] + fields + export_test_params)
    df = pd.DataFrame(values, columns=columns)
    if export_org:
        return org_table(df)
    else:
        return df


def parse_test_carlini_log(root, exclude=None, export_org=False,
                           export_test_params=None):
    fields = ['nll', 'acc', 'acc_ca', 'conf_ca',
              'l2_ca', 'l2_ca_norm']
    if exclude is not None:
        for f in exclude:
            fields.remove(f)
    if export_test_params is None:
        export_test_params = []

    failed = []
    values = []
    for index, load_dir in enumerate(sorted(Path(root).glob('*'))):
        load_dir = str(load_dir)
        with open(os.path.join(load_dir, 'tensorflow.log'), 'r') as f:
            test_param_str = f.readline()
            test_params = load_params(test_param_str)
            text = "\n".join(f.readlines())
        name = os.path.basename(load_dir)

        try:
            summary_results = re.findall("(?=Summary results).*", text)[0]
            test_results = re.findall("(?<=Test results).*", text)[-1]

            def parse_values(s):
                for field in fields:
                    curr_value.append(parse_float_value(field, s))

            curr_value = [name]
            parse_values(summary_results)
            parse_values(test_results)
            for param_name in export_test_params:
                curr_value.append(test_params[param_name])
            values.append(tuple(curr_value))
        except:
            failed.append(load_dir)

    if len(failed) > 0:
        print("Failed to parse directories: %s" % failed)

    columns = (['name'] + ['summary_' + f
                           for f in fields] + fields + export_test_params)
    df = pd.DataFrame(values, columns=columns)
    if export_org:
        return org_table(df)
    else:
        return df


def main(unused_args):
    np.random.seed(1)
    if FLAGS.wildcard is None:
        return

    if not FLAGS.wildcard.endswith("*"):
        FLAGS.wildcard += "*"

    export_params = []
    logs = parse_train_attack_log(FLAGS.wildcard, export_params=export_params)
    print(logs)

    def group_by(id):
        name = logs.get_value(id, "name")
        unique_name = re.findall("[^0-9]+", name)[0]
        return unique_name

    grouped = logs.groupby(group_by, axis=0)
    for name, group in grouped:
        print(name)
        print(group.agg([np.mean, np.std]))


if __name__ == '__main__':
    absl.app.run(main)
