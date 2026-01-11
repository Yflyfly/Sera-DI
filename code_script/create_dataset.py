import pandas as pd
import numpy as np
import json


def process_query(queries_csv, node_map_json, p_name):
    df = pd.read_csv(queries_csv)
    df.fillna(0, inplace=True)
    outer_key_col = df.columns[0]
    if p_name in ["lavaMD", "conv2d", "gemm", "pathfinder", "nn"]:  # kernel:1
        sub_dict_cols = df.columns[2:]
    elif p_name in ["lud"]:  # kernel:3
        sub_dict_cols = df.columns[4:]
    else:
        sub_dict_cols = df.columns[3:]
    nested_dict = {}
    for _, row in df.iterrows():
        outer_key = row[outer_key_col]
        outer_value = {}
        for col in sub_dict_cols:
            kn, pc = col.split('-')
            kn = kn.split('(')[0]
            pc = pc[2:].zfill(4)
            outer_value[f"{kn}-{pc}"] = row[col]
        nested_dict[outer_key] = outer_value

    with open(node_map_json, 'r', encoding='utf-8') as f:
        node_map_data = json.load(f)
    kp_dict = node_map_data["instruction"]
    node_list = list(kp_dict.keys())

    query_dict = {}
    for k, v in nested_dict.items():
        query = [v[node] for node in node_list]
        query_dict[k] = query

    normalized_query_dict = {}
    for input_k, query_vector in query_dict.items():
        query_array = np.array(query_vector, dtype=float)
        min_val = query_array.min()
        max_val = query_array.max()
        normalized_query = (query_array - min_val) / (max_val - min_val)
        query = normalized_query.tolist()
        normalized_query_dict[input_k] = query

    return normalized_query_dict, query_dict


def encode_feature_to_onehot(value, encode_len):
    feature_map = [0] * encode_len
    # print(value, encode_len)
    feature_map[int(value)] = 1
    return feature_map


def encode_feature_to_binary(value, value_max):
    binary_representation = bin(value)[2:]
    target_length = len(bin(value_max)[2:])
    if len(binary_representation) < target_length:
        padded_binary = '0' * (target_length - len(binary_representation)) + binary_representation
    else:
        padded_binary = binary_representation
    binary_list = [int(bit) for bit in padded_binary]
    return binary_list


def encode_feature(value, value_max):
    if value_max > 64:
        feature_map = encode_feature_to_binary(value, value_max)
    else:
        feature_map = encode_feature_to_onehot(value, value_max)
    return feature_map


# 格式化数据集
def dataset_format(p_name, data_path, encode_json, node_map_json, input_count_map):
    df = data_path
    with open(encode_json, 'r', encoding='utf-8') as f:
        encode_dict = json.load(f)
    with open(node_map_json, 'r', encoding='utf-8') as f:
        node_map_data = json.load(f)
    kp_dict = node_map_data["instruction"]

    df["static_instID"] = df.apply(lambda row: kp_dict.get(f"{row['kernel_name'].split('(')[0]}-{row['pcOffset'][2:].zfill(4)}", None), axis=1)

    input_dict, raw_input_dict = process_query(input_count_map, node_map_json, p_name)

    mask_id_list = []
    inject_input_list = []
    label_list = []
    for index, row in df.iterrows():
        k_id_map = encode_feature(int(row["kernel_index"]), encode_dict["max_kernel_index"]+1)
        i_id_map = encode_feature(int(row["instID"]), encode_dict["max_instID"])
        r_id_map = encode_feature(int(row["regNo"]), encode_dict["max_regNo"]+1)
        fip_pos_map = encode_feature(int(row["fip_pos"]), 32)
        fip_d_map = [int(str(row["fip_tp"])[-1])]
        b_id_map = encode_feature(row["blockID"], encode_dict["max_blockID"]+1)
        bt_id_map = encode_feature(row["blockTID"], encode_dict["max_blockTID"]+1)
        input_vector = input_dict[row["input"]]
        i_rate = [int(row["instID"]) / sum(raw_input_dict[row["input"]])]

        inject_inform = k_id_map + i_id_map + i_rate + r_id_map + fip_pos_map + fip_d_map + b_id_map + bt_id_map + input_vector

        mask_id_list.append([row["static_instID"]])
        inject_input_list.append(inject_inform)
        label_list.append([row["inject_result"]])

    if p_name == "lud":
        label_list = [[0] if x == [2] else x for x in label_list]

    return mask_id_list, inject_input_list, label_list

