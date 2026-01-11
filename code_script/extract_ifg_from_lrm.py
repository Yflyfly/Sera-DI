import json
import re
import numpy as np
from extract_bbcfg_from_lrm import extract_group_from_lrm, map_kn_and_group, extract_basic_block_control_flow


def check_index(text, char_to_find, separator):
    separator_index = text.find(separator)

    if separator_index != -1:
        search_start_index = separator_index + len(separator)
        substring = text[search_start_index:]
        indices = []
        current_index = substring.find(char_to_find)
        while current_index != -1:
            actual_index = search_start_index + current_index
            indices.append(actual_index)
            current_index = substring.find(char_to_find, current_index + 1)
        return indices


def extract_icfg_from_bbcfg(kn_bb_lrm_dict, kn_bb_cf_dict, kn_jcf_dict):
    kn_icfg_dict = {}
    for kn, bb_lrm_dict in kn_bb_lrm_dict.items():
        # instr. to bb, instr. to instr. (icfg)
        k_ii_relations = []
        for bbn, lrm_list in bb_lrm_dict.items():
            if len(lrm_list) <= 1:
                continue
            bb_pc_list = []
            for line in lrm_list:
                pc = re.findall(r'/\*(.*?)\*/', line)
                bb_pc_list.append(pc[0])
            for i in range(len(bb_pc_list)-1):
                k_ii_relations.append([bb_pc_list[i], bb_pc_list[i+1]])
        ii_j_relations = []
        bbcf = kn_bb_cf_dict[kn]['basic_block_control']
        for b2b in bbcf:
            b_lrm_last = bb_lrm_dict[b2b[0]][-1]
            last_i = re.findall(r'/\*(.*?)\*/', b_lrm_last)[0]
            if b2b[1] == '_Z4Fan1PfS_ii':
                b2b[1] = '.text._Z4Fan1PfS_ii'
            if b2b[1] == '_Z6euclidP7latLongPfiff':
                b2b[1] = '.text._Z6euclidP7latLongPfiff'
            b_lrm_first = bb_lrm_dict[b2b[1]][0]
            first_i = re.findall(r'/\*(.*?)\*/', b_lrm_first)[0]
            ii_j_relations.append([last_i, first_i])
        jcf = kn_jcf_dict[kn]
        for ji, t_bb in jcf.items():
            if t_bb == '_Z4Fan1PfS_ii':
                t_bb = '.text._Z4Fan1PfS_ii'
            if t_bb == '_Z6euclidP7latLongPfiff':
                t_bb = '.text._Z6euclidP7latLongPfiff'
            target_lrm = bb_lrm_dict[t_bb][0]
            ti = re.findall(r'/\*(.*?)\*/', target_lrm)[0]
            if [ji, ti] not in ii_j_relations:
                ii_j_relations.append([ji, ti])
        k_ii_relations.extend(ii_j_relations)
        kn_icfg_dict[kn] = k_ii_relations
    return kn_icfg_dict


def extract_idfg_from_lrm(k_lrm_dict):
    kn_idfg_dict = {}
    for kn, lrm_list in k_lrm_dict.items():
        k_ii_relations = []
        for i in range(len(lrm_list)-1):
            assign_indices = check_index(lrm_list[i], '^', '// ')
            assign_indices.extend(check_index(lrm_list[i], 'x', '// '))
            if len(assign_indices) != 0:
                pc = re.findall(r'/\*(.*?)\*/', lrm_list[i])[0]
                for ai in assign_indices:
                    follow_lrm = lrm_list[i+1:]
                    for fl in follow_lrm:
                        if fl[ai] == 'x':
                            pc_to = re.findall(r'/\*(.*?)\*/', fl)[0]
                            if [pc, pc_to] not in k_ii_relations:
                                k_ii_relations.append([pc, pc_to])
                            break
                        elif fl[ai] == 'v':
                            pc_to = re.findall(r'/\*(.*?)\*/', fl)[0]
                            if [pc, pc_to] not in k_ii_relations:
                                k_ii_relations.append([pc, pc_to])
                        elif fl[ai] == '^':
                            break
        kn_idfg_dict[kn] = k_ii_relations
    return kn_idfg_dict


def integrate_nodes_and_relations(hr_dict, icf_dict, idf_dict):
    element_dict = hr_dict['bi']
    unique_ids = {}
    result = {}
    current_id = 0
    for kernel_name, basic_block_dict in element_dict.items():
        unique_ids[kernel_name] = current_id
        current_id += 1
        kernel_name_id = unique_ids[kernel_name]
        result[kernel_name_id] = {}
        for basic_block_name, pcs in basic_block_dict.items():
            unique_ids[f"{kernel_name}*{basic_block_name}"] = current_id
            current_id += 1
            basic_block_name_id = unique_ids[f"{kernel_name}*{basic_block_name}"]
            result[kernel_name_id][basic_block_name_id] = []
            for pc in pcs:
                unique_ids[f"{kernel_name}*{pc}"] = current_id
                current_id += 1
                result[kernel_name_id][basic_block_name_id].append(unique_ids[f"{kernel_name}*{pc}"])

    full_map = []
    # kernel to kernel
    kk_relations = hr_dict['kk']
    for kk in kk_relations:
        full_map.append([unique_ids[kk[0]], unique_ids[kk[1]]])
    # kernel to basic block
    kb_relations = hr_dict['kb']
    for kb in kb_relations:
        full_map.append([unique_ids[kb[0]], unique_ids[f"{kb[0]}*{kb[1]}"]])
    # basic block to basic block
    bb_dict = hr_dict['bb']
    for kn, bb_relations in bb_dict.items():
        for bb in bb_relations:
            full_map.append([unique_ids[f"{kn}*{bb[0]}"], unique_ids[f"{kn}*{bb[1]}"]])
    # basic block to instr.
    kbi_dict = hr_dict['bi']
    for kn, bi_dict in kbi_dict.items():
        for bbn, pc_list in bi_dict.items():
            for pc in pc_list:
                full_map.append([unique_ids[f"{kn}*{bbn}"], unique_ids[f"{kn}*{pc}"]])
    icf_map = []
    for kn, icf_relations in icf_dict.items():
        for ii in icf_relations:
            if [unique_ids[f"{kn}*{ii[0]}"], unique_ids[f"{kn}*{ii[1]}"]] not in icf_map:
                icf_map.append([unique_ids[f"{kn}*{ii[0]}"], unique_ids[f"{kn}*{ii[1]}"]])
    idf_map = []
    for kn, idf_relations in idf_dict.items():
        for ii in idf_relations:
            if [unique_ids[f"{kn}*{ii[0]}"], unique_ids[f"{kn}*{ii[1]}"]] not in idf_map:
                idf_map.append([unique_ids[f"{kn}*{ii[0]}"], unique_ids[f"{kn}*{ii[1]}"]])

    icfg = full_map
    icfg.extend(icf_map)
    idfg = full_map
    idfg.extend(idf_map)

    return unique_ids, icfg, idfg, full_map


def pc_map_node_num(program, relation_dict):
    with open(f'../datasets/{program}/node_mapping.json', 'r', encoding='utf-8') as f:
        node_map_data = json.load(f)
    kp_dict = node_map_data["instruction"]
    edge_list = []
    for kn, r_list in relation_dict.items():
        for r in r_list:
            pc1, pc2 = r
            if pc1 != pc2:
                edge_list.append([kp_dict[f'{kn}-{pc1}'], kp_dict[f'{kn}-{pc2}']])

    return edge_list


def load_graph_adj(program, node_num):
    file_output_path = f"../datasets/{program}/{program}_lrm_clean.txt"
    pc_file_path = f"../datasets/{program}/pc_extraction.txt"
    groups = extract_group_from_lrm(file_output_path)
    kn_lrm_dict = map_kn_and_group(pc_file_path, groups)
    lrm_dict, cf_dict, jcf_dict = extract_basic_block_control_flow(kn_lrm_dict)
    icf = extract_icfg_from_bbcfg(lrm_dict, cf_dict, jcf_dict)
    c_edges = pc_map_node_num(program, icf)
    c_adj = np.zeros((node_num, node_num), dtype=np.float32)
    adj_eyes = np.eye(node_num)
    for i, j in c_edges:
        c_adj[i, j] = 1
        c_adj[j, i] = 1
    c_adj = c_adj + adj_eyes
    idf = extract_idfg_from_lrm(kn_lrm_dict)
    d_edges = pc_map_node_num(program, idf)
    d_adj = np.zeros((node_num, node_num), dtype=np.float32)
    for i, j in d_edges:
        d_adj[i, j] = 1
        d_adj[j, i] = 1
    d_adj = d_adj + adj_eyes

    return [c_adj, d_adj]

