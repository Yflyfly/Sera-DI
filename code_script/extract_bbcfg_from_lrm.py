import re


def extract_group_from_lrm(file_path):
    all_groups = []
    current_group = []
    with open(file_path, 'r') as file:
        for line in file:
            if ".global" in line:
                if current_group:
                    all_groups.append(current_group)
                    current_group = []
            current_group.append(line)
        if current_group:
            all_groups.append(current_group)

    return all_groups[1:]


def map_kn_and_group(pc_file, group_list):
    lines_list = []
    current_block = []
    with open(pc_file, 'r') as file1:
        for line in file1:
            line = line.strip()
            if line:
                current_block.append(line)
            elif current_block:
                lines_list.append(current_block)
                current_block = []
        if current_block:
            lines_list.append(current_block)

    kn_dict = {}
    for lines in lines_list:
        kn = lines[0].split(":")[1].strip()
        pc_dict = {}
        for line in lines[1:]:
            [pc, instruction, group_id] = line.split("\t")
            pc_dict[pc] = instruction
        kn_dict[kn] = pc_dict
    kn_lrm_dict = {}
    for group in group_list:
        for k in kn_dict.keys():
            kName = k[:k.index('(')]
            if kName in group[0]:
                kn_lrm_dict[kName] = group

    return kn_lrm_dict


def extract_basic_block_control_flow(lrm_data: dict):
    kn_bb_lrm_dict = {}
    kn_bb_cf_j_dict = {}
    bb_first_lrm = {}
    for kn, lrm_list in lrm_data.items():
        bb_dict = {}
        bb_ids = []
        pc_to_bb_dict = {}
        for i, lrm_line in enumerate(lrm_list):
            if lrm_line[0] == "." or lrm_line[0] == "$":
                bb_name = lrm_line[:lrm_line.index(":")]
                bb_dict[bb_name] = []
                bb_first_lrm[bb_name] = []
                bb_ids.append(i)
        for i in range(len(bb_ids)-1):
            bb_lines = lrm_list[bb_ids[i]:bb_ids[i+1]]
            bb_first_lrm[bb_lines[0][:bb_lines[0].index(":")]] = bb_lines[0]
            bb_lines_clean = []
            for line in bb_lines:
                pc = re.findall(r'/\*(.*?)\*/', line)
                if pc:
                    bb_lines_clean.append(line)
                    to_bb = re.findall(r'`\((.*?)\)', line)
                    if to_bb:
                        pc_to_bb_dict[pc[0]] = to_bb[0]
            bb_dict[bb_lines[0][:bb_lines[0].index(":")]] = bb_lines_clean
        kn_bb_lrm_dict[kn] = bb_dict
        kn_bb_cf_j_dict[kn] = pc_to_bb_dict

    kn_bb_pc_dict = {}
    for kn, bb_lrm_dict in kn_bb_lrm_dict.items():
        bbn_pc_dict = {}
        for bbn, lrm_list in bb_lrm_dict.items():
            bb_pc_list = []
            if len(lrm_list) == 0:
                continue
            for line in lrm_list:
                pc = re.findall(r'/\*(.*?)\*/', line)
                bb_pc_list.append(pc[0])
            bbn_pc_dict[bbn] = bb_pc_list
        kn_bb_pc_dict[kn] = bbn_pc_dict

    kn_bb_cf_dict = {}
    for kn, bb_pc_dict in kn_bb_pc_dict.items():
        bbn_list = list(bb_pc_dict.keys())
        k_bb_cf_j = []
        bb_cf_j = kn_bb_cf_j_dict[kn]
        for pc_j, to_bb in bb_cf_j.items():
            for bbn, pc_list in bb_pc_dict.items():
                if pc_j in pc_list:
                    if [bbn, to_bb] not in k_bb_cf_j:
                        k_bb_cf_j.append([bbn, to_bb])
                    break
        k_bb_cf_s = []
        for i in range(len(bbn_list)-1):
            if ([bbn_list[i], bbn_list[i+1]] not in k_bb_cf_j) and ('// +.' not in bb_first_lrm[bbn_list[i + 1]]):
                k_bb_cf_s.append([bbn_list[i], bbn_list[i+1]])
        k_bb_cf_j.extend(k_bb_cf_s)
        kn_bb_cf_dict[kn] = {
            "basic_block": bbn_list,
            "basic_block_control": k_bb_cf_j
        }

    return kn_bb_lrm_dict, kn_bb_cf_dict, kn_bb_cf_j_dict


def build_bbcfg(k_bbcf_dict, k_bblrm_dict):
    k_relations = []
    kn_list = list(k_bbcf_dict.keys())
    for i in range(len(kn_list)-1):
        k_relations.append([kn_list[i], kn_list[i+1]])
    # kernel to bb, bb to bb
    kb_relations = []
    bb_relations_dict = {}
    for kn, bb_dict in k_bbcf_dict.items():
        bb_list = bb_dict['basic_block']
        bbr_list = bb_dict['basic_block_control']
        for bb in bb_list:
            kb_relations.append([kn, bb])  # kernel to bb
        bb_relations_dict[kn] = bbr_list  # bb to bb
    kn_bb_pc_dict = {}
    for kn, bb_lrm_dict in k_bblrm_dict.items():
        bb_pc_dict = {}
        for bbn, lrm_list in bb_lrm_dict.items():
            if len(lrm_list) == 0:
                continue
            bb_pc_list = []
            for line in lrm_list:
                pc = re.findall(r'/\*(.*?)\*/', line)[0]
                bb_pc_list.append(pc)
            bb_pc_dict[bbn] = bb_pc_list
        kn_bb_pc_dict[kn] = bb_pc_dict
    k_b_i_dict = {
        "kk": k_relations,
        "kb": kb_relations,
        "bb": bb_relations_dict,
        "bi": kn_bb_pc_dict
    }
    for k, v in k_b_i_dict.items():
        print(k, v)
    return k_b_i_dict

