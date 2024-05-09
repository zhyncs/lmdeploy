// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/medusa_plugin/medusa_util.h"

#include <algorithm>
#include <fstream>
#include <queue>
#include <sstream>

namespace turbomind {

void MedusaPathTree::get_batched_output_ids(const int* output_preds_batched,
                                            const int  medusa_head_num,
                                            const int  batch_num,
                                            int*       output_ids_batched,
                                            int*       each_path_len)
{
    int  offset        = path_num_ * (1 + medusa_head_num);
    bool is_calculated = false;
    for (int bid = 0; bid < batch_num; bid++) {
        const int* output_preds = output_preds_batched + bid * len_;
        int*       output_ids   = output_ids_batched + bid * offset;
        if (is_calculated) {
            get_output_ids(output_preds, medusa_head_num, output_ids, nullptr);
        }
        else {
            get_output_ids(output_preds, medusa_head_num, output_ids, each_path_len);
            is_calculated = true;
        }
    }
}

void MedusaPathTree::get_batched_last_match_idx(const int* max_match_idx,
                                                const int* max_match_count,
                                                const int  batch_size,
                                                int*       last_input_idx)
{
    for (int bid = 0; bid < batch_size; bid++) {
        get_last_match_idx(max_match_idx[bid], max_match_count[bid], last_input_idx[bid]);
    }
}

void MedusaPathTree::get_batched_matched_part_idx(const int* max_match_idx,
                                                  const int* max_match_count,
                                                  const int  batch_size,
                                                  const int  medusa_head_num,
                                                  int*       matched_part_input_idx)
{
    for (int bid = 0; bid < batch_size; bid++) {
        get_matched_part_idx(
            max_match_idx[bid], max_match_count[bid], matched_part_input_idx + bid * (1 + medusa_head_num));
    }
}

void MedusaPathTree::get_batched_pseudo_ids_from_tree(const int* medusa_preds_batched,
                                                      const int  medusa_head_num,
                                                      const int  top_k,
                                                      const int* max_match_count,
                                                      const int* max_match_idx,
                                                      const int  batch_size,
                                                      int*       pseudo_inputs_batched)
{
    for (int bid = 0; bid < batch_size; bid++) {
        const int* medusa_preds  = medusa_preds_batched + bid * medusa_head_num * top_k;
        int*       pseudo_inputs = pseudo_inputs_batched + bid * (len_ - 1);
        get_pseudo_ids_from_tree(
            medusa_preds, medusa_head_num, top_k, max_match_count[bid], max_match_idx[bid], pseudo_inputs);
    }
}

void MedusaPathTree::get_or_create_medusa_ti(int** medusa_ti, int& len)
{
    if (!medusa_ti_) {
        bfs();
    }
    *medusa_ti = medusa_ti_;
    len        = len_;
}

void MedusaPathTree::get_or_create_medusa_mask(int** medusa_mask, int& len)
{
    if (!medusa_mask_) {
        dfs();
    }
    *medusa_mask = medusa_mask_;
    len          = len_;
    path_num_    = input_token_idx_of_paths_.size();
}

void MedusaPathTree::insert(const std::vector<std::vector<int>>& path_tuples)
{
    for (const auto& path_tuple : path_tuples) {
        insert(path_tuple);
    }
}

void MedusaPathTree::insert(const std::vector<int>& path_tuple)
{
    MedusaPathTreeNode* node  = root_;
    int                 depth = 0;
    for (int path : path_tuple) {
        ++depth;
        if (!node->childs_[path]) {
            node->childs_[path] = new MedusaPathTreeNode(path, depth, 0, false);
        }
        node = node->childs_[path];
    }
    node->is_leaf_ = true;
}

void MedusaPathTree::bfs(MedusaPathTreeNode* root)
{
    std::queue<MedusaPathTreeNode*> q;
    q.push(root);

    std::map<int, int> depth_count;
    len_ = 0;

    while (!q.empty()) {
        MedusaPathTreeNode* node = q.front();
        q.pop();

        if (!node) {
            break;
        }

        topk_value_of_paths_.push_back(node->top_k_idx_);
        node->input_token_index_ = len_++;
        ++depth_count[node->depth_];

        for (const auto& pair : node->childs_) {
            q.push(pair.second);
        }
    }
    medusa_ti_ = new int[len_];

    int l = 0, r = 0;
    for (const auto& ele : depth_count) {
        int pos = ele.first;
        int cnt = ele.second;
        r       = l + cnt;
        while (l < r) {
            medusa_ti_[l++] = pos;
        }
        l = r;
    }
}

void MedusaPathTree::bfs()
{
    bfs(root_);
}

void MedusaPathTree::dfs(MedusaPathTreeNode* node, std::vector<int>& ancestor_ids)
{
    if (node) {
        ancestor_ids.push_back(node->input_token_index_);
        if (node->is_leaf_) {
            input_token_idx_of_paths_.push_back(ancestor_ids);
        }
        int& row = node->input_token_index_;

        for (int col : ancestor_ids) {
            medusa_mask_[row * len_ + col] = 1;
        }

        for (const auto& pair : node->childs_) {
            dfs(pair.second, ancestor_ids);
        }
        ancestor_ids.pop_back();
    }
}

void MedusaPathTree::dfs()
{
    medusa_mask_ = new int[len_ * len_];
    std::fill_n(medusa_mask_, len_ * len_, 0);

    std::vector<int> ancestor_ids;
    dfs(root_, ancestor_ids);
}

void MedusaPathTree::delete_tree(MedusaPathTreeNode* node)
{
    if (node) {
        for (const auto& pair : node->childs_) {
            delete_tree(pair.second);
        }
        delete node;
    }
}

void MedusaPathTree::get_output_ids(const int* output_preds,
                                    const int  medusa_head_num,
                                    int*       output_ids,
                                    int*       each_path_len)
{
    // output_preds: [input_len]
    // output_ids: [path_num, 1 + medusa_head_num]
    int col_base = 1 + medusa_head_num;

    auto to_dst_idx = [&col_base](int r, int c) { return r * col_base + c; };

    int r = 0, c = 0;
    int index_now   = 0;
    int padding_val = output_preds[0];

    for (const auto& indices : input_token_idx_of_paths_) {
        c = 0;
        for (int each_index : indices) {
            index_now             = to_dst_idx(r, c);
            output_ids[index_now] = output_preds[each_index];
            ++c;
        }
        if (each_path_len) {
            each_path_len[r] = c - 1;  // exclude root.
        }
        while (c < col_base) {  // paddings
            index_now             = to_dst_idx(r, c);
            output_ids[index_now] = padding_val;
            ++c;
        }

        ++r;
    }
}

void MedusaPathTree::get_pseudo_ids_from_tree(const int* medusa_preds,
                                              const int  medusa_head_num,
                                              const int  top_k,
                                              const int  max_match_count,
                                              const int  max_match_idx,
                                              int*       pseudo_inputs)
{
    // medusa_preds: [medusa_head_num, topk]
    // pseudo_inputs: [len_ - 1] return the values except root
    int counter = 0;

    bool skip_root = true;
    for (int topk_value : topk_value_of_paths_) {
        if (skip_root) {
            skip_root = false;
            counter++;
            continue;
        }

        const int medusa_head_id    = medusa_ti_[counter] - 1;
        const int medusa_head_value = medusa_preds[medusa_head_id * top_k + topk_value];
        pseudo_inputs[counter - 1]  = medusa_head_value;
        counter++;
    }
}

void MedusaPathTree::get_last_match_idx(const int max_match_idx, const int max_match_count, int& last_input_idx)
{
    last_input_idx = input_token_idx_of_paths_[max_match_idx][max_match_count];
}

void MedusaPathTree::get_matched_part_idx(const int& max_match_idx,
                                          const int& max_match_count,
                                          int*       matched_part_input_idx)
{
    auto& paths = input_token_idx_of_paths_[max_match_idx];
    // include root
    for (int i = 0; i <= max_match_count; i++) {
        matched_part_input_idx[i] = paths[i];
    }
}

MedusaPathTree& MedusaUtil::get_path_tree() const
{
    return *path_tree_.get();
}

void MedusaUtil::get_medusa_ti(int** medusa_ti)
{
    path_tree_->get_or_create_medusa_ti(medusa_ti, input_len_);
}

void MedusaUtil::get_medusa_mask(int** medusa_mask)
{
    path_tree_->get_or_create_medusa_mask(medusa_mask, input_len_);
}

int MedusaUtil::get_input_len() const
{
    return input_len_;
}

int MedusaUtil::get_path_num() const
{
    return path_num_;
}

std::string MedusaUtil::remove_all_white_spaces(std::string& str)
{
    str.erase(std::remove_if(str.begin(), str.end(), [](unsigned char ch) { return std::isspace(ch); }), str.end());
    return str;
}

std::vector<int> MedusaUtil::parse_tuple_str_2_tuple_int(const std::string& tuple_str)
{
    std::vector<int>  tuple;
    std::stringstream ss(tuple_str);
    std::string       item;
    std::getline(ss, item, '(');
    while (std::getline(ss, item, ',')) {
        tuple.push_back(std::stoi(item));
    }
    return tuple;
}

std::vector<std::vector<int>> MedusaUtil::get_medusa_paths_from_local_file(const std::string& local_path,
                                                                           const std::string& medusa_choice_name)
{
    std::ifstream                 file(local_path);
    std::vector<std::vector<int>> tuples;
    std::string                   line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string       readin_model_name;
        std::string       tuple_str;
        std::getline(ss, readin_model_name, '=');

        readin_model_name = remove_all_white_spaces(readin_model_name);

        if (readin_model_name != medusa_choice_name) {
            continue;
        }
        else {
            while (std::getline(ss, tuple_str, ')')) {
                if (tuple_str.size() > 1) {
                    std::vector<int> tuple = parse_tuple_str_2_tuple_int(tuple_str);
                    tuples.push_back(tuple);
                }
            }
            return tuples;
        }
    }
    return tuples;
}
}  // namespace turbomind
