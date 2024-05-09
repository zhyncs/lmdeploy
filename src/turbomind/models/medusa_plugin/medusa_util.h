// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace turbomind {

struct MedusaPathTreeNode {
    int                                top_k_idx_;
    int                                depth_;
    int                                input_token_index_;
    bool                               is_leaf_;
    std::map<int, MedusaPathTreeNode*> childs_;

    MedusaPathTreeNode(int top_k_idx, int depth, int input_token_index, bool is_leaf):
        top_k_idx_(top_k_idx), depth_(depth), input_token_index_(input_token_index), is_leaf_(is_leaf)
    {
    }
};

class MedusaPathTree {
public:
    MedusaPathTree(): root_(new MedusaPathTreeNode(0, 0, 0, false)) {}
    ~MedusaPathTree()
    {
        delete_tree(root_);
        delete[] medusa_mask_;
        medusa_mask_ = nullptr;
        delete[] medusa_ti_;
        medusa_ti_ = nullptr;
    }

public:
    void get_batched_output_ids(const int* output_preds_batched,
                                const int  medusa_head_num,
                                const int  batch_num,
                                int*       output_ids_batched,
                                int*       each_path_len);
    void get_batched_last_match_idx(const int* max_match_idx,
                                    const int* max_match_count,
                                    const int  batch_size,
                                    int*       last_input_idx);
    void get_batched_matched_part_idx(const int* max_match_idx,
                                      const int* max_match_count,
                                      const int  batch_size,
                                      const int  medusa_head_num,
                                      int*       matched_part_input_idx);
    void get_batched_pseudo_ids_from_tree(const int* medusa_preds_batched,
                                          const int  medusa_head_num,
                                          const int  top_k,
                                          const int* max_match_count,
                                          const int* max_match_idx,
                                          const int  batch_size,
                                          int*       pseudo_inputs_batched);

    void get_or_create_medusa_ti(int** medusa_ti, int& len);
    void get_or_create_medusa_mask(int** medusa_mask, int& len);
    void insert(const std::vector<std::vector<int>>& path_tuples);

private:
    MedusaPathTreeNode*           root_        = nullptr;
    int*                          medusa_mask_ = nullptr;
    int*                          medusa_ti_   = nullptr;
    int                           len_         = 0;
    int                           path_num_    = 0;
    std::vector<std::vector<int>> input_token_idx_of_paths_;
    std::vector<int>              topk_value_of_paths_;

    void insert(const std::vector<int>& path_tuple);
    void bfs(MedusaPathTreeNode* root);
    void bfs();
    void dfs(MedusaPathTreeNode* node, std::vector<int>& ancestor_ids);
    void dfs();
    void delete_tree(MedusaPathTreeNode* root);
    void get_output_ids(const int* output_preds, const int medusa_head_num, int* output_ids, int* each_path_len);
    void get_pseudo_ids_from_tree(const int* medusa_preds,
                                  const int  medusa_head_num,
                                  const int  top_k,
                                  const int  max_match_count,
                                  const int  max_match_idx,
                                  int*       pseudo_inputs);
    void get_last_match_idx(const int max_match_idx, const int max_match_count, int& last_input_idx);
    void get_matched_part_idx(const int& max_match_idx, const int& max_match_count, int* matched_part_input_idx);
};

class MedusaUtil {
public:
    MedusaUtil(const std::string& medusa_choice_path, const std::string& medusa_choice_name):
        medusa_choice_path_(medusa_choice_path), medusa_choice_name_(medusa_choice_name)
    {
        medusa_path_tuples_ = get_medusa_paths_from_local_file(medusa_choice_path_, medusa_choice_name_);
        path_num_           = medusa_path_tuples_.size();
        path_tree_          = std::make_unique<MedusaPathTree>();
        path_tree_->insert(medusa_path_tuples_);
        int* medusa_ti   = nullptr;
        int* medusa_mask = nullptr;
        path_tree_->get_or_create_medusa_ti(&medusa_ti, input_len_);
        path_tree_->get_or_create_medusa_mask(&medusa_mask, input_len_);
    }

    ~MedusaUtil() = default;

public:
    MedusaPathTree& get_path_tree() const;
    void            get_medusa_ti(int** medusa_ti);
    void            get_medusa_mask(int** medusa_mask);
    int             get_input_len() const;
    int             get_path_num() const;

private:
    std::string                     medusa_choice_path_;
    std::string                     medusa_choice_name_;
    std::vector<std::vector<int>>   medusa_path_tuples_;
    size_t                          path_num_;
    int                             input_len_;
    std::unique_ptr<MedusaPathTree> path_tree_;

    std::string                   remove_all_white_spaces(std::string& str);
    std::vector<int>              parse_tuple_str_2_tuple_int(const std::string& tuple_str);
    std::vector<std::vector<int>> get_medusa_paths_from_local_file(const std::string& local_path,
                                                                   const std::string& medusa_choice_name);
};

}  // namespace turbomind
