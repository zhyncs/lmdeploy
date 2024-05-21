#include "src/turbomind/models/medusa_plugin/medusa_utils.h"
namespace turbomind {

void MedusaPathTree::insert(std::vector<int> path_tuple)
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
void MedusaPathTree::insert(std::vector<std::vector<int>> path_tuples)
{
    for (std::vector<int> path_tuple : path_tuples) {
        insert(path_tuple);
    }
}
void MedusaPathTree::deleteTree(MedusaPathTreeNode* node)
{
    if (node) {
        for (std::pair<int, MedusaPathTreeNode*> each_pair : node->childs_) {
            MedusaPathTreeNode* child_node = each_pair.second;
            deleteTree(child_node);
        }
        delete node;
    }
}
void MedusaPathTree::dbg()
{
    dbg(root_);
}
void MedusaPathTree::dbg(MedusaPathTreeNode* node)
{
    if (node) {
        for (int i = 0; i < node->depth_; i++) {
            std::cout << "\t";
        }
        std::cout << node->top_k_idx_;
        if (node->is_leaf_) {
            std::cout << "(l)";
        }
        std::cout << std::endl;
        for (std::pair<int, MedusaPathTreeNode*> each_pair : node->childs_) {
            MedusaPathTreeNode* child_node = each_pair.second;
            dbg(child_node);
        }
    }
}

void MedusaPathTree::getOrCreateMedusaTi(int** medusa_ti, int& len)
{
    if (!medusaTi_) {
        bfs();
    }
    *medusa_ti = medusaTi_;
    len        = len_;
}
void MedusaPathTree::bfs()
{
    bfs(root_);
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

        if (!node)
            break;

        topk_value_of_paths.push_back(node->top_k_idx_);

        node->input_token_index_ = len_++;

        ++depth_count[node->depth_];

        for (std::pair<int, MedusaPathTreeNode*> each_pair : node->childs_) {
            MedusaPathTreeNode* child_node = each_pair.second;
            q.push(child_node);
        }
    }
    medusaTi_ = new int[len_];

    int l = 0, r = 0;
    for (auto it : depth_count) {
        int pos = it.first;
        int cnt = it.second;
        r       = l + cnt;
        while (l < r) {
            medusaTi_[l++] = pos;
        }
        l = r;
    }
}
void MedusaPathTree::dfs()
{
    medusaMask_ = new int[len_ * len_];
    std::fill_n(medusaMask_, len_ * len_, 0);

    std::vector<int> ancestor_ids;
    dfs(root_, ancestor_ids);
}
void MedusaPathTree::dfs(MedusaPathTreeNode* node, std::vector<int>& ancestor_ids)
{
    if (node) {
        ancestor_ids.push_back(node->input_token_index_);
        if (node->is_leaf_) {
            std::vector<int> input_token_index_of_path = ancestor_ids;
            input_token_idx_of_paths.push_back(std::move(input_token_index_of_path));
        }
        int& row = node->input_token_index_;

        for (const int& col : ancestor_ids) {
            medusaMask_[row * len_ + col] = 1;
        }

        for (std::pair<int, MedusaPathTreeNode*> each_pair : node->childs_) {
            MedusaPathTreeNode* child_node = each_pair.second;
            dfs(child_node, ancestor_ids);
        }
        ancestor_ids.pop_back();
    }
}
void MedusaPathTree::getOrCreateMedusaMask(int** medusa_mask, int& len)
{
    if (!medusaMask_) {
        dfs();
    }
    *medusa_mask = medusaMask_;
    len          = len_;
    path_num_    = input_token_idx_of_paths.size();
}
void MedusaPathTree::getOutputIds(const int* output_preds,
                                  int*       output_ids,
                                  int*       each_path_len,
                                  const int  medusa_head_num)
{
    // input:
    //      output_preds : [input_len]
    // output:
    //      output_ids : [path_num, 1 + medusa_head_num]
    int col_base = 1 + medusa_head_num;

    auto to_dst_idx = [&col_base](int r, int c) { return r * col_base + c; };

    int r = 0, c = 0;
    int index_now   = 0;
    int padding_val = output_preds[0];

    for (std::vector<int>& indices : input_token_idx_of_paths) {
        c = 0;
        for (int each_index : indices) {
            index_now             = to_dst_idx(r, c);
            output_ids[index_now] = output_preds[each_index];
            ++c;
        }
        if (each_path_len) {
            each_path_len[r] = c - 1;  // only consider medusa path len, not include root.
        }
        while (c < col_base) {  // paddings
            index_now             = to_dst_idx(r, c);
            output_ids[index_now] = padding_val;
            ++c;
        }

        ++r;
    }
}

void MedusaPathTree::getBatchedOutputIds(const int* output_preds_batched,
                                         int*       output_ids_batched,
                                         int*       each_path_len,
                                         const int  medusa_head_num,
                                         const int  batch_num)
{
    // input:
    //      output_preds : [batch_num, input_len]
    // output:
    //      output_ids : [batch_num, path_num, 1 + medusa_head_num]
    //      each_path_len : [path_num, 1]

    int  offset        = path_num_ * (1 + medusa_head_num);
    bool is_calculated = false;
    for (int bid = 0; bid < batch_num; bid++) {
        const int* output_preds = output_preds_batched + bid * len_;
        int*       output_ids   = output_ids_batched + bid * offset;
        if (is_calculated) {
            getOutputIds(output_preds, output_ids, nullptr, medusa_head_num);
        }
        else {
            getOutputIds(output_preds, output_ids, each_path_len, medusa_head_num);
            is_calculated = true;
        }
    }
}

int MedusaPathTree::getMedusaPathNum()
{
    return path_num_;
}

int MedusaPathTree::getMedusaInputLen()
{
    return len_;
}

void MedusaPathTree::getPseudoIdsFromTree(const int* medusa_preds,
                                          const int  medusa_head_num,
                                          const int  top_k,
                                          const int  max_match_count,
                                          const int  max_match_idx,
                                          int*       pseudo_inputs)
{
    // input:
    //      medusa_preds: [medusa_head_num, topk]
    // output:
    //      pseudo_inputs:[len_ - 1], return the values except root.
    int counter = 0;

    bool skip_root = true;
    for (int& topk_value : topk_value_of_paths) {

        if (skip_root) {
            skip_root = false;
            counter++;
            continue;
        }

        const int& medusa_head_id    = medusaTi_[counter] - 1;
        const int& medusa_head_value = medusa_preds[medusa_head_id * top_k + topk_value];
        pseudo_inputs[counter - 1]   = medusa_head_value;
        counter++;
    }
}

void MedusaPathTree::getBatchedPseudoIdsFromTree(const int* medusa_preds_batched,
                                                 const int  medusa_head_num,
                                                 const int  top_k,
                                                 const int* max_match_count,
                                                 const int* max_match_idx,
                                                 int*       pseudo_inputs_batched,
                                                 const int  batch_size)
{
    // input:
    //      medusa_preds_batched: [batch_size, medusa_head_num, topk]
    // output:
    //      pseudo_inputs_batched:[batch_size, len_ - 1]
    for (int b_id = 0; b_id < batch_size; b_id++) {
        const int* medusa_preds  = medusa_preds_batched + b_id * medusa_head_num * top_k;
        int*       pseudo_inputs = pseudo_inputs_batched + b_id * (len_ - 1);
        getPseudoIdsFromTree(
            medusa_preds, medusa_head_num, top_k, max_match_count[b_id], max_match_idx[b_id], pseudo_inputs);
    }
}

void MedusaPathTree::getLastMatchIdx(const int& max_match_idx, const int& max_match_count, int& last_input_idx)
{
    last_input_idx = input_token_idx_of_paths[max_match_idx][max_match_count];
}

void MedusaPathTree::getBatchedLastMatchIdx(const int* max_match_idx,
                                            const int* max_match_count,
                                            int*       last_input_idx,
                                            const int  batch_size)
{
    /*
    input :
        max_match_idx : [batch_size],
        max_match_count : [batch_size],
    output :
        last_input_idx : [batch_size],
    */
    for (int b_id = 0; b_id < batch_size; b_id++) {
        getLastMatchIdx(max_match_idx[b_id], max_match_count[b_id], last_input_idx[b_id]);
    }
}

void MedusaPathTree::getMatchedPartIdx(const int& max_match_idx,
                                       const int& max_match_count,
                                       int*       matched_part_input_idx)
{
    auto& paths = input_token_idx_of_paths[max_match_idx];
    // include root
    for (int i = 0; i <= max_match_count; i++) {
        matched_part_input_idx[i] = paths[i];
    }
}
void MedusaPathTree::getBatchedMatchedPartIdx(const int* max_match_idx,
                                              const int* max_match_count,
                                              int*       matched_part_input_idx,
                                              const int  batch_size,
                                              const int  medusa_head_num)
{
    /*
    input :
        max_match_idx : [batch_size],
        max_match_count : [batch_size],
    output :
        matched_part_input_idx : [batch_size, 1 + medusa_head_num],
    */

    for (int b_id = 0; b_id < batch_size; b_id++) {
        getMatchedPartIdx(
            max_match_idx[b_id], max_match_count[b_id], matched_part_input_idx + b_id * (1 + medusa_head_num));
    }
}

void MedusaUtils::getTokenIdsAccordingToPath(int*                           medusa_path_tokens_out,
                                             const size_t&                  path_num,
                                             const int*                     medusa_pred_tokens,
                                             std::vector<std::vector<int>>& path_tuples,
                                             const int                      batch_size,
                                             const int                      medusa_head_num,
                                             const int                      K)
{
    // input:[medusa_head_num, batch_size, topk], output:[path_num, batch_size, medusa_head_num]

    auto to_src_idx = [batch_size, K](int h_idx, int b_idx, int topk_idx) {
        return h_idx * batch_size * K + b_idx * K + topk_idx;
    };
    auto to_dst_idx = [batch_size, medusa_head_num](int p_idx, int b_idx, int h_idx) {
        return p_idx * batch_size * medusa_head_num + b_idx * medusa_head_num + h_idx;
    };

    for (int b_id = 0; b_id < batch_size; ++b_id) {
        for (int path_id = 0; path_id < path_num; ++path_id) {
            std::vector<int>& path_tuple = path_tuples[path_id];
            paddingTuple(path_tuple,
                         medusa_head_num,
                         0);  // if path[medusa_head i] not defined, we use topk = 0 as default for i.
            for (int head_id = 0; head_id < medusa_head_num; ++head_id) {
                int topk_id          = path_tuple[head_id];
                int pred_token_value = medusa_pred_tokens[to_src_idx(head_id, b_id, topk_id)];
                medusa_path_tokens_out[to_dst_idx(path_id, b_id, head_id)] = pred_token_value;
            }
        }
    }
}
std::vector<int> MedusaUtils::parseTupleStr2TupleInt(const std::string& tuple_str)
{
    // tuple_str format is without ')'
    // like:
    // "[(0, 1", "[(0, 1, "
    std::vector<int>  tuple;
    std::stringstream ss(tuple_str);
    std::string       item;
    // remove the util '('
    std::getline(ss, item, '(');
    while (std::getline(ss, item, ',')) {
        tuple.push_back(std::stoi(item));
    }
    return tuple;
}
std::vector<std::vector<int>> MedusaUtils::getMedusaPathsFromLocalFile(const std::string& local_path,
                                                                       const std::string& aim_model_name)
{
    // the medusa_paths format should like :
    //    modelname = [(0,), (0, 1), (2, 9, ), ...]
    std::ifstream                 file(local_path);
    std::vector<std::vector<int>> tuples;
    std::string                   line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string       readin_model_name;
        std::string       tuple_str;
        std::getline(ss, readin_model_name, '=');

        readin_model_name = removeAllWhiteSpaces(readin_model_name);

        if (readin_model_name != aim_model_name) {
            continue;
        }
        else {
            while (std::getline(ss, tuple_str, ')')) {
                if (tuple_str.size() > 1) {
                    std::vector<int> tuple = parseTupleStr2TupleInt(tuple_str);
                    tuples.push_back(tuple);
                }
            }
            return tuples;
        }
    }
    return tuples;
}

void MedusaUtils::displayPathTuples(std::vector<std::vector<int>>& path_tuples)
{
    // path_tuples is a [path_num, path_length] 2D vector.
    const int n = path_tuples.size();

    std::cout << "[";
    for (int i = 0; i < n; i++) {
        const int m = path_tuples[i].size();
        std::cout << "[";
        for (int j = 0; j < m; j++) {
            std::cout << "(" << j << "):" << path_tuples[i][j];
            if (j != m - 1)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "]\n";
}
std::string MedusaUtils::removeAllWhiteSpaces(std::string str)
{
    str.erase(std::remove_if(str.begin(), str.end(), [](unsigned char ch) { return std::isspace(ch); }), str.end());
    return str;
}
void MedusaUtils::paddingTuple(std::vector<int>& tuple, int aim_size, int padding_value)
{
    if (tuple.size() < aim_size) {
        tuple.resize(aim_size, padding_value);
    }
}
std::vector<std::vector<int>>& MedusaUtils::getPathTuples()
{
    return medusa_path_tuples_;
}

std::pair<size_t, size_t>
MedusaUtils::resultOffsetAndLength(int batch_idx, int path_idx, int batch_size, int medusa_head_num)
{
    return std::pair<size_t, size_t>{path_idx * batch_size * medusa_head_num + batch_idx * medusa_head_num,
                                     medusa_head_num};
}
void MedusaUtils::getPathNum(int& path_num)
{
    path_num = path_num_;
}

void MedusaUtils::getMedusaTi(int** medusa_ti)
{
    path_tree_.getOrCreateMedusaTi(medusa_ti, input_len_);
}
void MedusaUtils::getMedusaMask(int** medusa_mask)
{
    path_tree_.getOrCreateMedusaMask(medusa_mask, input_len_);
}
void MedusaUtils::getInputLen(int& len)
{
    len = input_len_;
}

}  // namespace turbomind
