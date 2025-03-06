use std::fs::File;
use std::vec;
use tokenizers::Tokenizer;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            // todo!("self_attention(...)");
            self_attention(
                &mut hidden_states, 
                &mut att_scores,
                q, 
                full_k, 
                full_v,
                self.n_kv_h, 
                n_groups, 
                seq_len, 
                total_seq_len, 
                self.dqkv
            );
            // todo!("down_proj matmul and add residual");
            OP::matmul_transb(&mut residual, 1., &hidden_states, &self.params.wo[layer], 1.);

            // todo!("mlp(...)");
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32>{
        let mut result = Vec::<u32>::new();
        
        // todo!("实现文本生成");

        let mut cache = self.new_cache(); // 初始化缓存
        let mut input = Tensor::new(token_ids.to_vec(), &vec![token_ids.len()]);
        result.extend_from_slice(token_ids);

        // 生成循环
        while result.len() < max_len {
            // 前向计算
            let logits = self.forward(&input, &mut cache);
            
            // 采样下一个token
            let next_token = OP::random_sample(
                &logits, 
                top_p, 
                top_k, 
                temperature
            );
            
            // 终止条件检查
            if next_token == self.eos_token_id {
                break;
            }
            
            // 更新输入和结果
            input = Tensor::new(vec![next_token], &vec![1]);
            result.push(next_token);
            
        }
        
        result
    }

    pub fn generate_iter<'a>(
        &'a self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        mut cache: &'a mut KVCache<f32>,
    ) -> impl Iterator<Item = u32> + 'a {
        // 用于存储生成的 token 序列
        let mut generated_token_sequence = Vec::<u32>::new();
        // 将输入的 token 序列复制到一个新的 Vec 中
        let input_token_vec: Vec<u32> = token_ids.to_vec();
        // 把 token 序列转换为二维张量，形状为 (1, token_ids 的长度)
        let mut input_tensor = Tensor::<u32>::new(input_token_vec, &vec![1, token_ids.len()]);
        // 创建一个迭代器，通过闭包逻辑来生成 token
        std::iter::from_fn(move || {
            // 检查是否达到最大生成长度，如果达到则停止迭代
            if generated_token_sequence.len() >= max_len {
                return None;
            }
            // 执行前向传播，得到每个词的未归一化概率分布
            let probability_distribution = self.forward(&input_tensor, &mut cache);
            // 根据 top_p、top_k 和 temperature 策略从概率分布中采样下一个 token
            let next_generated_token = OP::random_sample(
                &probability_distribution,
                top_p,
                top_k,
                temperature,
            );
            // 将新生成的 token 添加到生成序列中
            generated_token_sequence.push(next_generated_token);
            // 检查是否生成了结束标记（EOS），如果是则停止迭代
            if next_generated_token == self.eos_token_id {
                return None;
            }
            // 更新输入张量，将新生成的 token 作为下一次的输入
            input_tensor = Tensor::<u32>::new(vec![next_generated_token], &vec![1, 1]);
            // 返回新生成的 token
            Some(next_generated_token)
        })
    }

    pub fn generate_cache(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        cache: &mut KVCache<f32>,
    ) -> Vec<u32>{

        let mut input = Tensor::new(token_ids.to_vec(), &vec![token_ids.len()]);
        let mut result = Vec::<u32>::new();

        while result.len() < max_len {
            let logits = self.forward(&input, cache);
            
            // 采样下一个token
            let next_token = OP::random_sample(
                &logits, 
                top_p, 
                top_k, 
                temperature
            );
            
            // 终止条件检查
            if next_token == self.eos_token_id {
                break;
            }
            
            // 更新输入和结果
            input = Tensor::new(vec![next_token], &vec![1]);
            result.push(next_token);
        }
        
        result
    }

    //AI chat
    pub fn chat(
        &self, 
        max_turns: usize,
        max_len: usize,
        top_p: f32, 
        top_k: u32, 
        temperature: f32) {

        let mut cache = self.new_cache();
        let tokenizer = Tokenizer::from_file("models/chat/tokenizer.json").unwrap();
        let mut dialog_history = String::new();
        
        // 对话模板常量
        const SYSTEM_PROMPT: &str = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n";
        const USER_PREFIX: &str = "<|im_start|>user\n";
        const ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n";
        const END_MARKER: &str = "<|im_end|>\n";

        dialog_history.push_str(SYSTEM_PROMPT);

        for turn in 0..max_turns {
            // 获取用户输入
            let mut user_input = String::new();
            println!("\nUser (turn {}):", turn + 1);
            std::io::stdin().read_line(&mut user_input).unwrap();
            user_input = user_input.trim().to_string();

            if user_input.to_lowercase() == "exit" {
                break;
            }

            // 构建prompt
            dialog_history.push_str(&format!("{}{}{}", 
                USER_PREFIX, 
                user_input, 
                END_MARKER
            ));
            dialog_history.push_str(ASSISTANT_PREFIX);

            // 编码输入
            let binding = tokenizer.encode(dialog_history.as_str(), true).unwrap();
            let input_ids = binding.get_ids();
            
            // 生成回复
            let result = self.generate(
                input_ids,
                500,  // 每轮最大生成长度
                top_p,
                top_k,
                temperature
            );
            
            // let mut input = Tensor::new(input_ids.to_vec(), &vec![input_ids.len()]);
            // let mut result = Vec::with_capacity(max_len);
            // result.extend_from_slice(input_ids);

            // // 生成循环
            // for _ in 0..max_len {
            //     // 前向计算
            //     let logits = self.forward(&input, &mut cache);
                
            //     // 采样下一个token
            //     let next_token = OP::random_sample(
            //         &logits, 
            //         top_p, 
            //         top_k, 
            //         temperature
            //     );
                
            //     // 终止条件检查
            //     if next_token == self.eos_token_id {
            //         break;
            //     }
                
            //     // 更新输入和结果
            //     input = Tensor::new(vec![next_token], &vec![1]);
            //     result.push(next_token);
                
            //     // 达到最大长度终止
            //     if result.len() >= max_len {
            //         break;
            //     }
            // }
            
            // 解码并更新历史
            let response = tokenizer.decode(&result, true).unwrap();
            let clean_response = response.replace(END_MARKER, "").trim().to_string();
            println!("Assistant: {}", clean_response);
            
            // 更新对话历史
            dialog_history.push_str(&format!("{}{}", clean_response, END_MARKER));
            
            // 缓存管理（限制历史长度）
            if dialog_history.len() > 4000 {
                dialog_history.drain(0..2000);
                cache = self.new_cache(); // 历史过长时重置缓存
            }
        }
    }

}



fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // todo!("Implement self_attention");
    {
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();
    let att_scores_data =  unsafe {att_scores.data_mut()};
    let hidden_data =  unsafe {hidden_states.data_mut()};
    let scale = 1. / (dqkv as f32).sqrt();
    
    for kv_head in 0..n_kv_h {
        for group in 0..n_groups {
            let q_group_start = kv_head * n_groups * dqkv + group * dqkv;
            
            for seq_pos in 0..seq_len {
                let q_offset = seq_pos * n_kv_h * n_groups * dqkv + q_group_start;
                
                for total_pos in 0..total_seq_len {
                    let k_offset = total_pos * n_kv_h * dqkv + kv_head * dqkv;
                    
                    let mut score = 0.0;
                    for i in 0..dqkv {
                        score += q_data[q_offset + i] * k_data[k_offset + i];
                    }
                    score *= scale;
                    
                    let att_index = kv_head * (n_groups * seq_len * total_seq_len)
                        + group * (seq_len * total_seq_len)
                        + seq_pos * total_seq_len
                        + total_pos;
                    att_scores_data[att_index] = score;
                }
            }
        }
    }
    }
    
    OP::masked_softmax(att_scores);
    
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();
    let att_scores_data =  unsafe {att_scores.data_mut()};
    let hidden_data =  unsafe {hidden_states.data_mut()};

    for kv_head in 0..n_kv_h {
        for group in 0..n_groups {
            let out_group_start = kv_head * n_groups * dqkv + group * dqkv;
            
            for seq_pos in 0..seq_len {
                let out_offset = seq_pos * n_kv_h * n_groups * dqkv + out_group_start;
                
                for i in 0..dqkv {
                    hidden_data[out_offset + i] = 0.0;
                }
                
                for total_pos in 0..total_seq_len {
                    let att_index = kv_head * (n_groups * seq_len * total_seq_len)
                        + group * (seq_len * total_seq_len)
                        + seq_pos * total_seq_len
                        + total_pos;
                    
                    let att_weight = att_scores_data[att_index];
                    let v_offset = total_pos * n_kv_h * dqkv + kv_head * dqkv;
                    
                    for i in 0..dqkv {
                        hidden_data[out_offset + i] += att_weight * v_data[v_offset + i];
                    }
                }
            }
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    let shape = residual.shape();
    assert!(shape == hidden_states.shape());

    // let _residual = unsafe { residual.data_mut() };
    // let _hidden_states = unsafe { hidden_states.data_mut() };
    // let _gate = unsafe { gate.data_mut() };
    // let _up = unsafe { up.data_mut() };
    // let _w_up = w_up.data();
    // let _w_down = w_down.data();
    // let _w_gate = w_gate.data();
    // let rms_w = rms_w.data();

    OP::rms_norm(hidden_states, residual, rms_w, eps);
    OP::matmul_transb(gate, 0., hidden_states, w_gate, 1.);
    OP::matmul_transb(up, 0., hidden_states, w_up, 1.);
    let mut act = Tensor::<f32>::new(up.data().to_vec(),up.shape());
    OP::swiglu(&mut act,gate);
    OP::matmul_transb(residual, 1., &act, w_down, 1.)
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}
