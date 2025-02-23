mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;


use eframe::egui;
use std::sync::{Arc, Mutex};
use crate::model::Llama;
use std::thread;
use std::sync::mpsc;

use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Clone)]
enum Message {
    User(String),
    Assistant(String),
}

struct ChatApp {
    llama_chat: Arc<Mutex<Llama<f32>>>,
    tokenizer_chat: Arc<Tokenizer>,
    llama_story: Arc<Mutex<Llama<f32>>>,
    tokenizer_story: Arc<Tokenizer>,
    // kvcache: Option<kvcache::KVCache<f32>>,

    messages: Vec<Message>,
    input_text: String,
    input_story: String,
    output_text: String,
    output_story: String,
    generating: bool,
    receiver: Option<mpsc::Receiver<String>>,
    mode: i32,
    
    max_turns: i32,
    max_len: usize,
    top_p: f32,
    top_k: u32,
    temperature: f32,
}

impl ChatApp {
    fn new(llama_c: Llama<f32>, tokenizer_c: Tokenizer, llama_s: Llama<f32>, tokenizer_s: Tokenizer) -> Self {
        Self {
            llama_chat: Arc::new(Mutex::new(llama_c)),
            tokenizer_chat: Arc::new(tokenizer_c),

            llama_story: Arc::new(Mutex::new(llama_s)),
            tokenizer_story: Arc::new(tokenizer_s),

            messages: vec![],
            input_text: String::new(),
            input_story: format!("Once upon a time"),
            output_story: String::new(),
            output_text: String::new(),
            generating: false,
            receiver: None,
            mode: 0,

            max_turns: 3,
            max_len: 250,
            top_p: 0.8,
            top_k: 30,
            temperature: 1.,
        }
    }

    fn generate_story(&mut self){
        let llama = self.llama_story.clone();
        let tokenizer = self.tokenizer_story.clone();
        
        let input = self.input_story.clone();
        let binding = tokenizer.encode(input.clone(), true).unwrap();
        let input_ids = binding.get_ids();
        print!("\n{}\n", input.clone());
        let output_ids = llama.lock().unwrap().generate(
            input_ids,
            self.max_len,
            self.top_p,
            self.top_k,
            self.temperature,
        );
        self.output_story = format!("{}",tokenizer.decode(&output_ids, true).unwrap());
    }

    fn generate_chat(&mut self){
        let llama = self.llama_chat.clone();
        let tokenizer = self.tokenizer_chat.clone();

        let mut dialog_history = String::new();
        let mut cache = llama.lock().unwrap().new_cache();
        
    //     // 对话模板常量
    //     const SYSTEM_PROMPT: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    //     const USER_PREFIX: &str = "<|im_start|>user\n";
    //     const ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n";
        const END_MARKER: &str = "<|im_end|>\n";

    //     // dialog_history.push_str(SYSTEM_PROMPT);

        
    //     println!("Starting chat session (type 'exit' to end):");
    //     // llama.chat(
    //     //     5,     // 最大对话轮数
    //     //     50,    // 最大生成
    //     //     0.8,    // top_p
    //     //     1,     // top_k 
    //     //     1.,      // temperature
    //     // );
        // for turn in 0..self.max_turns {
            // 获取用户输入
            let mut user_input = String::new();
            // println!("\nUser (turn {}):", turn + 1);
            // std::io::stdin().read_line(&mut user_input).unwrap();
            // user_input = user_input.trim().to_string();
            user_input = self.input_text.clone();

            // if user_input.to_lowercase() == "exit" {
            //     break;
            // }

            // // 构建prompt
            // dialog_history.push_str(&format!("{}{}{}", 
            //     USER_PREFIX, 
            //     user_input, 
            //     END_MARKER
            // ));
            // dialog_history.push_str(ASSISTANT_PREFIX);

            let mut input = String::new();
            // input = format!("<|im_start|>{}\n{}<|im_end|>\n<|im_start|>assistant\n", "user", user_input);
            input = user_input;

            println!("\nInput ({}):", input);

            // 编码输入
            // let binding = tokenizer.encode(dialog_history.as_str(), true).unwrap();
            let binding = tokenizer.encode(input, true).unwrap();
            let input_ids = binding.get_ids();
            
            println!("Assistant:");
            
            // 生成回复
            let result = llama.lock().unwrap().generate_cache(
                input_ids,
                self.max_len,  // 每轮最大生成长度
                self.top_p,
                self.top_k,
                self.temperature,
                &mut cache
            );


            // let output_iter = llama.generate_iter(
            //     input_ids,
            //     25,
            //     top_p,
            //     top_k,
            //     temperature,
            //     &mut cache
            // );
            // // 使用迭代器输出
            // for output_id in output_iter{
            //     // 适当添加空格然后输出
            //     let word = tokenizer.decode(&vec![output_id], true).unwrap();
            //     let word = if word.chars().all(|c| c.is_alphabetic()){
            //         " ".to_string() + &word
            //     }else{
            //         word
            //     };
            //     print!("{}", word);
            //     // std::io::stdout().flush().unwrap();
            // }
            
        
            
            // 解码并更新历史
            let response = tokenizer.decode(&result, true).unwrap();
            let clean_response = response.replace(END_MARKER, "").trim().to_string();
            // println!("Assistant: {}", clean_response);
            println!("Assistant_result: {}", clean_response);
            
            // 更新对话历史
            // dialog_history.push_str(&format!("{}{}", clean_response, END_MARKER));
            
            // // 缓存管理（限制历史长度）
            // if dialog_history.len() > 4000 {
            //     dialog_history.drain(0..2000);
            //     cache = llama.new_cache(); // 历史过长时重置缓存
            // }

            self.output_story = clean_response;
        // }

    }

    // fn generate_response(&mut self, input: String) {
    //     let llama = self.llama.clone();
    //     let tokenizer = self.tokenizer.clone();
    //     let mode = self.mode.clone();
    //     let (sender, receiver) = mpsc::channel();

    //     self.generating = true;
    //     self.receiver = Some(receiver);

    //     thread::spawn(move || {
    //         let mut prompt = String::new();
            
    //         // 根据模式构建不同的prompt
    //         // match mode {
    //         //     AppMode::Chat => {
    //         //         const SYSTEM_PROMPT: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    //         //         const USER_PREFIX: &str = "<|im_start|>user\n";
    //         //         const ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n";
    //         //         const END_MARKER: &str = "<|im_end|>\n";
                    
    //         //         prompt.push_str(SYSTEM_PROMPT);
    //         //         prompt.push_str(&format!("{}{}{}", USER_PREFIX, input, END_MARKER));
    //         //         prompt.push_str(ASSISTANT_PREFIX);
    //         //     }
    //         //     AppMode::Story => {
    //         //         prompt.push_str(&format!("Continue this story: {}\n", input));
    //         //     }
    //         // }

    //         let binding = tokenizer.encode(prompt.as_str(), true).unwrap();
    //         let input_ids = binding.get_ids();
            
    //         let mut cache = llama.lock().unwrap().new_cache();
    //         let output_ids = llama.lock().unwrap().generate_cache(
    //             input_ids,
    //             250,
    //             0.8,
    //             30,
    //             1.0,
    //             &mut cache
    //         );
            
    //         let response = tokenizer.decode(&output_ids, true).unwrap();
    //         sender.send(response.trim().to_string()).unwrap();
    //     });
    // }
    
}

impl eframe::App for ChatApp {
    
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 模式切换工具栏
        egui::TopBottomPanel::top("mode_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.mode, 0, "Chat Mode");
                ui.radio_value(&mut self.mode, 1, "Story Mode");
            });
        });
        //设置主题
        ctx.set_visuals(egui::Visuals::dark());

        egui::CentralPanel::default().show(ctx, |ui| {
            // ui.label(format!("P{}",self.mode))
            if self.mode == 0 {
                ui.heading("Chat");

                ui.label("User:");
                ui.text_edit_singleline(&mut self.input_text);
            
                if ui.button("Send").clicked() {
                    self.generate_chat();
                }

                ui.label("Assistant:");
                ui.label(&self.output_story);
            }
            
            if self.mode ==1 {
                ui.heading("Story");

                // self.input_story = format!("Once upon a time");
                ui.label("User:");
                ui.text_edit_singleline(&mut self.input_story);

                if ui.button("Generate").clicked() {
                    self.generate_story();
                }

                ui.label("Story:");
                ui.label(&self.output_story);

            }
        });
        

        // // 接收生成结果
        // if let Some(receiver) = &self.receiver {
        //     if let Ok(response) = receiver.try_recv() {
        //         self.messages.push(Message::Assistant(response));
        //         self.generating = false;
        //         self.receiver = None;
        //         ctx.request_repaint();
        //     }
        // }

    }
}

// 启动UI
pub fn run_ui() {
    
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama_chat = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer_chat = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama_story = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer_story = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
 
    let options = eframe::NativeOptions::default();
    let _ = eframe::run_native(
        "Llama Chat",
        options,
        Box::new(|_| Box::new(ChatApp::new(llama_chat, tokenizer_chat, llama_story, tokenizer_story))),
    );
}


    // fn story_type(){
    //     let project_dir = env!("CARGO_MANIFEST_DIR");
    //     let model_dir = PathBuf::from(project_dir).join("models").join("story");
    //     // print!("{}", model_dir.display());
    //     let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    //     let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    
    //     let input = "Once upon a time";
    //     let binding = tokenizer.encode(input, true).unwrap();
    //     let input_ids = binding.get_ids();
    //     print!("\n{}\n", input);
    //     let output_ids = llama.generate(
    //         input_ids,
    //         50,
    //         0.8,
    //         1,  //30原来，设置1限制输出一样方便调试
    //         1.,
    //     );
    //     println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    
    // }
    

    // fn chat_type(){
        
    //     let project_dir = env!("CARGO_MANIFEST_DIR");
    //     let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    //     let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    //     let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();


    //     let max_turns = 3;
    //     let max_len = 50;
    //     let top_p = 0.8;
    //     let top_k = 30;
    //     let temperature = 1.;

    //     let mut dialog_history = String::new();
    //     let mut cache = llama.new_cache();
        
    //     // 对话模板常量
    //     const SYSTEM_PROMPT: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    //     const USER_PREFIX: &str = "<|im_start|>user\n";
    //     const ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n";
    //     const END_MARKER: &str = "<|im_end|>\n";

    //     // dialog_history.push_str(SYSTEM_PROMPT);

        
    //     println!("Starting chat session (type 'exit' to end):");
    //     // llama.chat(
    //     //     5,     // 最大对话轮数
    //     //     50,    // 最大生成
    //     //     0.8,    // top_p
    //     //     1,     // top_k 
    //     //     1.,      // temperature
    //     // );
    //     for turn in 0..max_turns {
    //         // 获取用户输入
    //         let mut user_input = String::new();
    //         println!("\nUser (turn {}):", turn + 1);
    //         std::io::stdin().read_line(&mut user_input).unwrap();
    //         user_input = user_input.trim().to_string();

    //         if user_input.to_lowercase() == "exit" {
    //             break;
    //         }

    //         // // 构建prompt
    //         // dialog_history.push_str(&format!("{}{}{}", 
    //         //     USER_PREFIX, 
    //         //     user_input, 
    //         //     END_MARKER
    //         // ));
    //         // dialog_history.push_str(ASSISTANT_PREFIX);

    //         let mut input = String::new();
    //         // input = format!("<|im_start|>{}\n{}<|im_end|>\n<|im_start|>assistant\n", "user", user_input);
    //         input = user_input;

    //         println!("\nInput ({}):", input);

    //         // 编码输入
    //         // let binding = tokenizer.encode(dialog_history.as_str(), true).unwrap();
    //         let binding = tokenizer.encode(input, true).unwrap();
    //         let input_ids = binding.get_ids();
            
    //         println!("Assistant:");
            
    //         // 生成回复
    //         let result = llama.generate_cache(
    //             input_ids,
    //             250,  // 每轮最大生成长度
    //             top_p,
    //             top_k,
    //             temperature,
    //             &mut cache
    //         );


    //         // let output_iter = llama.generate_iter(
    //         //     input_ids,
    //         //     25,
    //         //     top_p,
    //         //     top_k,
    //         //     temperature,
    //         //     &mut cache
    //         // );
    //         // // 使用迭代器输出
    //         // for output_id in output_iter{
    //         //     // 适当添加空格然后输出
    //         //     let word = tokenizer.decode(&vec![output_id], true).unwrap();
    //         //     let word = if word.chars().all(|c| c.is_alphabetic()){
    //         //         " ".to_string() + &word
    //         //     }else{
    //         //         word
    //         //     };
    //         //     print!("{}", word);
    //         //     // std::io::stdout().flush().unwrap();
    //         // }
            
        
            
    //         // 解码并更新历史
    //         let response = tokenizer.decode(&result, true).unwrap();
    //         let clean_response = response.replace(END_MARKER, "").trim().to_string();
    //         // println!("Assistant: {}", clean_response);
    //         println!("Assistant_result: {}", clean_response);
            
    //         // 更新对话历史
    //         // dialog_history.push_str(&format!("{}{}", clean_response, END_MARKER));
            
    //         // // 缓存管理（限制历史长度）
    //         // if dialog_history.len() > 4000 {
    //         //     dialog_history.drain(0..2000);
    //         //     cache = llama.new_cache(); // 历史过长时重置缓存
    //         // }
    //     }
    // }

fn main() {

    // 启动UI
    run_ui();

    // let project_dir = env!("CARGO_MANIFEST_DIR");
    // let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    // let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    // let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    // let mut user_input = String::new();
    // println!("\n Choose your function:\n 1: chat;\n 2: story;\n");
    // std::io::stdin().read_line(&mut user_input).unwrap();
    // let user_input = user_input.trim();
    // match user_input{
    //     "1" => chat_type(),
    //     "2" => story_type(),
    //     _ => println!("Invalid input!\n"),
    // }


}
