from transformers import LlamaTokenizer, LlamaForCausalLM , BitsAndBytesConfig
from peft import prepare_model_for_kbit_training , LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
import transformers
model_name= ""          # The model that you want to train from the Hugging Face hub
num_train_epochs = 100  # THE NUMBER OF THE TRAINING EPOCHS .
char_id = ""            # The output directory where the model predictions and checkpoints will be written 
dataset_path = ""       # PUT HERE YOUR DATASET CSV OR JSON DATASET 

def run_train(model_name , num_train_epochs , char_id , dataset_path) :
  # Load the tokenizer and the model
  tokenizer = LlamaTokenizer.from_pretrained(model_name)
  tokenizer.pad_token = tokenizer.eos_token
  # quantization_config  
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16)
  # LOAD THE MODEL USING bnb_config
  model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True)
  model.gradient_checkpointing_enable()
  model = prepare_model_for_kbit_training(model)
  # INITIALIZE LORA CONFIG 
  config = LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=["q_proj","v_proj"],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM"
  )
  # LORA + QUANTIZATION
  model = get_peft_model(model, config)
  # LOAD OUR CUSTOM DATASET
  data = load_dataset("csv", data_files=dataset_path)
  # prompt_template = "### Instruction: {prompt}\n### Response:"
  train_dataset = data['train']
  # Tokenize the datasets
  train_encodings = tokenizer(train_dataset['input_text'], truncation=True, padding=True, max_length=256, return_tensors='pt')
  # CREATE CUSTOM TextDataset CLASS (FROM Dataset SUPER)
  class TextDataset(Dataset):
      def __init__(self, encodings):
          self.encodings = encodings

      def __getitem__(self, idx):
          item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
          item["labels"] = item["input_ids"].clone()
          return item

      def __len__(self):
          return len(self.encodings["input_ids"])
  # Create a train_dataset object
  train_dataset = TextDataset(train_encodings)
  # CREATE A Trainer OBJECT
  trainer = transformers.Trainer(
      model=model,
      train_dataset=train_dataset,
      # eval_dataset=val_dataset,
      args=transformers.TrainingArguments(
          num_train_epochs=num_train_epochs,
          per_device_train_batch_size=16,
          gradient_accumulation_steps=4,
          warmup_ratio=0.05,
          # max_steps=100,
          learning_rate=2e-4,
          fp16=True,
          logging_steps=1,
          output_dir=char_id,
          optim="paged_adamw_8bit",
          lr_scheduler_type='cosine',
      ),
      data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False))
  model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
  # START TRAINING 
  trainer.train()
  # SAVE THE MODEL 
  model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
  model_to_save.save_pretrained(char_id)
  # Clear the CUDA cache
  torch.cuda.empty_cache()
  return "the model has been finetuned and saved on" , char_id
