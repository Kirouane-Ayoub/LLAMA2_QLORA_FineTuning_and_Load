from peft import LoraConfig, get_peft_model
from transformers import LlamaTokenizer, LlamaForCausalLM , BitsAndBytesConfig , pipeline
from langchain.llms import HuggingFacePipeline

fine_tuned_model_path = "" # THE FINETUNED ADAPTER FILES
model_name = "georgesung/llama2_7b_chat_uncensored" # THE BASE MODEL
def model_inference(fine_tuned_model_path , model_name) : 
  # LOAD THE SAME BNB CONFIG (THE SAME BNB CONFIG FROM TRAINING)
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
  )
  # LOAD OUR Tokenizer
  tokenizer = LlamaTokenizer.from_pretrained(model_name)
  tokenizer.pad_token = tokenizer.eos_token
  # LOAD THE MODEL
  model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True)
  # LOAD LORA CONFIG
  lora_config = LoraConfig.from_pretrained(fine_tuned_model_path)
  # CREATE OUR NEW MODEL
  new_model = get_peft_model(model, lora_config)
  # CREATE A text-generation PIPELINE 
  pipe = pipeline(
      "text-generation",
      model=new_model,
      tokenizer=tokenizer,
      max_length=2048,
      temperature=0.5,
      top_p=0.95,
      repetition_penalty=1.15
  )
  # CREATE LANGCHAIN HuggingFacePipeline
  local_llm = HuggingFacePipeline(pipeline=pipe)
  # RETURN THE LOCAL LLM Pipeline
  return local_llm
