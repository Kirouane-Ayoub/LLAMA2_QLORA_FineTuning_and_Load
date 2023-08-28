from fastapi import FastAPI, HTTPException, Form, File, UploadFile, Body
import pinecone
from peft import LoraConfig, get_peft_model
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain

import torch

app = FastAPI()

@app.post("/model-inference")
async def model_inference(
    pinecone_api_key: str = Form(...),
    pinecone_environment: str = Form(...),
    fine_tuned_model_path: str = Form(...),
    text_input: str = Form(...),
):
    try:
        pinecone.init(
            api_key=pinecone_api_key,
            environment=pinecone_environment
        )
        # LOAD THE Embeddings FUNCTION
        embeddings_func = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        model_name = "georgesung/llama2_7b_chat_uncensored"
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
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True
        )
        # LOAD LORA CONFIG
        lora_config = LoraConfig.from_pretrained(fine_tuned_model_path)
        # CREATE OUR NEW MODEL
        new_model = get_peft_model(model, lora_config)
        # INITAILIZE text-generation pipeline
        pipe = pipeline(
            "text-generation",
            model=new_model,
            tokenizer=tokenizer,
            max_length=2048,
            temperature=0.5,
            top_p=0.95,
            repetition_penalty=1.15
        )
        # CREATE AN LLM LLMCHAIN PIPLINE 
        local_llm = HuggingFacePipeline(pipeline=pipe)

        # CREATE THE ConversationChain
        chat = ConversationChain(
            llm=local_llm,
            verbose=False
        )
        # CHANGE THE PROMPT
        chat.prompt.template = """
        ### HUMAN:
        {history}
        ### HUMAN: {input}
        ### RESPONSE:"""
        # MAKE THE PREDICTION
        result = chat.predict(input=text_input)
        return {"response": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
