# LLAMA2 QLORA FineTuning + Loading

## LLAMA2 : 
LLAMA2 is a large language model (LLM) developed by Meta AI. It is a successor to Meta's Llama 1 language model, released in the first quarter of 2023. LLAMA2 has been trained on a massive dataset of publicly available internet data, enjoying the advantage of a dataset both more recent and more diverse than that used to train Llama 1.

## Fine-tuning : 
Fine-tuning is a technique in machine learning where a pre-trained model is further trained on a smaller dataset to improve its performance on a specific task. The pre-trained model is typically trained on a large dataset of general data, while the fine-tuning dataset is specific to the task that the model is being fine-tuned for.

## Q-LORA : 
QLORA is a new approach to fine-tuning large language models (LLMs) that reduces memory usage without sacrificing performance. It was introduced by Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer in their paper "QLoRA: Efficient Finetuning of Quantized LLMs".QLORA works by backpropagating gradients through a frozen, 4-bit quantized pre-trained LLM into Low-Rank Adapters (LoRA). LoRA are additional layers that are added to the LLM to improve its performance on a specific task.
