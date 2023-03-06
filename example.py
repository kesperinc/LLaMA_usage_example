
import llama

MODEL = 'decapoda-research/llama-7b-hf'

tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL)
model = llama.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True).half()
model.to('cuda')

prompt = """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:"""

batch = tokenizer(prompt, return_tensors = "pt", add_special_tokens = False)
print(tokenizer.decode(model.generate(batch["input_ids"].cuda(), max_length=100)[0]))
