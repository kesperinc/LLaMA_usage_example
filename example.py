
import llama

MODEL = 'decapoda-research/llama-7b-hf'
REVISION = '84fd0de2f666324fe13da5642b047be4d55b5982'

tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL, revision=REVISION)
model = llama.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True, revision=REVISION).half()
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
