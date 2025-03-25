# %%
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train").select(range(100))

# %%

def format_context(example):
    contexts = example['context']['contexts']
    labels = example['context']['labels']
    text = [f'{label}: {context}' for label, context in zip(labels, contexts)]
    text = '\n'.join(text)
    text = f'This is a summary of a medical paper:\n{text}'
    return dict(text=text)

# sample_dataset = dataset.select(range(10))
dataset = dataset.map(format_context, num_proc=10)
dataset = dataset.select_columns(['text', 'pubid'])
df = dataset.to_pandas()
df['text'] = df['text'].astype(str)
df = df.drop_duplicates(subset=['text'])
dataset = Dataset.from_pandas(df)

# %%
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", model_max_length=1024)
tokenizer.pad_token_id = (
    # NOTE: set this to eos token, set to unk(0) while make output nan
    2  # unk. we want this to be different from the eos token
)
dataset = dataset.map(lambda x: tokenizer(x['text']), num_proc=10)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# %%
# train val split

# %%
dataset = dataset.train_test_split(test_size=0.02)
# save dataset
dataset.save_to_disk('datasets/hf_pub_med_cpt')
# %%
