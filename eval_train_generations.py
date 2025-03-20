# %%
import glob
import json
from eval_utils import extract_decision, load_data
from tqdm import tqdm

# %%
data_path = 'qiaojin/PubMedQA'
# subset = 'pqa_labeled'
subset = 'pqa_artificial'
data_module, tokenizer = load_data(
    data_path=data_path,
    base_model='meta-llama/Llama-3.2-1B-Instruct',
    subset=subset,
    data_length=None, # for testing
)
data = data_module['train_dataset'] 


# %%
# f = '/root/MoRA/pub-med-eval/lora_rank128_lr1e-4_witheval_ckpt400_train.json'
files = glob.glob("pub-med-eval/*_train.json")
print(files)

def eval_generations(f):

    ckpt_idx = int(f.split('ckpt')[1].split('_')[0])
    with open(f, 'r') as f:
        generations_str = json.load(f)['generations_str']


    # for i, g in enumerate(generations_str):
    def extract_instruction(generation):
        """
        Extract the instruction part from a generation string.
        
        Args:
            generation: The generated text string
            
        Returns:
            str: The extracted instruction
        """
        return generation.split('Instruction:')[1].split('### Input:')[0].strip()

    instructions = [extract_instruction(g) for g in generations_str]
    matching_indices = []
    for i, instruction in tqdm(enumerate(instructions), total=len(instructions)):
        found = False
        for j, sample in enumerate(data):
            if instruction in sample['input_ids']:
                matching_indices.append(j)
                found = True
                break

        if not found:
            print(f'No matching index found for instruction: {instruction}')
            matching_indices.append(None)
        
    score = 0
    for i in range(len(matching_indices)):
        if matching_indices[i] is None:
            continue
        ground_truth = extract_decision(data[matching_indices[i]]['labels'])
        prediction = extract_decision(generations_str[i])

        if ground_truth == prediction:
            score += 1
        else:
            print(ground_truth, prediction)
            print('True Response:\n', data[matching_indices[i]]['labels'])
            print('Predicted Response:\n', generations_str[i].split('### Response:')[1])
            print('-'*100)

    return ckpt_idx, score / len(matching_indices)

# %%
import pandas as pd
results = [eval_generations(f) for f in files]

# %%
results_df = pd.DataFrame(results, columns=['ckpt_idx', 'accuracy'])
results_df.to_csv('pub-med-eval/train_generations_results.csv', index=False)
# %%