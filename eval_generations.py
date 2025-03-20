# %%
import json
import glob
import os
def parse_checkpoint(f):
    # Assuming format like "prefix_ckpt{idx}_suffix"
    base = f.split('ckpt')[0].strip('_')  # Get the prefix before 'ckpt'
    remaining = f.split('ckpt')[1]
    
    # Split the remaining part into idx and suffix
    idx = int(remaining.split('_')[0])  # Extract the number after 'ckpt'
    suffix = remaining.split('_')[1] if '_' in remaining else ''  # Get suffix if exists
    suffix = suffix.strip('.json')
    return base, idx, suffix

# glob all files in pub-med-eval
files = glob.glob("pub-med-eval/*_test.json")


# Load true decisions once
with open("pub-med-eval/true_decisions_y.json", "r") as f:
    true_decisions_y = json.load(f)

# Create a table of results
results = []
for filename in files:
    try:
        # Parse checkpoint info
        run, idx, suffix = parse_checkpoint(filename)
        # if suffix != 'test':
            # continue
        
        # Load predictions
        with open(filename, "r") as f:
            data = json.load(f)
        
        # Calculate accuracy
        decisions_yhat = data["decisions_yhat"]
        accuracy = sum([decisions_yhat[i] == true_decisions_y[i] for i in range(len(true_decisions_y))]) / len(true_decisions_y)
        
        # Add to results
        results.append({
            "filename": filename,
            "run": run,
            "idx": idx,
            # "suffix": suffix,
            "accuracy": accuracy
        })
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# base model
f = "pub-med-eval/base_model.json"
if os.path.exists(f):
    with open(f, "r") as f:
        data = json.load(f)
    accuracy = sum([data["decisions_yhat"][i] == true_decisions_y[i] for i in range(len(true_decisions_y))]) / len(true_decisions_y)
    results.append({
        "filename": f,
        "run": "base_model",
        "idx": 0,
        "accuracy": accuracy
    })


# Display results as a table
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('pub-med-eval/eval_generations_results.csv', index=False)


# %%
