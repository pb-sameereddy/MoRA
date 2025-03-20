# %%
import json

# %%
with open("pub-med-eval/lora_rank128_lr1e-4_witheval_ckpt400_test.json", "r") as f:
    data = json.load(f)

generations_str = data["generations_str"]
# decisions_y = data["decisions_y"] # BROKEN
with open("pub-med-eval/true_decisions_y.json", "r") as f:
    decisions_y = json.load(f)
decisions_yhat = data["decisions_yhat"]

# %%
accuracy = sum([decisions_yhat[i] == decisions_y[i] for i in range(len(decisions_y))]) / len(decisions_y)
print(f"Accuracy: {accuracy}")

# %%
