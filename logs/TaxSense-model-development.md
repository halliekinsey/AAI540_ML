python3 train-TaxSense.py 
Using GPU: 0
Data prepared: prompt and target columns added.

--- Sample 1 ---
Sample input to the model:
You are a helpful tax advisor and legal expert. Use the provided context to answer the user's query in a clear and concise manner.

User Query: How can I reduce my tax bill?

Related Context:
System at IRS.gov/SAMS.
For more information, go to IRS.gov/Advocate.
How To Make a Contribution To
Reduce Debt Held by the Public
There are two ways to make a contribution to reduce the
debt held by the public.
• At Pay.gov, contribute online by credit card, debit card,
PayPal, checking account, or savings account.
• Write a check payable to “Bureau of the Fiscal Service.”
In the memo section, notate that it is a gift to reduce the
debt held by the public.
Mail the check to:
Attn: Dept G
WV 26106-2188. Or you can enclose
the check with your income tax return
when you file. In the memo section of
the check, make a note that it is a gift to
reduce the debt held by the public. Don’t
add your gift to any tax you may owe.
See the instructions for line 37 for de-
tails on how to pay any tax you owe. For
information on how to make this type of
gift online, go to TreasuryDirect.gov/
Help-Center/Public-Debt-FAQs/
#DebtFinance and click on “How do
you make a contribution to reduce the

Note: The above information is extracted from relevant forms or online sources. Use it to formulate your response.

Sample output the model is being trained to generate:
To lower your tax liability, consider maximizing deductions and credits, contributing to retirement accounts, and utilizing tax-efficient investments. Consult a tax professional for personalized strategies.

--- Sample 2 ---
Sample input to the model:
You are a helpful tax advisor and legal expert. Use the provided context to answer the user's query in a clear and concise manner.

User Query: What deductions am I eligible for?

Related Context:
Standard Deduction (Group I Only)
If you do not itemize your deductions, you can take the 2025
standard deduction listed below for your filing status.
Filing Status
Standard
Deduction
Married filing jointly or
Qualifying surviving spouse
. . . . . . . . . . . . . .
$30,000\*
Head of household . . . . . . . . . . . . . . . . . . . .
$22,500\*
Single or Married filing
separately
. . . . . . . . . . . . . . . . . . . . . . . . .
$15,000\*
33
Standard Deduction Worksheet for Dependents—Line 12
Keep for Your Records
Use this worksheet only if someone can claim you, or your spouse if filing jointly, as a dependent.
1.
Check if:
You were born before January 2, 1960.
You are blind.
Spouse was born before January 2, 1960.
Spouse is blind.
Total number of boxes
checked
. . . . . . . . . . . . . . . . . .
1.
2.
Is your earned income\* more than $850?
Yes.
Add $450 to your earned income. Enter the total.

Note: The above information is extracted from relevant forms or online sources. Use it to formulate your response.

Sample output the model is being trained to generate:
Eligibility for deductions varies based on individual circumstances. Common deductions include mortgage interest, state and local taxes, charitable contributions, and medical expenses exceeding a certain threshold.
Train and evaluation datasets created.
2025-03-01 20:10:13.924923: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-03-01 20:10:15.007397: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:15<00:00,  3.88s/it]
Base model loaded.
Gradient checkpointing disabled.
Model is on GPU 0.

=== Token Count Statistics for Training Prompts ===
count    120.000000
mean     337.000000
std       51.367845
min      264.000000
25%      304.750000
50%      325.000000
75%      346.500000
max      552.000000
Name: token_count, dtype: float64
==============================================

Due to CUDA memory limits, we can only go up to 512 tokens. We will truncate 4 samples.
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 96/96 [00:00<00:00, 2298.67 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 1589.60 examples/s]
Datasets tokenized.
comet_ml is installed but the Comet API Key is not configured. Please set the `COMET_API_KEY` environment variable to enable Comet logging. Check out the documentation for other ways of configuring it: https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#set-the-api-key
comet_ml is installed but the Comet API Key is not configured. Please set the `COMET_API_KEY` environment variable to enable Comet logging. Check out the documentation for other ways of configuring it: https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#set-the-api-key
Converting train dataset to ChatML: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 96/96 [00:00<00:00, 5827.87 examples/s]
Applying chat template to train dataset: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 96/96 [00:00<00:00, 5975.41 examples/s]
Truncating train dataset: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 96/96 [00:00<00:00, 1463.68 examples/s]
Converting eval dataset to ChatML: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 3603.74 examples/s]
Applying chat template to eval dataset: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 3443.25 examples/s]
Truncating eval dataset: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 1294.26 examples/s]
Using auto half precision backend
No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
Currently training with a batch size of: 4
The following columns in the training set don't have a corresponding argument in `PeftModelForCausalLM.forward` and have been ignored: Answer, target, Score2, Question, Context1, Context2, prompt, ID, Score1, Source. If Answer, target, Score2, Question, Context1, Context2, prompt, ID, Score1, Source are not expected by `PeftModelForCausalLM.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 96
  Num Epochs = 1
  Instantaneous batch size per device = 4
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 4
  Total optimization steps = 6
  Number of trainable parameters = 289,837,056
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [03:21<00:00, 33.72s/it]Saving model checkpoint to ./model_directory/models--zainnobody--TaxSense/checkpoint-6
loading configuration file ./model_directory/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7/config.json
Model config DeepseekV2Config {
  "architectures": [
    "DeepseekV2ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_deepseek.DeepseekV2Config",
    "AutoModel": "modeling_deepseek.DeepseekV2Model",
    "AutoModelForCausalLM": "modeling_deepseek.DeepseekV2ForCausalLM"
  },
  "aux_loss_alpha": 0.001,
  "bos_token_id": 100000,
  "eos_token_id": 100001,
  "ep_size": 1,
  "first_k_dense_replace": 1,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 10944,
  "kv_lora_rank": 512,
  "max_position_embeddings": 163840,
  "model_type": "deepseek_v2",
  "moe_intermediate_size": 1408,
  "moe_layer_freq": 1,
  "n_group": 1,
  "n_routed_experts": 64,
  "n_shared_experts": 2,
  "norm_topk_prob": false,
  "num_attention_heads": 16,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 27,
  "num_key_value_heads": 16,
  "pretraining_tp": 1,
  "q_lora_rank": null,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 0.707,
    "mscale_all_dim": 0.707,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 1.0,
  "scoring_func": "softmax",
  "seq_aux": true,
  "tie_word_embeddings": false,
  "topk_group": 1,
  "topk_method": "greedy",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.49.0",
  "use_cache": true,
  "v_head_dim": 128,
  "vocab_size": 102400
}

tokenizer config file saved in ./model_directory/models--zainnobody--TaxSense/checkpoint-6/tokenizer_config.json
Special tokens file saved in ./model_directory/models--zainnobody--TaxSense/checkpoint-6/special_tokens_map.json
Saving model checkpoint to ./model_directory/models--zainnobody--TaxSense/checkpoint-6
loading configuration file ./model_directory/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7/config.json
Model config DeepseekV2Config {
  "architectures": [
    "DeepseekV2ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_deepseek.DeepseekV2Config",
    "AutoModel": "modeling_deepseek.DeepseekV2Model",
    "AutoModelForCausalLM": "modeling_deepseek.DeepseekV2ForCausalLM"
  },
  "aux_loss_alpha": 0.001,
  "bos_token_id": 100000,
  "eos_token_id": 100001,
  "ep_size": 1,
  "first_k_dense_replace": 1,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 10944,
  "kv_lora_rank": 512,
  "max_position_embeddings": 163840,
  "model_type": "deepseek_v2",
  "moe_intermediate_size": 1408,
  "moe_layer_freq": 1,
  "n_group": 1,
  "n_routed_experts": 64,
  "n_shared_experts": 2,
  "norm_topk_prob": false,
  "num_attention_heads": 16,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 27,
  "num_key_value_heads": 16,
  "pretraining_tp": 1,
  "q_lora_rank": null,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 0.707,
    "mscale_all_dim": 0.707,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 1.0,
  "scoring_func": "softmax",
  "seq_aux": true,
  "tie_word_embeddings": false,
  "topk_group": 1,
  "topk_method": "greedy",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.49.0",
  "use_cache": true,
  "v_head_dim": 128,
  "vocab_size": 102400
}

tokenizer config file saved in ./model_directory/models--zainnobody--TaxSense/checkpoint-6/tokenizer_config.json
Special tokens file saved in ./model_directory/models--zainnobody--TaxSense/checkpoint-6/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


{'train_runtime': 228.0153, 'train_samples_per_second': 0.421, 'train_steps_per_second': 0.026, 'train_loss': 2.0226850509643555, 'mean_token_accuracy': 0.5813325420022011, 'epoch': 1.0}    
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [03:47<00:00, 37.98s/it]
Configuration saved in ./model_directory/models--zainnobody--TaxSense/config.json
Configuration saved in ./model_directory/models--zainnobody--TaxSense/generation_config.json
The model is bigger than the maximum size per checkpoint (5GB) and is going to be split in 3 checkpoint shards. You can find where each parameters has been saved in the index located at ./model_directory/models--zainnobody--TaxSense/model.safetensors.index.json.
TaxSense model saved to: ./model_directory/models--zainnobody--TaxSense/
Copied: tokenizer.json -> ./model_directory/models--zainnobody--TaxSense/
Copied: tokenizer_config.json -> ./model_directory/models--zainnobody--TaxSense/
Tokenizer file transfer completed.