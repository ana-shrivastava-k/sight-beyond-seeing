from unsloth import FastLanguageModel
import torch
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that
provides further context.
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = ""
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

def train(run_config):
    run = run_config["run"]

    global EOS_TOKEN
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/tinyllama-bnb-4bit", #for 16bit loading "unsloth/tinyllama"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0, # Currently only supports dropout = 0
        bias = "none",    # Currently only supports bias = "none"
        use_gradient_checkpointing = True, # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    EOS_TOKEN = tokenizer.eos_token

    from datasets import load_dataset
    dataset = load_dataset("anashrivastava/tl_rephrase", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    dataset_dict = dataset.train_test_split(test_size=0.10)

    import wandb
    wandb.login()

    import os

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from transformers.utils import logging
    import wandb

    logging.set_verbosity_info()
    project_name = "tiny-llama-rephrase" 
    entity = "wandb"
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    wandb.init(project=project_name, name = "tiny-llama-rephrase-" + str(run))

    args = TrainingArguments(
            per_device_train_batch_size = run_config["batch_size"],
            per_device_eval_batch_size=4,
            gradient_accumulation_steps = run_config["gradient_acc_steps"],
            evaluation_strategy="no",
            warmup_ratio = 0.1,
            num_train_epochs = run_config["epoch"],
            learning_rate = 2e-5,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            optim = "adamw_8bit",
            weight_decay = 0.1,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to="wandb",  # enable logging to W&B
            logging_steps=1,  # how often to log to W&B
            logging_strategy = 'steps',
            save_total_limit=2,
        )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Packs short sequences together to save time!
        args = args,
    )

    trainer_stats = trainer.train()
    trainer_stats = trainer.evaluate(dataset_dict["test"][:109])
    print(trainer_stats)
    wandb.finish()

    model_name="tinyllama_rephrase_" + str(run)
    model.save_pretrained(model_name) # Local saving
    model.save_pretrained_gguf(model_name + "_4bit", tokenizer, quantization_method = "q4_k_m")


def main():
    runs = [{
    "run": 1,
    "epoch": 1,
    "batch_size": 4,
    "gradient_acc_steps": 4
    },
    {
    "run": 2,
    "epoch": 1,
    "batch_size": 2,
    "gradient_acc_steps": 4
    },
    {
    "run": 3,
    "epoch": 1,
    "batch_size": 1,
    "gradient_acc_steps": 4
    },
    {
    "run": 4,
    "epoch": 2,
    "batch_size": 4,
    "gradient_acc_steps": 4
    },
    {
    "run": 5,
    "epoch": 2,
    "batch_size": 2,
    "gradient_acc_steps": 4
    },
    {
    "run": 6,
    "epoch": 2,
    "batch_size": 1,
    "gradient_acc_steps": 4
    },
    {
    "run": 7,
    "epoch": 5,
    "batch_size": 4,
    "gradient_acc_steps": 4
    },
    {
    "run": 8,
    "epoch": 5,
    "batch_size": 2,
    "gradient_acc_steps": 4
    },
    {
    "run": 9,
    "epoch": 5,
    "batch_size": 1,
    "gradient_acc_steps": 4
    }]
    for run in runs:
        train(run)

if __name__ == "__main__":
    main()
