import torch
#################### code changes ####################  # noqa F401
import intel_extension_for_pytorch as ipex
######################################################  # noqa F401
import argparse
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# args
parser = argparse.ArgumentParser("Generation script (static quantization path)", add_help=False)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16"],
    default="float32",
    help="choose the weight dtype and whether to enable auto mixed precision or not",
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument(
    "--prompt", default="What are we having for dinner?", type=str, help="input prompt"
)
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--calibration-samples", default=512, type=int, help="total number of calibration samples")
parser.add_argument("--int8-qconfig", nargs="?", default="./qconfig.json", help="static quantization factors summary files generated by calibration")
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k")
parser.add_argument("--alpha", default=0.5, type=float, help="alpha value for smoothquant")
args = parser.parse_args()
print(args)


# dtype
amp_enabled = True if args.dtype != "float32" and not calibration else False
amp_dtype = getattr(torch, args.dtype) if not calibration else torch.float32

# load model
model_id = "meta-llama/Llama-2-7b-hf"
config = AutoConfig.from_pretrained(
    model_id, torchscript=True, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=amp_dtype,
    config=config,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
model = model.eval()
model = model.to(memory_format=torch.channels_last)

num_beams = 1 if args.greedy else 4
beam_idx_tmp = torch.zeros(
    (2048, int(args.batch_size * num_beams)), dtype=torch.long
).contiguous()
global_past_key_value = [
    (
        torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
        torch.zeros(
            [
                1,
                model.config.num_attention_heads,
                1,
                int(
                    model.config.hidden_size
                    / model.config.num_attention_heads
                ),
            ]
        ).contiguous(),
        torch.zeros(
            [
                1,
                user_model.config.num_attention_heads,
                1,
                int(
                    model.config.hidden_size
                    / model.config.num_attention_heads
                ),
            ]
        ).contiguous(),
        beam_idx_tmp,
    )
    for i in range(model.config.num_hidden_layers)
]

# Intel(R) Extension for PyTorch*
#################### code changes ####################  # noqa F401
class Calibration:
    def __init__(self, dataset, tokenizer, batch_size=1, pad_val=1, pad_max=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max

        # tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def tokenize_function(self, examples):
        if "prompt" in examples:
            example = self.tokenizer(examples["prompt"])
        elif "text" in examples:
            example = self.tokenizer(examples["text"])
        elif "code" in examples:
            example = self.tokenizer(examples["code"])
        return example

    @torch.no_grad()
    def collate_batch(self, batch):
        position_ids_padded = []
        input_ids_padded = []
        last_ind = []
        attention_mask_padded = []
        for text in batch:
            input_ids = text["input_ids"]
            input_ids = (
                input_ids[: int(self.pad_max)]
                if len(input_ids) > int(self.pad_max)
                else input_ids
            )
            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids))
            position_ids = torch.arange(len(input_ids))
            input_ids_padded.append(input_ids)
            attention_mask_padded.append(attention_mask)
            position_ids_padded.append(position_ids)
        return (
            (
                torch.vstack(input_ids_padded),
                torch.vstack(attention_mask_padded),
                torch.vstack(position_ids_padded),
                tuple(global_past_key_value),
            ),
            torch.tensor(last_ind),
        )

calib_dataset = load_dataset(args.dataset, split="train")
calib_evaluator = Calibration(calib_dataset, tokenizer, args.batch_size)
calib_dataloader = DataLoader(
    calib_evaluator.dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=calib_evaluator.collate_batch,
)

qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=args.alpha)

if args.calibration:
    example_inputs = None
    for i, (
        (input_ids, attention_mask, position_ids, past_key_values),
        last_ind,
    ) in enumerate(calib_dataloader):
        example_inputs =
            (input_ids, attention_mask, position_ids, past_key_values)
        break
    from intel_extension_for_pytorch.quantization import prepare, convert
    model = ipex.optimize_transformers(
        model.eval(),
        dtype=amp_dtype,
        quantization_config=qconfig,
        inplace=True,
        deployment_mode=False,
    )
    prepared_model = prepare(
        model.eval(), qconfig, example_inputs=example_inputs
    )
    with torch.no_grad():
        for i, (
            (input_ids, attention_mask, position_ids, past_key_values),
            last_ind,
        ) in enumerate(calib_dataloader):
            if i == args.calibration_samples:
                break
            prepared_model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )

    prepared_model.save_qconf_summary(qconf_summary=args.int8_qconfig)
    print("calibration Done! Will exit and please launch model quantization and benchmark")
    exit(0)
else:
    model = ipex.optimize_transformers(
        model.eval(),
        dtype=amp_dtype,
        quantization_config=qconfig,
        qconfig_summary_file=args.int8_qconfig,
        inplace=True,
        deployment_mode=True,
    )
    print("model quantization - Done!")

######################################################  # noqa F401

# generate args
num_beams = 1 if args.greedy else 4
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=num_beams)

# input prompt
prompt = args.prompt
input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
print("---- Prompt size:", input_size)
prompt = [prompt] * args.batch_size

# inference
with torch.no_grad(), torch.inference_mode(), torch.cpu.amp.autocast(enabled=amp_enabled):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        **generate_kwargs
    )
    gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    input_tokens_lengths = [x.shape[0] for x in input_ids]
    output_tokens_lengths = [x.shape[0] for x in gen_ids]
    total_new_tokens = [
        o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)
    ]
    print(gen_text, total_new_tokens, flush=True)
