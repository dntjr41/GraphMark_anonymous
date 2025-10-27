import os
import tqdm
import torch
import json
import argparse
from models import KGWatermarker

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=["opengen", "c4"], default="c4")
parser.add_argument('--output_path', type=str, default="/home/wooseok/KG_Mark/outputs/c4/llama-3-8b-inst_GraphMark_30.jsonl")
parser.add_argument('--llm', type=str, choices=["gpt-4", "llama-3-8b", "llama-3-8b-inst", "mistral-7b-inst", "Qwen2_5-7b_inst"], default="llama-3-8b-inst")
parser.add_argument('--ratio', type=float, default=0.30)
parser.add_argument('--pruning_ratio', type=float, default=0.3, help="Adaptive pruning ratio (default: 0.3)")
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--n', type=int, default=200, help="Number of examples to process (default: all)")
args = parser.parse_args()
print(args)

# Set CUDA_VISIBLE_DEVICES to force using only the specified GPU
if args.device_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    print(f"Set CUDA_VISIBLE_DEVICES to {args.device_id}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name()}")

if args.dataset == "opengen":
    input_path = "data/opengen_500.jsonl"
elif args.dataset == "c4":
    input_path = "data/c4.json"
else:
    raise NotImplementedError

output_path = args.output_path

watermarker = KGWatermarker(args.llm,
                            ratio=args.ratio,
                            device_id=args.device_id)

if args.n:
    input_data = [json.loads(line) for line in open(input_path, 'r')][:args.n]
else:
    input_data = [json.loads(line) for line in open(input_path, 'r')]
print(f"Loaded {len(input_data)} examples for {args.dataset}.")

def append_to_output_file(output_path, generation_record):
    with open(output_path, 'a') as fout:
        fout.write(json.dumps(generation_record, ensure_ascii=False) + "\n")

for idx, dd in tqdm.tqdm(enumerate(input_data)):
    if args.dataset == "opengen":
        prefix = dd["prefix"]
        target = ", ".join(dd["targets"])
    elif args.dataset == "c4":
        prefix = dd["prompt"]
        target = dd["natural_text"]
    else:
        raise NotImplementedError
    
    result = watermarker.insert_watermark(prefix, target, enable_adaptive_pruning=True, pruning_ratio=args.pruning_ratio)
    append_to_output_file(output_path, result)