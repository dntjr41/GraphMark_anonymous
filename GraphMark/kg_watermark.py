import os
import tqdm
import torch
import json
import argparse
from models import KGWatermarker

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=["opengen", "c4"], default="opengen")
parser.add_argument('--output_path', type=str, default="/home/wooseok/KG_Mark/outputs/opengen/llama-3-8b_GraphMark_60.jsonl")
parser.add_argument('--llm', type=str, choices=["gpt-4", "llama-3-8b", "llama-3-8b-inst", "mistral-7b-inst", "Qwen2_5-7b_inst"], default="llama-3-8b")
parser.add_argument('--ratio', type=float, default=0.6)
parser.add_argument('--pruning_ratio', type=float, default=0.2, help="Adaptive pruning ratio (default: 0.3)")
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--n', type=int, default=500, help="Number of examples to process (default: all)")
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
                            topk=5,
                            device_id=args.device_id)

# Check existing output file to determine where to start
start_index = 0
existing_count = 0
if os.path.exists(output_path):
    try:
        with open(output_path, 'r') as f:
            existing_lines = f.readlines()
            existing_count = len(existing_lines)
            start_index = existing_count
            print(f"Found existing output file with {existing_count} processed examples")
    except Exception as e:
        print(f"Warning: Could not read existing output file: {e}")
        existing_count = 0
        start_index = 0

# Calculate how many more examples to process to reach total of 500
target_total = 500
remaining_to_process = max(0, target_total - existing_count)

print(f"Target total: {target_total}")
print(f"Already processed: {existing_count}")
print(f"Remaining to process: {remaining_to_process}")

if remaining_to_process == 0:
    print(f"Already reached target of {target_total} examples. No more processing needed.")
    exit(0)

# Load input data and slice from where we left off
if args.n:
    # Use args.n as the remaining count, but don't exceed remaining_to_process
    actual_n = min(args.n, remaining_to_process)
    print(f"Requested to process {args.n}, but will process {actual_n} to reach target total")
else:
    # Process all remaining to reach target total
    actual_n = remaining_to_process

all_input_data = [json.loads(line) for line in open(input_path, 'r')]
input_data = all_input_data[start_index:start_index + actual_n]

print(f"Starting from index {start_index}")
print(f"Will process {len(input_data)} examples")
print(f"Expected final total: {existing_count + len(input_data)}")

if not input_data:
    print(f"No more data to process starting from index {start_index}")
    exit(0)

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