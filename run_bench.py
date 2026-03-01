import argparse
import copy
import time
from vllm.benchmarks.serve import add_cli_args, main

NUM_PROMPTS_AUTO = "auto"


def parse_args():
    parser = argparse.ArgumentParser()
    add_cli_args(parser)

    parser.add_argument(
        "--max-concurrency-list",
        type=lambda x: [int(i) for i in x.split(",")],
        default=[128, 256, 512, 1024],
        help="Comma-separated list of max concurrency values to sweep.",
    )

    parser.set_defaults(
        request_rate=float("inf"),
        random_input_len=1024,
        random_output_len=1024,
        random_range_ratio=0.0,
        result_dir="results",
        dataset_name="random",
        ignore_eos=True,
        save_result=True,
        percentile_metrics="ttft,tpot,itl,e2el",
        metric_percentiles="50,90,99",
    )

    args = parser.parse_args()
    return args


def run(args: argparse.Namespace, max_concurrency: int):
    args = copy.deepcopy(args)
    args.max_concurrency = max_concurrency
    args.num_prompts = max(max_concurrency * 2, 256)
    if args.result_filename:
        args.result_filename = f"{args.result_filename}_{max_concurrency}.json"
    return main(args)


if __name__ == "__main__":
    args = parse_args()
    for max_concurrency in args.max_concurrency_list:
        print(f"Running with max concurrency: {max_concurrency}")
        result = run(args, max_concurrency=max_concurrency)
        time.sleep(15)
