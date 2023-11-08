import os
import argparse
import requests
import json, time, re, datetime, ray
import random
import pandas as pd
from num2words import num2words
import transformers
import csv

sys_prompt = "You are a helpful assistant that respeonds with the answer in the most concise possible way."

def prompt_generator(num_digits=3, min_lines=15, max_lines=1000, file_lines=[]) -> str:
    # Step 1: Generate a random number
    # Generate the number of digits specified (e.g. if NUM_DIGITS = 3, then
    # any number between 100 and 1000 is OK).
    rnd_num = random.randrange(10 ** (num_digits - 1), 10 ** (num_digits))
    max_lines = max_lines if max_lines < len(file_lines) else len(file_lines)
    rnd_num_lines = random.randrange(min_lines, max_lines)
    rnd_picked_lines = "\n".join(random.sample(file_lines, rnd_num_lines))

    # Step 2: convert to words.
    rnd_num_words = num2words(rnd_num)

    # Step 3: convert to a prompt
    user_prompt = f"The numerical value of the following sequence of words : {rnd_num_words} is"

    return user_prompt, rnd_num

@ray.remote(num_cpus=1)
def validate(args, sample_lines):
    # The 4 is for the end and start tokens of the messages
    prompt, rnd_num = prompt_generator(
        args.num_digits, args.min_lines, args.max_lines, sample_lines
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True, device_map="auto"
        )
    tokens_in = len(tokenizer.encode(prompt)) + len(tokenizer.encode(sys_prompt)) + 4
    words = ""
    id = None
    st = et = ttft = 0
    url = f"http://localhost:8080/predictions/{args.model_name}"
    headers = {"Content-Type": "application/text; charset=utf-8"}
    st = time.time()
    data = sys_prompt + prompt
    try:
        response = requests.post(url, data=data, timeout=120, headers=headers, stream=True)
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                if ttft == 0:
                    ttft = time.time() - st
                data = chunk.decode("utf-8")
                words+=data
                print(data, end="\n", flush=True)
        et = time.time()
    except requests.exceptions.RequestException as e:
        return ("Exception", -1, -1, -1, -1, str(e), "")
    # Get rid of commas.
    tokens_out = len(tokenizer.encode(words))
    nums = re.findall(r"\d+", words)
    if len(nums) > 0:
        retval = int(nums[0])
        valid = "OK"
        cause = ""
        if retval != rnd_num:
            valid = "Mismatch"
            cause = f"Input = {rnd_num} output = {retval}\n.Output:\n {words}"
    else:
        valid = "Mismatch"
        cause = f"Output unparseable. Input = {rnd_num}. Output:\n {words}"
    return (valid, ttft, et - st, tokens_in, tokens_out, cause, id)

def store_result(results_dict):
    # Define the column names
    field_names = results_dict.keys()
    csv_file = "benchmark.csv"

    # Create and open the CSV file for writing
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        # Sample data rows
        # Write the data from the dictionary
        # for i in range(len(results_dict["total_requests"])):
        row = {key: results_dict[key] for key in field_names}
        writer.writerow(row)
    print(f"Benchmarking data has been written to {csv_file}")


def results_analysis(query_results, results_dict):
    df = pd.DataFrame(
        query_results,
        columns=[
            "valid",
            "ttft",
            "total_time",
            "tokens_in",
            "tokens_out",
            "cause",
            "id",
        ],
    )
    ts = int(time.time())
    fn = f'{results_dict["framework"]}-{ts}_raw.json'
    # df.to_json(fn)
    print(f"Results saved to: {fn}")

    print("Validity results:")
    print(df["valid"].value_counts())

    value_counts = df["valid"].value_counts()
    results_dict["num_valid"] = int(value_counts.get("OK", 0))
    results_dict["num_exceptions"] = int(value_counts.get("Exception", 0))
    results_dict["num_mismatch"] = int(value_counts.get("Mismatch", 0))
    results_dict["valid_rate"] = float(
        results_dict["num_valid"] / results_dict["total_requests"]
    )
    results_dict["mismatch_rate"] = float(
        results_dict["num_mismatch"] / results_dict["total_requests"]
    )
    results_dict["exception_rate"] = float(
        results_dict["num_exceptions"] / results_dict["total_requests"]
    )
    cdf = df[df.valid != "Exception"].copy()
    print(f"Clean DF is: {len(cdf)}")
    if len(cdf) > 0:
        cdf["total_tokens_per_s"] = (cdf.tokens_out + cdf.tokens_in) / cdf.total_time
        cdf["out_tokens_per_s"] = cdf.tokens_out / cdf.total_time
        cdf["inter_tokens_delay"] = cdf.total_time / cdf.tokens_out
        mean_e2e = cdf["total_time"].mean()
        mean_tokens_in = cdf["tokens_in"].mean()
        mean_tokens_out = cdf["tokens_out"].mean()
        mean_ttft = cdf["ttft"].mean()
        max_ttft = cdf["ttft"].max()
        gt_3_ttft = len(cdf[cdf["ttft"] > 3]) / len(cdf)
        print(f"Mean End-to-end: {mean_e2e*1000.0:.0f} ms")
        print(
            f"Mean TTFT: {mean_ttft*1000:.0f} ms (mean tokens in: {mean_tokens_in:.0f}, out: {mean_tokens_out:.0f})"
        )
        print(f"Max TTFT: {max_ttft*1000:.0f} ms")
        print(f"TTFT > 3 s: {gt_3_ttft*100:.2f}%")
        print(
            f"ITL (out): {cdf.inter_tokens_delay.mean()*1000:.2f} ms/token, mean tokens/s output (out): {cdf.out_tokens_per_s.mean():.2f} token/s"
        )
        # Put things in a dictionary and save the results
        results_dict["end_timestamp"] = datetime.datetime.fromtimestamp(ts).isoformat()
        results_dict["total_time"] = float(cdf.total_time.mean())
        results_dict["mean_ttft"] = int(f"{mean_ttft*1000:.0f}")
        results_dict["mean_tokens_in"] = mean_tokens_in
        results_dict["mean_tokens_out"] = mean_tokens_out
        results_dict["total_tokens_per_s"] = float(cdf.total_tokens_per_s.mean())
        results_dict["out_tokens_per_s"] = float(cdf.out_tokens_per_s.mean())
        results_dict["inter_token_delay"] = float(cdf.inter_tokens_delay.mean() * 1000)
        # store_result(results_dict)


def endpoint_evaluation(args, sample_lines):
    query_results = []
    overall_start_time = time.time()
    num_rounds = int(args.total_requests / args.concur_requests)
    for i in range(num_rounds):
        print(f"Starting round {i}")
        st = time.time()
        futures = [
            validate.remote(args, sample_lines)
            for _ in range(args.concur_requests)
        ]
        results = ray.get(futures)
        query_results.extend(results)
        et = time.time()
        elt = et - st
        tosleep = args.sleep - elt
        if tosleep > 0:
            print("Sleeping for %.4f seconds" % tosleep)
            time.sleep(tosleep)
        else:
            print(f"No need to sleep for the next round")
        print(f"Round {i} complete")
    overall_end_time = time.time()
    print(f"Overall execution time {overall_end_time-overall_start_time}")
    return query_results    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--framework", type=str, default="torchserve", help="Test frame name"
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="",
        help="model name",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="model name",
        required=True,
    )
    parser.add_argument(
        "--random-lines-file-name",
        type=str,
        default="/home/ubuntu/nai-llmrepo/nai-llm/llm/sonnet.txt",
        help="Prompt sample file name",
    )
    parser.add_argument("--min-lines", type=int, default=15, help="min number of lines")
    parser.add_argument("--max-lines", type=int, default=50, help="max number of lines")
    parser.add_argument(
        "--req-lines",
        type=int,
        default=7,
        help="Number of lines to request in prompt. Affects tokens out.",
    )
    parser.add_argument(
        "--num-digits", type=int, default=3, help="number of digits for mismatch search"
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=0,
        help="sleep between rounds of requests (to deal with rate limiting)",
    )
    parser.add_argument(
        "-c",
        "--concur-requests",
        type=int,
        default=4,
        help="number of concurrent requests",
    )
    parser.add_argument(
        "-r", "--total-requests", type=int, default=4, help="number of total requests"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=384,
        help="Upper limit on the number of returned tokens to prevent 'runaway LLMs'.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=117,
        help="Random seed to standardize results. By default fully random.",
    )
    args = parser.parse_args()
    f = open(args.random_lines_file_name, "r")
    sample_lines = f.readlines()
    f.close()
    query_results = endpoint_evaluation(args, sample_lines)
    print(query_results)
    results_analysis(query_results, vars(args))
    # test_inference_txt_file_success(args, sample_lines)