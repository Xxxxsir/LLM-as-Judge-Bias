import json
from time import sleep
from typing import Dict, List
from judge_agent.prompt import judge_prompt, generate_prompt, bias_dicts,easy_judge_prompt,comb_generate_prompt,dialogue_judge_prompt
from tqdm import tqdm
from judge_agent.response_eval import (
    score_config,
    run_llm_judge,
)
from judge_agent.pipeline import run_pipeline
from score import compute_average_score

prompt_template_dict = {
    "judge_prompt": judge_prompt,
    "generate_prompt": generate_prompt,
    "comb_generate_prompt":comb_generate_prompt
}


if __name__ == "__main__":
    model_name = "gpt-5.1"
    aspects = ["score"]

    bias_list = [
        "clean",
    ]

    for bias_type in bias_list:

        input_file_path = f"/home/chenchen/gjx/Judge/data/ours/bias/{bias_type}_50p_gpt4o.jsonl"
        
        run_llm_judge(
            input_path=input_file_path,
            output_path=input_file_path,
            model_name=model_name,
            prompt_template=judge_prompt,
            score_aspects=aspects,
            **score_config["0-10"],
        ) 

        print(f"Finished evaluating {bias_type} bias. Computing average score...")
        compute_average_score(input_file_path, limit=100)

        sleep(10)