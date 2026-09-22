import time
import json
import requests
import psutil
import subprocess
import argparse

import pandas as pd

from datetime import datetime
from typing import Dict, List, Any


class VLLMBenchmark:
    """
    Benchmark script for OpenWebUI -> OpenAI Compatible vLLM Server

    Supports:
        - Gemma4
        - Qwen
    """

    def __init__(self, base_url: str = "http://localhost:portno./"):
        # Example:
        # http://<openwebui-host>:3000/api
        self.base_url = base_url.rstrip("/")
        self.results = []

    # ------------------------------------------------------------------
    # NEW: load prompts from an Excel sheet instead of hardcoding them
    # ------------------------------------------------------------------
    @staticmethod
    def load_prompts_from_excel(
        file_path: str,
        sheet_name: str = 0,
        prompt_column: str = "Prompt",
    ) -> List[str]:
        """
        Read only the prompt text from an Excel file.

        The sheet may contain other columns (e.g. "Benchmark Category",
        "Capability Tested") — those are ignored entirely; only the
        prompt_column is extracted.

        Returns a plain list of prompt strings.
        """
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # normalize column names so header casing/spacing doesn't matter
        df.columns = [str(c).strip() for c in df.columns]

        matches = [c for c in df.columns if c.strip().lower() == prompt_column.strip().lower()]
        if not matches:
            raise ValueError(
                f"Could not find a '{prompt_column}' column in {file_path}. "
                f"Available columns: {list(df.columns)}"
            )
        actual_column = matches[0]

        # drop rows with no prompt text
        df = df.dropna(subset=[actual_column])

        prompts = [str(p).strip() for p in df[actual_column].tolist()]

        return prompts

    def benchmark_model(self, model_name: str, prompts: List[str]) -> Dict[str, Any]:

        print(f"\nBenchmarking Model : {model_name}")

        model_results = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }

        for i, prompt in enumerate(prompts, start=1):

            print("\n" + "=" * 70)
            print(f"Running Prompt {i}/{len(prompts)}")
            print("=" * 70)

            result = self._run_single_test(
                model_name=model_name,
                prompt=prompt,
                prompt_id=i,
            )

            model_results["tests"].append(result)
            print(json.dumps(result, indent=4))

        return model_results

    def _run_single_test(
            self,
            model_name: str,
            prompt: str,
            prompt_id: Any,
        ) -> Dict[str, Any]:
    
        process = psutil.Process()

        memory_before = process.memory_info().rss
        cpu_before = psutil.cpu_percent(interval=None)

        
    
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False
                },
                timeout=300
            )

            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "status": "FAILED",
                "error": str(e)
            } 
        
        end_time = time.time()
        
        response_time = end_time - start_time

        memory_after = process.memory_info().rss
        cpu_after = psutil.cpu_percent(interval=None)

        gpu_after = self.get_gpu_metrics()

        

        data = response.json()

        response_text = ""

        if "choices" in data:
                response_text = (
                    data["choices"][0]
                    .get("message", {})
                    .get("content", "")
                )

        usage = data.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", None)
        completion_tokens = usage.get("completion_tokens", None)
        total_tokens = usage.get("total_tokens", None)

        if completion_tokens is None:
            completion_tokens = len(response_text.split())

        tokens_per_second = (
            completion_tokens / response_time
            if response_time > 0 else 0
        )

        return {

        "timestamp": datetime.now().isoformat(),

        "model": model_name,

        "prompt_id": prompt_id,

        "prompt": prompt,

        "prompt_length": len(prompt),

        "response_time_sec": round(response_time, 4),

        "response_length": len(response_text),

        "prompt_tokens": prompt_tokens,

        "completion_tokens": completion_tokens,

        "total_tokens": total_tokens,

        "tokens_per_second": round(tokens_per_second, 2),

        "cpu_before_percent": cpu_before,

        "cpu_after_percent": cpu_after,

        "memory_before_mb": round(memory_before / (1024 * 1024), 2),

        "memory_after_mb": round(memory_after / (1024 * 1024), 2),

        "memory_delta_mb": round(
            (memory_after - memory_before) / (1024 * 1024), 2
        ),

        "gpu_utilization_percent": gpu_after.get("gpu_utilization"),

        "gpu_memory_used_mb": gpu_after.get("gpu_memory_used"),

        "gpu_memory_total_mb": gpu_after.get("gpu_memory_total"),

        "http_status": response.status_code,

        "response": response_text

        }
 
    @staticmethod
    def get_gpu_metrics():
        """
        Collect GPU metrics using nvidia-smi.
        Returns None if NVIDIA GPU is unavailable.
        """
        try:
                result = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits"
                    ],
                    encoding="utf-8"
                ).strip()
    
                gpu_util, mem_used, mem_total = result.split(",")
    
                return {
                    "gpu_utilization": float(gpu_util.strip()),
                    "gpu_memory_used": float(mem_used.strip()),
                    "gpu_memory_total": float(mem_total.strip())
                }
    
        except Exception:
                return {
                    "gpu_utilization": None,
                    "gpu_memory_used": None,
                    "gpu_memory_total": None
                }

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM models using prompts from an Excel file.")
    parser.add_argument("--base-url", default="url/api", help="OpenWebUI/vLLM OpenAI-compatible base URL, e.g. http://host:3000/api")
    parser.add_argument("--prompts-file", default="#path", help="Path to prompts")
    parser.add_argument("--sheet-name", default= "Benchmarks", help="Sheet name or index to read (default: first sheet)")
    parser.add_argument("--prompt-column", default="Prompt", help="Column header containing prompt text")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["google/gemma-4-E4B-itge"],
        help="Model name(s) as registered on the vLLM server",
        )


    args = parser.parse_args()

    benchmark = VLLMBenchmark(args.base_url)

    prompts = benchmark.load_prompts_from_excel(
        file_path=args.prompts_file,
        sheet_name=args.sheet_name,
        prompt_column=args.prompt_column,
    )
    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    

    
    for model_name in args.models:
        model_results = benchmark.benchmark_model(
            model_name=model_name,
            prompts=prompts
        )

        

    print("\n" + "=" * 70)
    print("Benchmark Completed Successfully.")
    print(f"Total Prompts Executed : {len(prompts)}")
    print("=" * 70)


if __name__ == "__main__":
    main()  