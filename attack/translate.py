import os
import json
import asyncio
import requests
import argparse
from langcodes import Language
from call_openai import CallOpenAI

def valid_location():
    res = requests.get('https://ipinfo.io', timeout=5).text
    res = json.loads(res)
    country = res.get('country', '')
    print(json.dumps(res, indent=2))
    return country not in ["HK", "CN", "RU"]

def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def main(args):
    # assert valid_location(), "Invalid location"
    assert os.getenv("OPENAI_API_KEY"), "Set the OPENAI_API_KEY environment variable"

    def input_to_requests_func(input_file: str, output_file: str) -> list:
        """
        Convert input file to a list of requests for OpenAI API.

        Args:
            input_file (str): The path to the input file.
            output_file (str): The path to the output file.

        Returns:
            list: A list of requests for OpenAI API.

        Note: Exclude the requests that have been done.
        """

        rqs = [] # list of requests
        done_ids = [] # list of ids that have been done

        # read the output file to get the ids that have been done
        if os.path.isfile(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    done_ids.append(json.loads(line.strip())["id"])

        # read the input file to form the requests
        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                if i in done_ids:
                    continue
                prompt = json.loads(line.strip())["prompt"]
                response = json.loads(line.strip())["response"]
                rq = {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Translate the following {Language.make(language=args.src_lang).display_name()} text to {Language.make(language=args.tgt_lang).display_name()}:\n\n{response}"
                        }
                    ],
                    "temperature": args.temperature,
                    "metadata": {"row_id": i, "prompt": prompt} # store the id of the request as metadata
                }
                rqs.append(rq)
        return rqs

    def response_to_output_func(response: dict, output_file_path: str):
        """
        Invoked after each succesful API call.
        Convert OpenAI response to output and write it to output file.

        Args:
            response (dict): Dict of OpenAI response in the format of {"response": ..., "metadata": ...}.
            output_file_path (str): The path to the output file.

        Returns:
            None
        """

        translation = response["response"]["choices"][0]["message"]["content"] # Extract the translation from the response
        id = response["metadata"]["row_id"] # Extract the row ID from the metadata
        prompt = response["metadata"]["prompt"]

        # Write the translation to the output file as a json string. Temporarily, we use the output file as a jsonl file.
        json_string = json.dumps(
            {
                "id": id,
                "prompt": prompt,
                "response": translation
            },
            ensure_ascii=False
        )
        with open(output_file_path, "a") as f:
            f.write(json_string + "\n")

    def post_run_func(output_file_path: str):
        """
        Invoked after all API calls are done.
        Organize the output file into the desired format

        Args:
            output_file_path (str): The path to the output file.

        Returns:
            None
        """

        # Read the output file and sort the translations by id
        results = []
        with open(output_file_path, 'r') as f:
            for line in f:
                results.append(json.loads(line.strip()))
        results = sorted(results, key=lambda x: x['id'])

        # Write the translations to the output file
        with open(output_file_path, "w") as f:
            for r in results:
                json_string = json.dumps({"prompt": r["prompt"], "response": r["response"]}, ensure_ascii=False)
                f.write(f"{json_string}\n")

    def is_all_done(input_file_path: str, output_file_path: str) -> bool:
        """
        Check if all the requests in the input file have been done.

        Args:
            input_file_path (str): The path to the input file.
            output_file_path (str): The path to the output file.

        Returns:
            bool: True if all the requests have been done, False otherwise.
        """

        if not os.path.isfile(output_file_path):
            return False

        with open(input_file_path, "r") as f:
            num_requests = len(f.readlines())

        with open(output_file_path, "r") as f:
            num_done = len(f.readlines())

        return num_requests == num_done

    openai_caller = CallOpenAI(
        request_url="https://api.openai.com/v1/chat/completions",
        api_key=os.getenv("OPENAI_API_KEY"),
        input_file_path=args.input_file,
        output_file_path=args.output_file,
        max_attempts=5,

        # Set the functions for converting input to requests, converting response to output, running after all API calls are done, and checking if all requests have been done
        input_to_requests_func=input_to_requests_func,
        response_to_output_func=response_to_output_func,
        post_run_func=post_run_func,
        is_all_done_func=is_all_done
    )

    asyncio.run(
        openai_caller.run()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--model", type=str, required=True, help="Model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--src_lang", type=str, required=True, help="Source language")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
    args = parser.parse_args()
    main(args)