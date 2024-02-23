import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def main(args):
    input_list = []
    with open(args.input_file, 'r') as f:
        input_list = json.load(f)

    output_list = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            output_list = json.load(f)

    if len(output_list) == len(input_list):
        print("Data already translated.")
        return

    input_list = input_list[len(output_list):]
    for ii in tqdm(input_list):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a Chinese-English translator, and now you need to translate the following Chinese text into English:
{ii}
English translation (just give the translated text directly):"""
                }
            ],
            model="gpt-3.5-turbo-0613",
            temperature=0.7,
        )
        oo = chat_completion.choices[0].message.content
        output_list.append(oo)

        with open(args.output_file, 'w') as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate from English to Chinese')
    parser.add_argument('--input_file', type=str, required=True, help="A input json file of list of strings.")
    parser.add_argument('--output_file', type=str, required=True, help="A output json file of list of strings.")

    args = parser.parse_args()
    main(args)