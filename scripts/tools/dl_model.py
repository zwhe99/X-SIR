import sys
from transformers import AutoModel, AutoTokenizer

AutoTokenizer.from_pretrained(sys.argv[1], use_fast=False, trust_remote_code=True)
AutoModel.from_pretrained(sys.argv[1], trust_remote_code=True)


