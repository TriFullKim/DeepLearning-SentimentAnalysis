import torch
import argparse

# Input Must Be Sentence and Each Senstence Must Be Seperated by LF(\n)
argparser = argparse.ArgumentParser()
argparser.add_argument("--file", "-f", required=True, type=str)

SENTENSE_FILE_PATH = argparser.file


with open(SENTENSE_FILE_PATH ,"r") as f:
    inference_target = f.readlines()
    
    
with torch.no_grad():
    pass
    # model(inference_target)
    