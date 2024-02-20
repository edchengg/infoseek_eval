import os
import pandas as pd
import itertools
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from lavis.models import load_model_and_preprocess
import torch
from PIL import Image
import json
from infoseek_eval import evaluate as evaluate_infoseek
import argparse

split2data = {
        "val": "infoseek/infoseek_val.jsonl",
        "test": "infoseek/infoseek_test.jsonl",
        "human": "infoseek/infoseek_human.jsonl",
        "train": "infoseek/infoseek_train.jsonl"
    }

id2path = dict()

# load image paths
with open("id2image.jsonl", "r") as f:
    for line in f:
        line = json.loads(line)
        image_id = line["image_id"]
        path = line["image_path"]
        id2path[image_id] = path

def create_eval_data(split):
    # Read the input JSONL file
    with open(split2data[split], 'r') as f:
        batch_data = [json.loads(line) for line in f]

    clean_batch_data = []
    not_exit = []
    for idx, item in enumerate(batch_data):
        if idx % 10000 == 0:
            print(f"Processing {idx}/{len(batch_data)}")
        path = id2path[item["image_id"]]
        # check path exists
        if not os.path.exists(path):
            not_exit.append(item["image_id"])
        else:
            clean_batch_data.append(item)
    return clean_batch_data

def load_and_process_image(item):
    # Load and preprocess the image
    raw_image = Image.open(id2path[item["image_id"]]).convert("RGB").resize((224, 224))    
    processed_image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    return processed_image, item["question"], item["data_id"]

def process_images_in_batches(model, batch_data, batch_size, prompt):
    # Create a pool of workers
    # Monitor the progress of the pool
    
    output = []
    print("Generate predictions...")
    # Process images in batches
    for idx, i in enumerate(range(0, len(batch_data), batch_size)):
        if (idx + 1) % 100 == 0:
            print(f"Processing batch {idx}/{len(batch_data)/batch_size}")
        # Subset results for the current batch
        batch_subset = batch_data[i:i+batch_size]

        # Separate the images, questions, and ids
        batch_images, batch_questions, batch_ids = [], [], []

        # Load and preprocess the images
        for item in batch_subset:
            tmp_img, tmp_q, tmp_id = load_and_process_image(item)
            batch_images.append(tmp_img)
            batch_questions.append(tmp_q)
            batch_ids.append(tmp_id)

        # Concatenate the batch images
        image_batch = torch.cat(batch_images, dim=0)
        
        # add prompt to questions
        batch_questions = [prompt.format(q) for q in batch_questions]
        # Generate predictions for the batch
        
        answers = model.generate({"image": image_batch, "prompt": batch_questions},
                                 length_penalty=-1)
        # print(batch_questions)
        # print(answers)
        
        for idx, ans in zip(batch_ids, answers):
            output.append({"data_id": idx, "prediction": ans})
    return output

def evaluate_model(split, model, batch_size, step, prompt, name):
    # Create evaluate data
    batch_data = create_eval_data(split)
    # Process the data in batches
    output = process_images_in_batches(model, batch_data, batch_size, prompt)

    # Save the predictions
    pred_path = f"development/blip2_t5_{name}_flant5xxl_{split}_{step}.jsonl"
    ref_path = f"infoseek/infoseek_{split}.jsonl"
    with open(pred_path, "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")

    result = evaluate_infoseek(pred_path, ref_path)
    return result


class BLIP2Dataset(torch.utils.data.Dataset):
    def __init__(self, split, processor, PROMPT="Question: {} Short answer:"):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        self.image_path = []
        self.question = []
        self.answer = []
        with open(split2data[split], "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                image_id = line["image_id"]
                path = id2path[image_id]
                self.image_path.append(path)
                self.question.append(line["question"])
                self.answer.append(line["answer"][0])

        self.vis_processor = processor
        self.prompt = PROMPT
 
    def __getitem__(self, idx):
        raw_image = Image.open(self.image_path[idx]).convert("RGB").resize((224, 224))
        question = self.prompt.format(self.question[idx])
        answer = self.answer[idx]
        processed_image = self.vis_processor["train"](raw_image).unsqueeze(0)
        inputs = {"image": processed_image, "text_input": question, "text_output": answer}
        return inputs
 
    def __len__(self):
        return len(self.question)
    

if __name__ == "__main__":
    print("Initialize Processor...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", help="val, test, or human")
    parser.add_argument("--name", type=str, default="pretrain", help="blip2_t5 | blip2_vicuna_instruct | blip2_t5_instruct")
    parser.add_argument("--output_dir", type=str, default="predictions", help="output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="accumulation size")


    args = parser.parse_args()

    if args.name == "pretrain":
        model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", 
                                                            model_type="pretrain_flant5xxl", 
                                                            is_eval=True, 
                                                            device="cuda")
    elif args.name == "instruct":
        model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5_instruct", 
                                                            model_type="flant5xxl", 
                                                            is_eval=True, 
                                                            device="cuda")
        

    raw_image = Image.open("aircraft.png").convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to("cuda")
    output = model.generate({"image": image, "prompt": "Question: what is the date this aircraft took the first flight? Answer:"})
    print(output)

    blip_dataset = BLIP2Dataset(
        split="train",
        processor=vis_processors,
        PROMPT="Question: {} Short answer:"
    )
    print("Initialize Dataloader...")
    # Padding dataloader
    train_dataloader = DataLoader(
        blip_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6
    )

    # # freeze everything except qformer
    print("Freeze Model...")
    for name, param in model.named_parameters():
        if "Qformer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # optmizer adamw for all parameters require grad
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    accum_iter = args.accumulation_steps

    optimization_step = 0
    for epoch in range(1):
        print("Epoch:", epoch)
        for idx, batch in enumerate(train_dataloader):
            batch["image"] = batch["image"].squeeze(1).to(device)
            output = model(samples=batch)
            loss = output["loss"]
            # Gradient acculation
            loss = loss / accum_iter
            loss.backward()
            print(loss.item())
            if (idx + 1) % accum_iter == 0:
                optimization_step += 1
                optimizer.step()
                optimizer.zero_grad()

                if (optimization_step + 1) % 100 == 0:
                    print("Evaluation...")
                    model.eval()
                    val_result = evaluate_model(split="val", model=model, batch_size=8, step=optimization_step, prompt="Question: {} Short answer:",
                                            name=args.name)      
                    print("Step:", idx)
                    print("Validation result:")
                    print(val_result)
                    cur_val_score = val_result["final_score"]
                    torch.save(model.state_dict(), f"development/blip2_t5_{args.name}_flant5xxl_{optimization_step}_val={cur_val_score}.pt")
                    model.train()

            if optimization_step > 1000:
                break