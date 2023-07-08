""" Infoseek Validation Set Evaluation script."""
from infoseek_eval import evaluate

if __name__ == "__main__":
    for split in ["val"]:
        for model in ["instruct", "pretrain"]:
            print(f"===BLIP2 {model} Flan T5 XXL===")
            pred_path = f"predictions/zeroshot_blip2_t5_{model}_flant5xxl_{split}.jsonl"
            reference_path = f"infoseek/infoseek_{split}.jsonl"
            reference_qtype_path = f"infoseek/infoseek_{split}_qtype.jsonl"

            result = evaluate(pred_path, reference_path, reference_qtype_path)
            final_score = result["final_score"]
            unseen_question_score = result["unseen_question_score"]["score"]
            unseen_entity_score = result["unseen_entity_score"]["score"]
            print(f"{split} final score: {final_score}")
            print(f"{split} unseen question score: {unseen_question_score}")
            print(f"{split} unseen entity score: {unseen_entity_score}")
