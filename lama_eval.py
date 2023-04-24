from datasets import load_from_disk
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator
import pandas as pd

max_len_completion = 5
max_len_prompt = 32
num_beams = 3
batch_size = 16
n_data_eval = 100000

model_dir = '/home/leiyu/scratch/cma_hallucination/models/gpt2-xl/'
data_dir = '/home/leiyu/scratch/cma_hallucination/data/'


def main():
    accelerator = Accelerator()
    device = accelerator.device

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    lama_dataset = load_from_disk(
        data_dir + 'lama/prompt-{}/'.format(max_len_prompt)).select(list(range(n_data_eval)))
    lama_loader = DataLoader(lama_dataset, batch_size=batch_size, shuffle=False)
    n_batch = int(len(lama_dataset) / batch_size)

    model, dataloader = accelerator.prepare(model, lama_loader)

    model.eval()
    lama_completion_df = {
        'prompt': [],
        'true object': [],
        'LM completion': []
    }

    for i, batch_inputs in tqdm(enumerate(lama_loader), total=n_batch):
        inputs = tokenizer(
            batch_inputs['prompt'],
            padding='max_length',
            truncation=True,
            max_length=max_len_prompt,
            return_tensors="pt"
        )
        with torch.no_grad():
            # pass through model(batch) once to avoid caffe errors
            if i == 0:
                model(
                    input_ids=inputs['input_ids'].to(device),
                    attention_mask=inputs['attention_mask'].to(device)
                )

            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=max_len_completion + max_len_prompt,
                num_beams=num_beams,
                early_stopping=True
            )
            # print('generated_tokens: ', generated_tokens)
            # print('generated_tokens.shape: ', generated_tokens.shape)
            generated_tokens = generated_tokens.cpu()
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            lama_completion_df['prompt'] += batch_inputs['prompt']
            lama_completion_df['true object'] += batch_inputs['obj_label']
            lama_completion_df['LM completion'] += decoded_preds

    lama_completion_df = pd.DataFrame(lama_completion_df)
    lama_completion_df.to_csv(
        data_dir + 'lama/gpt2-xl-completion-prompt-{}.csv'.format(max_len_prompt), index=False)


if __name__ == '__main__':
    main()
    # test_model_wrap()
