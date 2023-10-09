import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

data_dir = '/home/leiyu/scratch/hallucination_mech_evol/data/'
model_dir = '/home/leiyu/scratch/cma_hallucination/models/{}/'
results_dir = '/home/leiyu/scratch/hallucination_mech_evol/hall_mech/cpi/results/'
batch_size = 64
max_len_prompt = 32


class ParaRelDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        return {
            'prompt': self.df['prompt'][i].strip(),
            'true object': self.df['object'][i],
            'true object first token id': self.df['true object first token id'][i]
        }


def get_ent_vocab_ids(tokenizer):
    vocab = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    ent_vocab_ids = []
    for i in range(len(vocab)):
        t = vocab[i]
        if len(t) > 3 and t[0] == 'Ġ':
            if t[1].isupper() and t[1:].isalpha() and t[1:].lower() not in stop_words:
                ent_vocab_ids.append(i)
    return ent_vocab_ids


def get_top_pred_ents(batch_logits, ent_vocab_ids, tokenizer):
    batch_top_ent_ids, batch_top_ents = [], []
    batch_ent_logits = batch_logits[:, ent_vocab_ids]
    for x in torch.argsort(-batch_ent_logits, -1)[:, 0].cpu():
        batch_top_ent_ids.append(ent_vocab_ids[x])
        batch_top_ents.append(tokenizer.decode(ent_vocab_ids[x]).strip())
    return batch_top_ent_ids, batch_top_ents


def get_true_obj_ranks(batch_logits, batch_true_obj_ids):
    batch_sorted_tok_idx = torch.argsort(batch_logits, -1).cpu()
    batch_true_obj_ranks = []
    for i in range(batch_sorted_tok_idx.shape[0]):
        true_obj_rank_i = torch.where(batch_sorted_tok_idx[i] == batch_true_obj_ids[i])[0]
        if len(true_obj_rank_i) > 0:
            batch_true_obj_ranks.append(true_obj_rank_i[0].item())
        else:
            batch_true_obj_ranks.append(-1)

    return batch_true_obj_ranks


def prepare_batch_position_ids(batch_inputs):
    position_ids = []
    batch_max_len = batch_inputs['attention_mask'].shape[1]
    for i in range(batch_inputs['attention_mask'].shape[0]):
        n_tokens_i = batch_inputs['attention_mask'][i].sum()
        n_pad_i = batch_max_len - n_tokens_i
        pos_ids_i = torch.tensor([0] * n_pad_i + list(range(n_tokens_i)))
        position_ids.append(pos_ids_i)
    return torch.stack(position_ids)


def main(model_name='gpt2-xl'):
    device = torch.device("cuda")
    model_path = model_dir.format(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = nn.DataParallel(model).to(device)
    model.eval()

    pararel_df = pd.read_csv(data_dir + 'pararel_questions.csv')
    ds = ParaRelDataset(pararel_df.reset_index())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    ent_vocab_ids = get_ent_vocab_ids(tokenizer)

    top_ent_idx, top_ents, true_obj_ranks = [], [], []
    n_batch = int(len(ds) / batch_size) + 1
    for batch in tqdm(loader, total=n_batch):
        batch_inputs = tokenizer(
            batch['prompt'],
            return_tensors="pt",
            padding=True
        )
        batch_inputs['position_ids'] = prepare_batch_position_ids(batch_inputs)
        with torch.no_grad():
            batch_logits = model(**batch_inputs.to(device)).logits[:, -1]  # (B, vocab_size)
            batch_top_ent_ids, batch_top_ents = get_top_pred_ents(batch_logits, ent_vocab_ids, tokenizer)
            batch_true_obj_ranks = get_true_obj_ranks(batch_logits, batch['true object first token id'])

            top_ent_idx += batch_top_ent_ids
            top_ents += batch_top_ents
            true_obj_ranks += batch_true_obj_ranks

        torch.cuda.empty_cache()

    results = {
        'model predicted next entity': top_ents,
        'model predicted next token id': top_ent_idx,
        'model predicted true object rank': true_obj_ranks
    }
    with open(results_dir + "pararel_eval_{}.json".format(model_name), "w") as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
