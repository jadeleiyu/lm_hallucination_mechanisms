from datasets import load_from_disk
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

max_len_completion = 32
max_len_prompt = 32
num_beams = 3
batch_size = 16
n_data_eval = 100000

model_dir = '/home/leiyu/scratch/cma_hallucination/models/gpt2-xl/'
data_dir = '/home/leiyu/scratch/cma_hallucination/data/'


def main():
    lama_completion_df = pd.read_csv(
        data_dir + 'lama/gpt2-xl-completion-prompt-{}.csv'.format(max_len_prompt))

    lm_completions = []
    for i, row in tqdm(lama_completion_df.iterrows(), total=lama_completion_df.shape[0]):
        try:
            prompt = row['prompt']
            completion = row['LM completion']
            lm_completions.append(completion.split(prompt)[1].strip())
        except Exception as e:
            lm_completions.append('')

    detected_ents = []
    for doc in nlp.pipe(lm_completions, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]):
        detected_ents.append([(ent.text, ent.label_) for ent in doc.ents])

    detected_ent_types = []
    n_factual = 0
    n_hallucinate = 0
    for i, row in tqdm(lama_completion_df.iterrows(), total=lama_completion_df.shape[0]):
        if detected_ents[i]:
            if lm_completions[i] != '' and not row['true object'] in lm_completions[i]:
                detected_ent_types.append('hallucinated')
                n_hallucinate += 1
            elif lm_completions[i] != '' and row['true object'] in lm_completions[i]:
                detected_ent_types.append('factual')
                n_factual += 1
            else:
                detected_ent_types.append('NA')
        else:
            detected_ent_types.append('NA')

    lama_completion_df['detected entities in completion'] = detected_ents
    lama_completion_df['detected entities types'] = detected_ent_types
    detected_ent_df = {
        'lama dataset idx': [],
        'entity name': [],
        'entity factuality': []
    }
    for i in tqdm(range(len(detected_ents))):
        if detected_ent_types[i] != 'NA':
            for ent, _ in detected_ents[i]:
                detected_ent_df['lama dataset idx'].append(i)
                detected_ent_df['entity name'].append(ent)
                if lama_completion_df['true object'][i] == ent and detected_ent_types[i] == 'factual':
                    detected_ent_df['entity factuality'].append('factual')
                else:
                    detected_ent_df['entity factuality'].append('hallucination')

    detected_ent_df = pd.DataFrame(detected_ent_df)
    ent_factuality_type_count_df = detected_ent_df.groupby(['entity name']).agg({
        'entity factuality': lambda x: len(set(x))
    }).reset_index()

    target_ents = set(ent_factuality_type_count_df.loc[
                          ent_factuality_type_count_df['entity factuality'] > 1
                          ]['entity name'])

    n_target_completions_factual = len(detected_ent_df.loc[
                                           (detected_ent_df['entity name'].isin(target_ents)) & \
                                           (detected_ent_df['entity factuality'] == 'factual')
                                           ]['lama dataset idx'])

    n_target_completions_hallucinate = len(detected_ent_df.loc[
                                               (detected_ent_df['entity name'].isin(target_ents)) & \
                                               (detected_ent_df['entity factuality'] == 'hallucination')
                                               ]['lama dataset idx'])

    detected_ent_df.to_csv(data_dir + 'lama/gpt2-xl-completion-detected_ent-{}.csv'.format(max_len_prompt))
    lama_completion_df.to_csv(data_dir + 'lama/gpt2-xl-completion-prompt-{}.csv'.format(max_len_prompt))

    print('{} target entities has both factual and hallucinated generations'.format(len(target_ents)))
    print('number of examples with hallucinated target entity predicted: {}'.format(n_target_completions_hallucinate))
    print('number of examples with factual target entity predicted: {}'.format(n_target_completions_factual))


