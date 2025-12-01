from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import pickle, os
from tqdm import tqdm

FOLDER_NAME = "/content/drive/MyDrive/HateMM/"

class Text_Model(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        if not model_name:
            raise ValueError("Invalid model name provided.")
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=True
            )
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        self.device = device
        
    def forward(self, input_ids, attention_mask):
        if self.model is None:
            raise RuntimeError("Model is not loaded correctly.")
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        return outputs


def tokenize(sentences, tokenizer, max_len=512):
    input_ids, attention_masks = [], []
    for sent in sentences:
        try:
            encoded_dict = tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )
        except Exception as e:
            print(f"Error encoding sentence '{sent}': {e}")
            continue
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return {'input_ids': input_ids, 'attention_masks': attention_masks}


def extract_features_from_pickled_file(file_path, model_name, output_file):
    # Load transcript dict
    with open(file_path, 'rb') as fp:
        transCript = pickle.load(fp)
    if not isinstance(transCript, dict):
        raise ValueError("Expected 'transCript' to be a dictionary")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Text_Model(model_name, device)

    allEmbedding = {}

    for text_id, text in tqdm(transCript.items()):
        try:
            apr = tokenize([text], tokenizer)
            # move to device
            input_ids = apr['input_ids'].to(device)
            attention_mask = apr['attention_masks'].to(device)

            outputs = model(input_ids, attention_mask)
            # last hidden layer CLS token: shape (hidden_dim,)
            last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden)
            cls_vec = last_hidden[:, 0, :].squeeze(0).cpu().numpy()

            allEmbedding[text_id] = cls_vec

        except Exception as e:
            print(f"Error processing text ID {text_id}: {e}")
            continue

    with open(output_file, 'wb') as fp:
        pickle.dump(allEmbedding, fp)
    print("Saved embeddings to:", output_file)


if __name__ == "__main__":
    extract_features_from_pickled_file(
        FOLDER_NAME + "all_whisper_tiny_transcripts.pkl",
        "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",
        FOLDER_NAME + "all_HateXPlainembedding_whisper.pkl"
    )
