from dataset import Dataset
import datasets


class TottoDataset(dataset.Dataset):

    def __init__(self, name, split, data_dir, proto_name, index_path, edit_dist_thresh=50, edit_dist_index_size=50, retrieval_k=5, sentence_encoder_name="stsb-distilbert-base"):
        super().__init__(name, split, data_dir, proto_name, index_path, retrieval_k)
        self.edit_dist_thresh = edit_dist_thresh
        self.edit_dist_index_size = edit_dist_index_size
        self.retrieval_k = retrieval_k
        self.sentence_encoder_name = sentence_encoder_name


    def mask_target(source, target):
        def hasNumbers(s):
            return any(char.isdigit() for char in s)

        keep_pos = ["AUX", "ADP", "DET", "SCONJ", "CCONJ", "PRON", "PART", "PUNCT", "VERB", "ADV"]
        target_doc = nlp(target)
        final_toks = []
        upper_flag = False
        for i, tok in enumerate(target_doc):
            word = tok.text
            if word[0].isupper() and (i > 0 or tok.pos_ == "PROPN"):
                if not upper_flag:
                    upper_flag = True
                    final_toks.append(Dataset.mask_str)
            else:
                if upper_flag:
                    upper_flag = False
                if tok.pos_ not in keep_pos and tok.text.lower() in source.lower():
                    final_toks.append(Dataset.mask_str)
                else:
                    final_toks.append(word)
                
        final_toks = [tok if not hasNumbers(tok) else Dataset.mask_str for tok in final_toks]
        return " ".join(final_toks)


    def create_eval_set(self, eval_k=40):
        data = datasets.load_from_disk(self.index_path)
        if "source_embed" not in data[0]:
            print("Source embeddings not in index, adding them now...")
            self.add_embed_index(source)
            data = datasets.load_from_disk(self.index_path)

        eval_src_lines = []
        
        for i in tqdm(range(len(data))):
            results = data.get_nearest_examples("source_embed", np.array(data[i]["source_embed"], dtype=np.float32), k=eval_k)

            proto_start = 1
            while results[0][proto_start] == 0:
                print("\n[{}] low score: {}".format(i, results[0][proto_start]))
                proto_start += 1

            eval_src_lines.append(proto + " [SEP] " + val_data[i]["source"] + "\n")

        with open("{}/{}.source".format(self.proto_data_dir, split), "w+") as f:
            f.writelines(eval_src_lines)

        shutil.copy("{}/{}.target".format(self.data_dir, split), "{}/{}.target".format(self.proto_data_dir, split))

