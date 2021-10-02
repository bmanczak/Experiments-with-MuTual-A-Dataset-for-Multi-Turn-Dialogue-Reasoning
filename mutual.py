from run import *

def mutual_data_loader(data_dir, tokenizer, max_doc_len, max_query_len, max_option_len, is_training):
    """Replace the load data fucntion in run.py to use"""
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    if is_training:
        subset_list = ['train']
    else:
        subset_list = ['dev', 'test']

    examples, features = {}, {}
    for subset in subset_list:
        level_example_dict = {"high": None, "middle": None}
        level_features_dict = {"high": None, "middle": None}

        for level in ['high', 'middle']:
            subset_dir = os.path.join(data_dir, subset)
            file_list = os.listdir(subset_dir)
            file_list = [file for file in file_list if file.endswith('txt')]
            file_list = sorted(file_list)

            alldata = []
            for file in file_list:
                data = json.load(open(os.path.join(subset_dir, file)))
                alldata.append(data)
        
            examples_list = []
            for data in alldata:
                doc_id = data['id']
                doc = data["article"].replace('\\n', '\n')
                doc_token = []
                prev_is_whitespace = True
                for c in doc:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_token.append(c)
                        else:
                            doc_token[-1] += c
                        prev_is_whitespace = False

                for i, answer in enumerate(data["answers"]):
                    example = InputExample(
                        guid=doc_id + '-%d' % i,
                        doc_token=doc_token,
                        question_text="",
                        options=data["options"],
                        answer=answer)
                    examples_list.append(example)

            level_example_dict[level] = examples_list
            level_features_dict[level] = convert_examples_to_features(
                level_example_dict[level], tokenizer, max_doc_len, max_query_len, max_option_len)

        examples[subset] = level_example_dict
        features[subset] = level_features_dict

    return examples, features

def dataset_selector(data_dir):
    if data_dir = "RACE":
        return load_data
    elif data_dir = "mutual"
        return mutual_data_loader
    else:
        print("dataset not implemeted.")
        raise NotImplementedError 