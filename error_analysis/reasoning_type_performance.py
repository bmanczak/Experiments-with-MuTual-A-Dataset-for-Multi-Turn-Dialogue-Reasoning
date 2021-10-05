import json
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict, Counter

def id_to_correct_pred(results):
    """
    Returns a dictionary with dialogue_id: 0 or 1
    
    Arguments:
    
    results: list
        A nested list (from a json object) produced in the evalaution of the models
    """
    
    dialogue_id_to_correct = {} # defaultdict(int) # stores which dialouge id 

    for i in range(len(results)):
        subres = results[i][1]#[0] # [1] contains the prediction and the choices, [0] we get the dic
        for question_ix, res in enumerate(subres):
            
            for opt_ix, opt in enumerate(res["choice"]):
                if opt == res["answer"]:
                    results[i][1][question_ix]["answer_ix"] = opt_ix
                    continue

            if res["prediction"]==results[i][1][question_ix]["answer_ix"]:
                results[i][1][question_ix]["correct_pred"] = 1
            else:
                results[i][1][question_ix]["correct_pred"] = 0

            dialogue_id_to_correct[(results[i][2], str(question_ix))] = results[i][1][question_ix]["correct_pred"] 
    
    return dialogue_id_to_correct

def load_data_to_dict(annotation_files = None):
    """
    Loads the DREAM annotations and saves them as a dictionary.
    
    Arguments:

    annotation_files: list
        List of paths to the annotation files.
    Returns 
    {
        (dialalogue_ID, question_ix): annotation_type
    }
    """
    
    data_dict = {}

    if annotation_files is None:
        files = [
            #'annotator1_dev.txt',
            'annotator1_test.txt',
            #'annotator2_dev.txt',
            'annotator2_test.txt'
        ]

    else:
        files = annotation_files


    for file in files:   
        data_dict[file] = {}     
        with open(file, 'r') as f_in:
            data_temp = f_in.readlines()
        
        for i, line in enumerate(data_temp):
            if i == 0:
                continue # Dont use column names
            line_temp = line.strip().split('\t')
            q_id = line_temp[0]
            q_ix = str(int(line_temp[1]) - 1) # They start ix from 1, not 0...
            q_type = line_temp[2]
            
            data_dict[file][(q_id, q_ix)] = q_type
            
    return data_dict


def get_accuracy_per_type(id_to_correct_pred_dic, annotations_dic):
    """
    Returns a dictionary in the format:
    {reasoning_type: mean accuracy}
    
    Arguments:
        id_to_correct_pred: dict
            Output of the id_to_correct_pred function
        annotations_dic: dict
            load_data_to_dict
    """
    
    list_accuracies = defaultdict(list)
    
    for key, annotation_type in annotations_dic.items():
        list_accuracies[annotation_type].append(id_to_correct_pred_dic[key])
        
    mean_accuracies = {}
    for annotation_type, binary_accs in list_accuracies.items():
        mean_accuracies[annotation_type] = np.mean(binary_accs)
        
    return mean_accuracies
        

def get_accuracy_per_type(id_to_correct_pred_dic, annotations_dics):
    """
    Returns a dictionary in the format:
    {reasoning_type: mean accuracy}
    
    Arguments:
        id_to_correct_pred: dict
            Output of the id_to_correct_pred function
        annotations_dic: dict or list of dicts.
           Output(s) of load_data_to_dict. In case of more than one annotation the accuracies
            the accuracies are averaged over the different annotations.
            
    """
    
    if type(annotations_dics) == dict:
        annotations_dics = [annotations_dics]
    
    mean_accuracies = defaultdict(list)
    list_accuracies = defaultdict(list)
    for annotations_dic in annotations_dics: 
        
        #list_accuracies = defaultdict(list)
        
        for key, annotation_type in annotations_dic.items():
            list_accuracies[annotation_type].append(id_to_correct_pred_dic[key])
        
    for annotation_type, binary_accs in list_accuracies.items():
        mean_accuracies[annotation_type].append(np.mean(binary_accs))
        
    final_dic = {}
    for key, val in mean_accuracies.items():
        final_dic[key] = val[0] # have to get rid of the list
    
    return final_dic

if __name__=="__main__":
    with open("reasoning_type_data/preprocessed_test_ocn.json", "r") as json_in:
        results_ocn = json.load(json_in)
    
    with open("reasoning_type_data/preprocessed_test_no_ocn.json", "r") as json_in:
        results_no_ocn = json.load(json_in)

    id_to_correct_pred_ocn = id_to_correct_pred(results_ocn)
    id_to_correct_pred_no_ocn = id_to_correct_pred(results_no_ocn)

    annotators = load_data_to_dict(["reasoning_type_data/annotator1_test.txt",
                                    "reasoning_type_data/annotator2_test.txt"])
    annotator1_test, annotator2_test = annotators["reasoning_type_data/annotator1_test.txt"], annotators["reasoning_type_data/annotator2_test.txt"]

    res_no_ocn = get_accuracy_per_type(id_to_correct_pred_dic = id_to_correct_pred_no_ocn,
                     annotations_dics = [annotator1_test, annotator2_test ] )
    
    res_ocn = get_accuracy_per_type(id_to_correct_pred_dic = id_to_correct_pred_ocn,
                     annotations_dics = [annotator1_test, annotator2_test])

    dic = {"c": "Commonsene", "m":"Matching","l": "Logic", "cs": "Commonsesne summary","cl": "Commonsesne logic", "al" :"Artihemtic logic", "s": "Summary", "acl": "Artihemtic commonsense logic","a": "Arithmetic"}
    print(f"Result of the no ocn network")
    for key, value in res_no_ocn.items():
        print(f"{dic[key]}: {value} ")
    print("-"*40, "\n")
    print(f"Result of the ocn network {res_ocn}")
    for key, value in res_ocn.items():
        print(f"{dic[key]}: {value} ")
    print("Plotting ... ")

    ind = np.arange(len(res_no_ocn))
    width = 0.35       # the width of the bars

    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(111)


    rects1 = ax.bar(ind, res_no_ocn.values(), width, color='royalblue')
    rects2 = ax.bar(ind + width , res_ocn.values(), width, color='orange')

    
    ax.set_ylabel('Accuracy', size = 14)
    ax.set_title('Reasoning type', size = 14)
    ax.set_xticks(ind + width / 2)
    assert res_no_ocn.keys() == res_ocn.keys(), "Set the labels manually!" 
    ax.set_xticklabels( res_no_ocn.keys(),size = 14 )
    ax.legend((rects1[0], rects2[0]), ('No OCN', 'OCN'))

    plt.show()
    fig.savefig("reasoning_type_data/ReasoningType_Performance.jpg")