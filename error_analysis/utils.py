import re

"""
The functions below contain general helper functions useful for dissecting
MuTual samples.
"""

def convert_answer_to_index(answer):
    """
    Converts an answer such as "C" to a number starting from 0.
    
    Input:
    
    @answer - Answer cast as a char.
    
    Output:
    
    @answer - Answer as an integer index.
    """
    
    print(answer, ord(answer) - 65)
    
    return ord(answer) - 65
    
def convert_context_to_sentences(context, strip_annotation = True,
                                                            filter_by = None):
    """
    Splits an article (context) into seperate sentences.
    
    Input:
    
    @context          - String of context.
    @strip_annotation - If True, annotations (e.g. 'm: ' and 'f: ' will be
                        stripped and excluded from the returned sentences).
    @filter_by        - If set (regardless of @strip_annotation), only samples
                        annotated by the value of @filter_by will be included
                        in the returned sentences. (E.g. 'm' for 'm: blabla').
                        WARNING: this can be max ONE character of length!
    
    Output:
    
    @sentences - List of separate sentences contained in the context.
    """

    regex_pattern = "(([a-z]) : ((?:.(?![a-z] :))*))"

    matches   = re.findall(regex_pattern, context)
    sentences = []
    
    for full_sentence, annotation, text in matches:
        if filter_by and filter_by == annotation:
            continue
        
        sentences.append(text if strip_annotation else full_sentence)
        
    return sentences