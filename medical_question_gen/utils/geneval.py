from nlgeval import NLGEval

def calculate_rouge(prediction, ground_truth, tokenizer):
    nlgeval = NLGEval()
    references = []
    hypotheses = []
    for x, y in zip(ground_truth, prediction):
        x= tokenizer.decode(x,skip_special_tokens=True)
        y= tokenizer.decode(y,skip_special_tokens=True)
        references.append([x])
        hypotheses.append(y)

    metrics_dict = nlgeval.compute_metrics(references, hypotheses)
    return metrics_dict['ROUGE_L'], references, hypotheses
