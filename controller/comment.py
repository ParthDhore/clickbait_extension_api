from transformers import AutoTokenizer
from controller.cleantext import clean_text

def commentcheck(model, comments, output):
    comments=clean_text(comments)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encodding = tokenizer(comments, return_tensors="np",truncation=True, padding=True)
    data=dict(encodding)

    op=model(data)
    output['comment_a']=op['logits'].__array__()[0][0]
    output['comment_b']=op['logits'].__array__()[0][1]
    # return [op['logits'].__array__()[0][0], op['logits'].__array__()[0][1]]