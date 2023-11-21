from transformers import AutoTokenizer
from controller.cleantext import clean_text

def titlecheck(model, title, output):
    title=clean_text(title)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encodding = tokenizer(title, return_tensors="np",truncation=True, padding=True)
    data=dict(encodding)

    op=model(data)
    output['title_a']=op['logits'].__array__()[0][0]
    output['title_b']=op['logits'].__array__()[0][1]
    # return [op['logits'].__array__()[0][0], op['logits'].__array__()[0][1]]
