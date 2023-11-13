import re
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    row= re.sub(emoj, '', data)
    row = re.sub("(\\t)", ' ', row).lower()  # Remove Escape character
    row = re.sub("(\\r)", ' ', row).lower() 
    row = re.sub("(\\n)", ' ', row).lower()
    row = re.sub("(__+)", ' ', row).lower() # remove _ if it occors more than one time consecutively
    row = re.sub("(--+)", ' ', row).lower() # remove - if it occors more than one time consecutively
    row = re.sub("(~~+)", ' ', row).lower() # remove ~ if it occors more than one time consecutively
    row = re.sub("(\+\++)", ' ', row).lower() # remove + if it occors more than one time consecutively
    row = re.sub("(\.\.+)", ' ', row).lower() # remove . if it occors more than one time consecutively
    row = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', row).lower() # remove <>()|&©ø"',;?~*!

    return row

def remove_stop_words(row):
    #remove stop words from list of tokenized words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(row)
    filtered_sentence=[]
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    sent=" ".join(filtered_sentence)
    return sent

def text_processing(text):
    clear_text = clean_text(text)
    words = remove_stop_words(clear_text)
    return words

