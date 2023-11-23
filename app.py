from flask import Flask, Response, request,jsonify, make_response 
from googleapiclient.discovery import build
from controller.title import titlecheck
from controller.comment import commentcheck
from controller.thumbnail import check_thumbnail
import pickle
import numpy as np
from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf
import threading
import logging

# Single Threaded
# start=time.time()
# title_model=TFAutoModelForSequenceClassification.from_pretrained("titlemodel")
# comment_model=TFAutoModelForSequenceClassification.from_pretrained("commentmodel")
# thumbnail_model=tf.keras.models.load_model('thumbnailmodel')
# end=time.time()
# print("normal: {}".format(end-start))

title_model=None
comment_model=None
thumbnail_model=None

def load_titlemodel():
    global title_model
    title_model=TFAutoModelForSequenceClassification.from_pretrained("title_model")

def load_commentmodel():
    global comment_model
    comment_model=TFAutoModelForSequenceClassification.from_pretrained("comment_model")

def load_thumbnailmodel():   
    global thumbnail_model
    thumbnail_model=tf.keras.models.load_model('thumbnail_model')

threads=[
    threading.Thread(target=load_titlemodel),
    threading.Thread(target=load_commentmodel),
    threading.Thread(target=load_thumbnailmodel)
]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print('Setup done....')

app=Flask(__name__)

@app.route('/check/<video_id>',methods=['GET'])
def check_clickbait(video_id):
    app.logger.info('Youtube API start....')
    print('Youtube API start....')
    api_key="AIzaSyCCIcLVZQbpOvY7dci6KwNs81IisOm6hOo"
    youtube = build('youtube','v3', developerKey=api_key)
    video_response=youtube.videos().list(part='snippet,statistics',id=video_id).execute()
    title=""
    thumbnail=""
    comments=[]
    count=0
    if(len(video_response['items'])>0):
        try:    
            title = video_response['items'][0]['snippet']['title']
            thumbnail = video_response['items'][0]['snippet']['thumbnails']['high']['url']
        except Exception as e:
            title=title
            thumbnail=thumbnail
    else:
        title=""
        thumbnail=""
    
    try:
        comment_response=youtube.commentThreads().list(part='snippet,replies',videoId=video_id).execute()
        for item in comment_response['items']:
            comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
            count+=1
            if count==4:
                break
    except Exception as e:
        print(e)

    com=""
    for i in comments:
        com+=i

    app.logger.info('Youtube API finish....')
    print('Youtube API exit....')
    # Single Threaded
    # title_start=time.time()
    # tn=titlecheck(title_model, title)
    # title_end=time.time()
    # print("title: {}".format(title_end-title_start))

    # comment_start=time.time()
    # cn=commentcheck(comment_model, com)
    # comment_end=time.time()
    # print("comment: {}".format(comment_end-comment_start))

    # thumbnail_start=time.time()
    # thn=check_thumbnail(thumbnail_model, thumbnail)
    # thumbnail_end=time.time()
    # print("thumbnail: {}".format(thumbnail_end-thumbnail_start))
    output={}
    predict_threads=[
        threading.Thread(target=titlecheck, args=(title_model, title, output)),
        threading.Thread(target=commentcheck, args=(comment_model, com, output)),
        threading.Thread(target=check_thumbnail, args=(thumbnail_model, thumbnail, output))
    ]

    for thread in predict_threads:
        thread.start()

    for thread in predict_threads:
        thread.join()


    with open('ensemble_model.pkl', 'rb') as file:
        ensemble_model = pickle.load(file)

    inp=np.array([output['title_a'],output['title_b'],output['comment_a'],output['comment_b'],output['thumbnail_a']]).reshape(-1,5)
    flag=ensemble_model.predict(inp)

    ans={
        "nonclickbait/clickbait":str(flag[0]),
        "title":str(np.argmax([output['title_a'],output['title_b']])),
        "thumbnail": str(int(output['thumbnail_a']>0.5)),
        "comments": str(np.argmax([output['comment_a'],output['comment_b']]))
    }


    return make_response(jsonify(ans))


if __name__=="__main__":
    app.run(debug=False, threaded=True)