from flask import Flask, render_template, request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import caption

app = Flask(__name__)

def analysetext(processtext):
    sid = SentimentIntensityAnalyzer()
    output = sid.polarity_scores(processtext)
    return output

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/',methods = ['POST'])
def captioning():
    if request.method == 'POST':
        f = request.files['userfile']
        path = './static/{}'.format(f.filename)
        f.save(path)

        cap = caption.caption_this_image(path)
        result_dic = {
            'image':path,
            "caption":cap
        }
        senti = analysetext(cap)
        if senti['compound'] >= 0.05 :
            output="Positive"
    
        elif senti['compound'] <= - 0.05 :
            output="Negative"
        else :
            output="Neutral"
    
    return render_template("index.html",your_result = result_dic,output=output,
                                       neg="{0:.2f}".format(senti['neg']*100), neu="{0:.2f}".format(senti['neu']*100), pos="{0:.2f}".format(senti['pos']*100))



if __name__ == '__main__':
    app.run(debug=True)
