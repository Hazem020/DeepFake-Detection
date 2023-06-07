from flask import Flask, request, jsonify, render_template
from load_video import start
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/index.html', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


@app.route('/detection.html', methods=['POST', 'GET'])
def detection():
    return render_template('detection.html')


@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        f = request.files['video']
        video_path = 'storage/' + f.filename
        f.save(video_path)
        result = start(video_path)
        print(result)
        return jsonify(result), 200
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
