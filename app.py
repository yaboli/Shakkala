from flask import Flask, request, jsonify
from Shakkala import Shakkala

app = Flask(__name__)
sh = Shakkala('./', version=3)
model, graph = sh.get_model()


@app.route("/")
def hello():
    return "<h1 style='color:blue'>Shakkala API</h1>"


@app.route('/vocalize', methods=['POST'])
def vocalize():
    request_json = request.json
    input_text = request_json['text']
    length = len(input_text.split(' '))
    input_int = sh.prepare_input(input_text)
    with graph.as_default():
        logits = model.predict(input_int)[0]
    predicted_harakat = sh.logits_to_text(logits)
    final_output = sh.get_final_text(input_text, predicted_harakat)
    return jsonify(result=final_output, length=length), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8085)
