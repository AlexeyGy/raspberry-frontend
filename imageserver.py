import os
from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('images.html', images=os.listdir('static'))


if __name__ == '__main__':
    app.run(port=5001, host='0.0.0.0')
