import flask

from labelme_server import conf
from labelme_server.api import main_api

app = flask.Flask(__name__)
app.config.from_object(conf)
# app.config.from_envvar('LABELME_SETTINGS', silent=True)
app.register_blueprint(main_api)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, use_reloader=False, debug=True)
