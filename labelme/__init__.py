import flask
import pymongo
from labelme.backend import ModelLoader
from labelme.config import get_config
from labelme.utils import Config
import labelme.conf

app = flask.Flask(__name__)
app.config.from_object('labelme.config')
app.config.from_envvar('LABELME_SETTINGS', silent=True)

# Register config to global position.
# see labelme/config/default_config.yaml for valid configuration
config = get_config()
Config.set_all(config)

print("Start")
ModelLoader.main_thread_ctor()
mongodb_client = pymongo.MongoClient(labelme.conf.MONGODB_URL)
mongodb_db = mongodb_client[labelme.conf.MONGODB_DATABASE]
mongodb_collection = mongodb_db[labelme.conf.MONGODB_COLLECTION]

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, use_reloader=False, debug=True)

import labelme.api
import labelme.backend
import labelme.shape
import labelme.utils
import labelme.model
