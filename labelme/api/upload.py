""" REST API for upload result"""
import os

import flask
import labelme

ALLOWED_EXTENSIONS = set('json')


def allowed_file(filename):
    return '.' in filename and str.split(filename, '.')[1].lower() in ALLOWED_EXTENSIONS


@labelme.app.route('/upload/<string:img_id>/', methods=["POST"])
def upload(img_id):
    """
    :return: status code
    """
    if flask.request.method == 'POST':
        if 'file' not in flask.request.files:
            return '', 400
        file = flask.request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return '', 400
        filename = file.filename
        file.save(os.path.join(labelme.config.UPLOAD_PATH, filename))
        labelme.model.modify_collection_row(img_id, 'in_use', False)
        labelme.model.modify_collection_row(img_id, 'complete', True)
        return '', 200
    return '', 400
