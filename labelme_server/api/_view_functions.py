import os

import flask
from flask import Blueprint

from .. import conf
from ..backend import database

main_api = Blueprint('main_api', __name__)

ALLOWED_EXTENSIONS = set('json')


# @main_api.route('/image/<string:img_id>/', methods=["GET"])
# def get_images(img_id):
#     """
#     :param img_id: image id
#     :return: image
#     """
#
#     if flask.request.method == 'GET':
#         filename = database.get_image_path(img_id)
#         return flask.send_file(filename)
#
#     return '', 400


@main_api.route('/imagelist/', methods=["GET"])
def get_image_list():
    """
    :return: image list [image1, image2, ...]
    """

    if flask.request.method == 'GET':
        image_list = database.get_image_list()
        return flask.jsonify(**image_list)
    return '', 400


# @main_api.route('/', methods=["GET"])
# def get_preprocessed():
#     """
#     :return: if GET, return preprocessed image and json file
#     """
#     if flask.request.method == 'GET':
#         img_id, [yolo_result, pose_estm_result] = database.get_incomplete_img()
#         database.modify_collection_row(img_id, 'in_use', True)
#         content = {'image_id': img_id,
#                    "yolo_result": yolo_result,
#                    "pose_estm_result": pose_estm_result}
#         return flask.jsonify(**content)
#     return '', 400


# @main_api.route('/<string:img_id>', methods=["GET"])
@main_api.route('/image/<string:img_id>/', methods=["GET"])
def get_preprocessed(img_id):
    """
    :return: if GET, return preprocessed image and json file
    """
    if flask.request.method == 'GET':
        if not database.get_collection_value(img_id, 'preprocessed'):
            database.add_prior_preprocess_task(img_id)
        while not database.get_collection_value(img_id, 'preprocessed'):
            pass

        database.modify_collection_row(img_id, 'in_use', True)
        yolo_result_path = database.get_collection_value(img_id, 'preprocess_yolo')
        pose_estm_result_path = database.get_collection_value(img_id, 'preprocess_pose_estm')
        [yolo_result, pose_estm_result] = database.get_preprocessed_result(yolo_result_path,
                                                                           pose_estm_result_path)
        content = {'image_id': img_id,
                   "yolo_result": yolo_result,
                   "pose_estm_result": pose_estm_result}
        return flask.jsonify(**content)
    return '', 400


def allowed_file(filename):
    return '.' in filename and str.split(filename, '.')[1].lower() in ALLOWED_EXTENSIONS


@main_api.route('/upload/<string:img_id>/', methods=["POST"])
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
        file.save(os.path.join(conf.UPLOAD_PATH, filename))
        database.modify_collection_row(img_id, 'in_use', False)
        database.modify_collection_row(img_id, 'complete', True)
        database.modify_collection_row(img_id, 'complete_result',
                                       os.path.join(conf.UPLOAD_PATH, filename))
        return '', 200
    return '', 400
