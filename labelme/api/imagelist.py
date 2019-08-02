import flask
import labelme


@labelme.app.route('/imagelist/', methods=["GET"])
def get_image_list():
    """
    :return: image list [image1, image2, ...]
    """

    if flask.request.method == 'GET':
        image_list = labelme.model.get_image_list()
        return flask.jsonify(**image_list)
    return '', 400
