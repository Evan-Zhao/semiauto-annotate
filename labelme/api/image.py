import flask
import labelme


@labelme.app.route('/image/<string:img_id>/', methods=["GET"])
def get_images(img_id):
    """
    :param img_id: image id
    :return: image
    """

    if flask.request.method == 'GET':
        filename = labelme.model.get_image_path(img_id)
        return flask.send_file(filename)

    return '', 400
