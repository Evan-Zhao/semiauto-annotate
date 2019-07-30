""" REST API for preprocessed image"""
import flask
import labelme

@labelme.app.route('/', methods=["GET"])
def get_preprocessed():
    """
    :return: if GET, return preprocessed image and json file
    """
    if flask.request.method == 'GET':
        img_id, [yolo_result, pose_estm_result] = labelme.model.get_incomplete_img()
        content = {'image_id': img_id,
                   "yolo_result": yolo_result,
                   "pose_estm_result": pose_estm_result}
        return flask.jsonify(**content)
    return '', 400
