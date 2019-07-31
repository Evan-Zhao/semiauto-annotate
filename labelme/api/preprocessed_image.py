""" REST API for preprocessed image"""
import flask
import labelme

@labelme.app.route('/<string:img_id>', methods=["GET"])
def get_preprocessed(img_id):
    """
    :return: if GET, return preprocessed image and json file
    """
    if flask.request.method == 'GET':
        if not labelme.model.get_collection_value(img_id, 'preprocessed'):
            labelme.model.add_prior_preprocess_task(img_id)
        while not labelme.model.get_collection_value(img_id, 'preprocessed'):
            pass

        labelme.model.modify_collection_row(img_id, 'in_use', True)
        yolo_result_path = labelme.model.get_collection_value(img_id, 'preprocess_yolo')
        pose_estm_result_path = labelme.model.get_collection_value(img_id, 'preprocess_pose_estm')
        [yolo_result, pose_estm_result] = labelme.model.get_preprocessed_result(yolo_result_path, pose_estm_result_path)
        content = {'image_id': img_id,
                   "yolo_result": yolo_result,
                   "pose_estm_result": pose_estm_result}
        return flask.jsonify(**content)
    return '', 400
