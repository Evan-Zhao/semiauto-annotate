import json
import os
import labelme
from labelme import ModelLoader
from labelme.conf import PREPROCESSED_YOLO_RESULT_PATH, PREPROCESSED_POSE_ESTM_RESULT_PATH


def get_image_list():
    result = labelme.mongodb_collection.find({}, {'filename': 1})
    for i in range(len(result)):
        filename = result[i]['filename']
        filename = str.split(filename, '/')
        filename = str.split(filename[len(filename) - 1], '.')[0]
        result[i] = filename
    return result

def add_prior_preprocess_task(img_id):
    result = labelme.mongodb_collection.find_one({'id': img_id})
    ModelLoader.add_prior_task([result['filename'], img_id, on_infer_complete()])


def get_collection_value(img_id, column):
    result = labelme.mongodb_collection.find_one({'id': img_id}, {column: 1})
    return result[column]


def get_image_path(img_id):
    result = labelme.mongodb_collection.find_one({'id': img_id})
    filename = result['filename']
    return filename


def modify_collection_row(img_id, column, content):
    result = labelme.mongodb_collection.find_one({'id': img_id})
    result[column] = content
    labelme.mongodb_collection.update({'id': img_id}, result)


def get_unpreprocessed_img():
    results = labelme.mongodb_collection.find({"preprocessed": False})
    return results


def on_infer_complete(results, img_id, filename):
    yolo = results[0]
    pose_estm = results[1]
    filename = str.split(filename, '/')
    filename = str.split(filename[len(filename) - 1], '.')[0]
    pose_estm_path = os.path.join(PREPROCESSED_POSE_ESTM_RESULT_PATH, filename + "_pose_estm.json")
    yolo_path = os.path.join(PREPROCESSED_YOLO_RESULT_PATH, filename + "_yolo.json")
    with open(pose_estm_path, "w") as fp1:
        fp1.write(pose_estm)
        fp1.flush()
    fp1.close()
    with open(yolo_path, 'w') as fp2:
        fp2.write(yolo)
        fp2.flush()
    fp2.close()
    modify_collection_row(img_id, 'preprocess_yolo', yolo_path)
    modify_collection_row(img_id, 'preprocess_pose_estm', pose_estm_path)
    modify_collection_row(img_id, 'preprocessed', True)


def get_incomplete_img():
    '''

    :return: img_id, [preprocessed_yolo_result, preprocessed_pose_estm_result]
    '''

    result = labelme.mongodb_collection.find_one({"$and":
                                                      [{'complete': False},
                                                       {'in_use': False},
                                                       {'preprocessed': True}]})
    while result is None:
        result = labelme.mongodb_collection.find_one({"$and":
                                                          [{'complete': False},
                                                           {'in_use': False},
                                                           {'preprocessed': True}]})
    return result['id'], get_preprocessed_result(result['preprocess_yolo'], result['preprocess_pose_estm'])


def get_preprocessed_result(preprocess_yolo_path, preprocess_pose_estm_path):
    with open(preprocess_pose_estm_path, 'r') as fp1:
        pose_estm = json.load(fp1)
    fp1.close()
    with open(preprocess_yolo_path, 'r') as fp2:
        yolo = json.load(fp2)
    fp2.close()
    return [yolo, pose_estm]