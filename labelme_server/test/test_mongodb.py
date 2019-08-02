import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["joe"]
mycol = mydb["test3"]
'''
test = [{
    'id': '1', 'filename': '/home/joezhong/PycharmProjects/labelme_client-server/vars/test_images/timg.jpeg', 'timestamp': '12',
    'in_use': False, 'complete': False, 'preprocessed': False, 'preprocess_yolo':None, 'preprocess_pose_estm':None}, {
    'id': '2', 'filename': '/home/joezhong/PycharmProjects/labelme_client-server/vars/test_images/sample.jpeg',
    'timestamp': '13', 'in_use': False, 'complete': False, 'preprocessed': False, 'preprocess_yolo':None, 'preprocess_pose_estm':None}]
mycol.insert_many(test)
'''
for x in mycol.find():
    print(x)

print(mycol.find_one({"$and":
                          [{'complete': False},
                           {'in_use': False}]}))
print(mydb.collection_names())
