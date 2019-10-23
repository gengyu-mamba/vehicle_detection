import argparse
import os,sys
import logging
sys.path.append('object_detection')
sys.path.append('slim')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from preprocessing import preprocessing_factory
from nets import nets_factory

from utils import visualization_utils as vis_util
from utils import label_map_util

import datetime
from flask import request, send_from_directory
from flask import Flask, request, redirect, url_for
import uuid
import math
import cv2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', './faster_rcnn_nas_lowproposals_coco_2018_01_28', '')
tf.app.flags.DEFINE_string('slim_checkpoint_dir', './slim', '')
tf.app.flags.DEFINE_string('dataset_dir', './data_quiz_car', '')
tf.app.flags.DEFINE_string('output_dir', './output', '')
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")
tf.app.flags.DEFINE_string('label_dir', './slim', '')
tf.app.flags.DEFINE_integer('port', '5001', '')
tf.app.flags.DEFINE_boolean('debug', False, '')
tf.app.flags.DEFINE_string('upload_folder', '/tmp/', '')
tf.app.flags.DEFINE_string('f', '', 'kernel')

NUM_CLASSES = 90
MAX_INFERENCE_NUM = 15
UPLOAD_FOLDER = FLAGS.upload_folder

app = Flask(__name__)
app._static_folder = FLAGS.upload_folder

network_fn = nets_factory.get_network_fn('inception_v3',
                                        num_classes=764,
                                        is_training=False)
image_size = network_fn.default_image_size
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            'inception_v3',
            is_training=False)

img_path = os.path.join(FLAGS.dataset_dir, 'car.jpg')  #image输入 
PATH_DETECTION_CKPT = os.path.join(FLAGS.checkpoint_dir, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'mscoco_label_map.pbtxt')
PATH_CLASSIFY_CKPT = os.path.join(FLAGS.slim_checkpoint_dir, 'freezen_graph.pb')
LABEL = os.path.join(FLAGS.label_dir,'labels.txt')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG', 'png'])

# print(category_index)

graph = tf.Graph()
with graph.as_default():
    starttime = datetime.datetime.now()
    detection_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_DETECTION_CKPT,'rb') as f:
        detection_graph_def.ParseFromString(f.read())
        tf.import_graph_def(detection_graph_def, name='')
        detection_sess = tf.Session(graph=graph)
        endtime = datetime.datetime.now()
        print (endtime - starttime)
    #载入分类模型pb文件
    starttime = datetime.datetime.now()
    classify_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_CLASSIFY_CKPT, 'rb') as f:
        classify_graph_def.ParseFromString(f.read())
        tf.import_graph_def(classify_graph_def, name='')
        classify_sess = tf.Session(graph=graph)
        endtime = datetime.datetime.now()
        print (endtime - starttime)


def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def rename_filename(old_file_name):
    basename = os.path.basename(old_file_name)
    name, ext = os.path.splitext(basename)
    new_name = str(uuid.uuid1()) + ext
    return new_name

#定义将image图片转成numpyarray函数
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

#做一个中央裁切，用在分类模型上的函数
def preprocess_image(image, height, width,
                        central_fraction=0.875, scope=None):
  image = load_image_into_numpy_array(image)
  image = np.true_divide(image,255).astype('float32')

  H,W,_ = image.shape
  x = W/2
  y = H/2
  winW = W*central_fraction/2
  winH = H*central_fraction/2
  image = image[math.floor(y-winH):math.ceil(y + winH),math.floor(x-winW):math.ceil(x + winW)]

  image = cv2.resize(image, (height, width),interpolation=cv2.INTER_LINEAR)
  image = (image-0.5)*2
  return image

# def inference_detect(img_path,graph,sess):
#     starttimeall = datetime.datetime.now()
#     imageArr = []
#     with graph.as_default():
#         image_tensor = graph.get_tensor_by_name('image_tensor:0')
#         detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
#         detection_scores = graph.get_tensor_by_name('detection_scores:0')
#         detection_classes = graph.get_tensor_by_name('detection_classes:0')
#         num_detections = graph.get_tensor_by_name('num_detections:0')
#         image = Image.open(img_path)
#         image_np = load_image_into_numpy_array(image)
#         image_np_expanded = np.expand_dims(image_np, axis=0)
#         starttime = datetime.datetime.now()
#         print('before sess.run')
#         (boxes, scores, classes, num) = sess.run(
#             [detection_boxes, detection_scores, detection_classes, num_detections],
#             feed_dict={image_tensor: image_np_expanded})
#         print('after sess.run')
#         endtime = datetime.datetime.now()
#         print (endtime - starttime)
#         boxes_squeeze = np.squeeze(boxes)
#         classes_squeeze = np.squeeze(classes)
#         '''vis_util.visualize_boxes_and_labels_on_image_array(   #可视化
#             image_np,
#             boxes_squeeze,
#             classes_squeeze.astype(np.int32),
#             np.squeeze(scores),
#             category_index,
#             use_normalized_coordinates=True,
#             line_thickness=8)
#         plt.imsave(os.path.join(FLAGS.output_dir, 'output.png'), image_np)#输出图片'''
#     return boxes_squeeze,classes_squeeze,image

# (box,classes,image) = inference_detect(img_path,graph,detection_sess)

# print ('box_squeeze',box)
# print('classes_squeeze',classes)

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_path=None):
    if not label_path:
      tf.logging.fatal('please specify the label file.')
      return
    self.node_lookup = self.load(label_path)

  def load(self, label_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(label_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(label_path).readlines()
    id_to_human = {}
    for line in proto_as_ascii_lines:
      if line.find(':') < 0:
        continue
      _id, human = line.rstrip('\n').split(':')
      id_to_human[int(_id)] = human

    return id_to_human

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

def run_inference_on_image(imageArr, sess, boxnum):
    softmax_tensor = sess.graph.get_tensor_by_name('final_probs:0')
    predictions = sess.run(softmax_tensor,
                         {'input:0': imageArr})
    print("predictions:",predictions)
    node_lookup = NodeLookup(LABEL)

    carnames = []
    scores = []
    classids = []
    if boxnum == 1:
        prediction = np.array(predictions)
        for i in range(1):
            top_k = prediction.argsort()[-FLAGS.num_top_predictions:][::-1]
            print('top_k:')
            print(top_k)
            node_id = top_k[0]
            print('node_id')
            print(node_id)
        
            score = prediction[node_id]

            carname = node_lookup.id_to_string(node_id)
        
            scores.append(score) 
            classids.append(node_id)
            carnames.append(carname)
            print('id:[%d] name:[%s] (score = %.5f)' % (node_id, carname, score))
    else:
        print("predictions.shape")
        print(predictions.shape)
        for i in range(predictions.shape[0]):
            prediction = predictions[i]
            print("predictions[i]:",predictions[i])
            top_k = prediction.argsort()[-FLAGS.num_top_predictions:][::-1]
            print('top_k:')
            print(top_k)
            node_id = top_k[0]
            print('node_id')
            print(node_id)
        
            score = prediction[node_id]

            carname = node_lookup.id_to_string(node_id)
        
            scores.append(score) 
            classids.append(node_id)
            carnames.append(carname)
            print('id:[%d] name:[%s] (score = %.5f)' % (node_id, carname, score))
    print(carnames)
    print(scores)
    return carnames, scores, classids

def inference_detect(img_path,graph,sess):
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
    image = Image.open(img_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    starttime = datetime.datetime.now()
    print('before sess.run')
    (boxes, classes) = sess.run(
        [detection_boxes, detection_classes],
        feed_dict={image_tensor: image_np_expanded})
    print('after sess.run')
    endtime = datetime.datetime.now()
    print (endtime - starttime)
    
    boxes_squeeze = np.squeeze(boxes)
    classes_squeeze = np.squeeze(classes)
    
    boxnum = 0
    newboxes = []
    for i in range(len(classes_squeeze)):
        if classes_squeeze[i] == 3 or classes_squeeze[i] == 6 or classes_squeeze[i] == 8:
            newboxes.append(boxes_squeeze[i])
            boxnum += 1
            if boxnum >= MAX_INFERENCE_NUM:
                break
    print(classes_squeeze)
    print(boxnum)
    print(newboxes)
    return image, image_np, newboxes, boxnum

#检测和分类结合部分
def inference(img_path,graph,sess1,sess2):
    starttimeall = datetime.datetime.now()
    imageArr = []
    with graph.as_default():
      image, image_np, newboxes, boxnum = inference_detect(img_path,graph,sess1)
      if len(newboxes) > 0:
          starttimett = datetime.datetime.now()
          for i in range(len(newboxes)):
              bbox = (image.size[0]*newboxes[i][1], image.size[1]*newboxes[i][0], image.size[0]*newboxes[i][3], image.size[1]*newboxes[i][2])
              bbox=tuple(bbox)
              try:
                  starttime = datetime.datetime.now()
                  newim=image.crop(bbox)      #裁剪图片
                  endtime = datetime.datetime.now()
                  print('finish image.crop(bbox)')
                  print (endtime - starttime)

                  starttime = datetime.datetime.now()
                  newim = preprocess_image(newim, image_size, image_size)                   
                  endtime = datetime.datetime.now()
                  
                  print('finish image_preprocessing_fn')
                  print (endtime - starttime)
                  
                  starttime = datetime.datetime.now()
                  imageArr.append(newim)      #裁剪后的图片添加到列表
                  endtime = datetime.datetime.now()
                  print('finish newim.eval(session=sess1)')
                  print (endtime - starttime)

              except SystemError:
                      print("crop Error")
          endtimett = datetime.datetime.now()
          print('total process img')
          print (endtimett - starttimett)

          print('befor run_inference_on_image')
          starttime = datetime.datetime.now()
          #导入分类模型as classes
          carnames, scores, classids = run_inference_on_image(imageArr, sess2, boxnum)
          endtime = datetime.datetime.now()
          print('after run_inference_on_image')
          print (endtime - starttime)
          
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              newboxes,
              classids,
              scores,
              category_index,
              use_normalized_coordinates=True,
              line_thickness=1,
              carnames = carnames)
          plt.imsave(os.path.join(UPLOAD_FOLDER, img_path), image_np)
          new_url = '/static/%s' % os.path.basename(img_path)
          print(new_url)
          image_tag = '<img src=%s></img><p>'
          new_tag = image_tag % new_url
          
          format_string = '<b>图片中有以下车型：</b><br/>'
          for score, carname in zip(scores, carnames):
              format_string += '%s (相似度:%.5f&#37;)<BR>' % (carname, score*100)
          
          ret_string = new_tag  + format_string + '<BR>' 
      else:
          new_url = '/static/%s' % os.path.basename(img_path)
          image_tag = '<img src=%s></img><p>'
          new_tag = image_tag % new_url
          
          ret_string = new_tag  + '<b>图片中没有发现任何汽车</b><br/><BR>' 
            
    endtimeall = datetime.datetime.now()
    print('all inference')
    print (endtimeall - starttimeall)
    ret_string = ret_string + '<b>预测耗时：%s</b><br/><BR>' % (endtimeall - starttimeall)
    return ret_string
            
inference_detect(img_path,graph,detection_sess)
# inference(img_path,graph,detection_sess,classify_sess)

@app.route("/", methods=['GET', 'POST'])
def root():
  result = """
    <!doctype html>
    <title>车辆检测项目</title>
    <h1>请上传一张需要检测汽车的照片吧！</h1>
    
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file value='选择图片'>
      <br/><br/>
         <input type=submit value='上传预测'>
    </form>
    <p>%s</p>
    """ % "<br>"

  if request.method == 'POST':
    file = request.files['file']
    old_file_name = file.filename
    if file and allowed_files(old_file_name):
      filename = rename_filename(old_file_name)
      file_path = os.path.join(UPLOAD_FOLDER, filename)
      file.save(file_path)
      type_name = 'N/A'
      print('file saved to %s' % file_path)
    out_html = inference(file_path,graph,detection_sess,classify_sess)
    return result + out_html 
  return result

print('listening on port %d' % FLAGS.port)
app.run(host='127.0.0.1', port=FLAGS.port, debug=FLAGS.debug, threaded=True)


