"""--Learn and Draw---"""
"""--adapted from: sketch-RNN --"""
"""--Mavis Brace--"""


"""
FOR TRAINING:
python sketch_rnn_train.py --log_root=models/house --data_dir=datasets/quickdraw/house --hparams="data_set=[house.npz],num_steps=20000,dec_model=hyper,dec_rnn_size=2048,enc_model=layer_norm,enc_rnn_size=512,save_every=5000,grad_clip=1.0,use_recurrent_dropout=0"
"""

# required libraries for machine learning aspect
import numpy as np
import time
import random
import cPickle
import codecs
import collections
import os
import math
import json
import tensorflow as tf
from six.moves import xrange

# libraries required for visualisation:
from IPython.display import SVG, display
import svgwrite # conda install -c omnia svgwrite=1.1.6
import PIL
#from PIL import Image
import image
import matplotlib.pyplot as plt

# libraries providing command line tools
from sketch_rnn_train import *
from model import *
from utils import *
from rnn import *

file_prefix = '/Users/mavisbrace/Desktop'
image_prefix = "//Users//mavisbrace//Desktop//understanding_place//svg//"

"""------------------------------------------------"""

# this function displays vector images and saves them to .svg
#def draw_strokes(data, factor=0.2, svg_filename = '/tmp/sketch_rnn/svg/sample.svg'):
#def draw_strokes(data, factor=0.2, svg_filename = file_prefix + '/sketch_rnn/svg/sample.svg'):
def draw_strokes(data, factor=0.2, svg_filename= file_prefix + '//sketch_rnn//svg//temp_sample.svg'):
  tf.gfile.MakeDirs(os.path.dirname(svg_filename))
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white',fill_opacity=0))
  lift_pen = 1
  abs_x = 25 - min_x
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in xrange(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  display(SVG(dwg.tostring()))

# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):
  def get_start_and_end(x):
    x = np.array(x)
    x = x[:, 0:2]
    x_start = x[0]
    x_end = x.sum(axis=0)
    x = x.cumsum(axis=0)
    x_max = x.max(axis=0)
    x_min = x.min(axis=0)
    center_loc = (x_max+x_min)*0.5
    return x_start-center_loc, x_end
  x_pos = 0.0
  y_pos = 0.0
  result = [[x_pos, y_pos, 1]]
  for sample in s_list:
    s = sample[0]
    grid_loc = sample[1]
    grid_y = grid_loc[0]*grid_space+grid_space*0.5
    grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
    start_loc, delta_pos = get_start_and_end(s)

    loc_x = start_loc[0]
    loc_y = start_loc[1]
    new_x_pos = grid_x+loc_x
    new_y_pos = grid_y+loc_y
    result.append([new_x_pos-x_pos, new_y_pos-y_pos, 0])

    result += s.tolist()
    result[-1][2] = 1
    x_pos = new_x_pos+delta_pos[0]
    y_pos = new_y_pos+delta_pos[1]
  return np.array(result)

"""----------------------------------------------------"""

def encode(eval_model, sess, input_strokes):
  strokes = to_big_strokes(input_strokes).tolist()
  strokes.insert(0, [0, 0, 1, 0, 0])
  seq_len = [len(input_strokes)]
  draw_strokes(to_normal_strokes(np.array(strokes)))
  return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]

def decode(eval_model, sample_model, sess, z_input=None, draw_mode=True, temperature=0.1, factor=0.2):
  z = None
  if z_input is not None:
    z = [z_input]
  sample_strokes, m = sample(sess, sample_model, seq_len=eval_model.hps.max_seq_len, temperature=temperature, z=z)
  strokes = to_normal_strokes(sample_strokes)
  if draw_mode:
    draw_strokes(strokes, factor)
  return strokes

"""-----------------------------------------------------"""

def search_files(all_data, img_counter):
  print("Searching for new data...")
  print(img_counter)
  #all_data.close()  #I close this below
  all_data = open("all_data.txt",'r')
  data_found = False
  light_value = 0.0
  #audio_found = False
  audio_value = 0.0
  #movement_found = False
  movement_value = 0.0
  all_lines = all_data.readlines()
  all_data.close()

  if len(all_lines) >= img_counter+1:
      print("Found new data.")
      try:
          line_list = all_lines[img_counter].split()
          return (line_list)
      except:
          Print("Error retrieving new data.")
          return None
  else:
      print("No new data found.")
      return None


  #saveterm = "%ssample%04d.svg" %(image_prefix, img_counter)

"""===================================="""


def main():
    img_counter = 0
    t1 = random.uniform(0,10)
    #t2 = random.uniform(0,1)
    t2 = random.uniform(0,10)
    tCombine = random.uniform(0,1) # - start from reasonable value, but is allowed to go anywheres
    print("--Starting temperatures are " + str(t1) + " and " + str(t2))
    print("--Combine temperature is starting at " + str(tCombine))
    temperatures = [t1,t2]  # beginning data is random; in loop system learns and expands on it
    print("--opening files...")
    #(audio_data, movement_data) = open_files()
    all_data = open("all_data.txt",'r')
    # set numpy output
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
    tf.logging.info("TensorFlow Version: %s", tf.__version__)
    print("--setup complete.")

    print("--LOOP BEGIN--")
    while True:
      # luminance, movement, audio
      value_list = search_files(all_data, img_counter)

      if value_list is not None and len(value_list) == 3:
        light_value = value_list[0]
        movement_value = float(value_list[1])
        print("MOVMENT VALUE: " + str(movement_value))
        audio_value = value_list[2]
        """-------------------machine creates sketch--------------------"""
        # construct the sketch-rnn model here:
        model_dir_house = file_prefix + '/understanding_place/models/house'
        model_dir_ocean = file_prefix + '/understanding_place/models/ocean'
        # -- PICK ONE MAIN MODEL THAT SUIT THE DATA BEST (TODO) -- #
        # -- PICK A SECONDARY MODEL (TODO) -- #

        model_dirs = [model_dir_house, model_dir_ocean]

        drawings = []

        for i in range(0,2):
          model_dir = model_dirs[i]
          [hps_model, eval_hps_model, sample_hps_model] = load_model(model_dir)
          # --GRAPH / MODEL SETUP -- #
          reset_graph()
          model = Model(hps_model)
          eval_model = Model(eval_hps_model, reuse=True)
          sample_model = Model(sample_hps_model, reuse=True)
          # --BEGINNING TENSORFLOW SESSION-- #
          sess = tf.InteractiveSession()
          sess.run(tf.global_variables_initializer())
          # --LOADING WEIGHTS -- #
          load_checkpoint(sess, model_dir)
          the_drawing = np.random.randn(eval_model.hps.z_size)
          _ = decode(eval_model, sample_model, sess, the_drawing, temperature=temperatures[i])

          drawings.append(the_drawing)

        drawings_list = []
        N = 10
        # --combining drawings using linear interpolation -- #
        for t in np.linspace(0,1,N):
          drawings_list.append(slerp(drawings[0],drawings[1],t))
        reconstructions = []
        for i in range(N):
          reconstructions.append([decode(eval_model, sample_model, sess, drawings_list[i], draw_mode=False, temperature=tCombine), [0,i]])
        # temperature WAS 0.1

        # testing: saving more reconstructions (because it takes so long to make 1)
        # --SAVING TO SVG FILE-- #
        #first, make sub-directory
        dir_name = "%ssketch%04d" %(image_prefix, img_counter)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        num = int(movement_value * N)
        for i in range(num):
            saveterm = ""
            saveterm = saveterm + "%s//sample%i.svg" %(dir_name,i)
            print("--[ " + saveterm + " ]--")
            draw_strokes((reconstructions[i])[0], svg_filename = saveterm)

        print("---new drawing saved.")
        img_counter = img_counter + 1


        """-----------ELSE if no new data found, wait for a bit----------"""
      else:
        time.sleep(10) #delay 10 seconds
        print("--delay finished, looking for new data...")


if __name__ == '__main__':
    main()
