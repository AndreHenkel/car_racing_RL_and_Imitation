 #taken from https://raw.githubusercontent.com/gui-miotto/DeepLearningLab/master/Assignment%2003/Code/utils.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import multiprocessing as mp
from skimage.color import rgb2gray
from skimage import filters

import pickle,os,gzip
import cv2

# Tells how long should the history be.
# Altering this variable has effects on ALL modules
history_length = 1
# Number of first states of each episode that shall be ignored
# from the expert dataset:
dead_start = 0 #was 50
# Set of allowed actions:
actions = np.array([
    [ 0.0, 0.0, 0.0],  # STRAIGHT
    [ 0.0, 1.0, 0.0],  # ACCELERATE
    [ 1.0, 0.0, 0.0],  # RIGHT
    [ 1.0, 0.0, 0.4],  # RIGHT_BRAKE
    [ 0.0, 0.0, 0.4],  # BRAKE
    [-1.0, 0.0, 0.4],  # LEFT_BRAKE
    [-1.0, 0.0, 0.0],  # LEFT
], dtype=np.float32)
n_actions = len(actions)


def show_img(img_arr, c='gray'):
    if c=='gray':
        if len(img_arr.shape)==3:#reshapes image array from (1,X,Y) to (X,Y)
            img_arr = np.reshape(img_arr,(img_arr.shape[1],img_arr.shape[2]))
        plt.imshow(img_arr,cmap='gray')
    else:
        plt.imshow(img_arr)
    plt.show()


def show_run(img_arr_list, c='gray'):
    """ Expects an array of the form (n,96,96) as greyscale """
    #create two subplots
    ax1 = plt.subplot(1,2,1)
    im1 = 0
    if c=='gray':
        im1 = ax1.imshow(img_arr[0],cmap='gray')
    else:
        im1 = ax1.imshow(img_arr[0],cmap='gray')
    plt.ion()
    for i in range(img_arr.shape[0]):
        im1.set_data(img_arr[i])
        plt.pause(0.2)

    plt.ioff()
    plt.show()

def store_video(img_arr_list, w=96, h=96):
    # initialize video writer
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 50
    video_filename = 'output.avi'
    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    for i in range(len(img_arr_list)):
        # frame = cv2.normalize(img_arr_list[i], None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # gray_3c = cv2.merge([frame, frame, frame])
        out.write(img_arr_list[i])

    out.release()
    print("Video was written to: ",video_filename )

def read_data(path):
    """Reads the states and actions recorded by drive_manually.py"""
    print("Reading data")
    with gzip.open(path,'rb') as f:
        data = pickle.load(f)
    states = vstack(data["state"])
    actions = vstack(data["action"])
    reward = vstack(data["reward"],is_nr=True)
    next_states = vstack(data["next_state"])
    done = vstack(data["terminal"],is_nr=True)
    return states, actions, reward, next_states, done


def action_arr2id(arr):
    """ Converts action from the array format to an id (ranging from 0 to n_actions) """
    ids = []
    for a in arr:
        id = np.where(np.all(actions==a, axis=1))
        ids.append(id[0][0])
    return np.array(ids)

def action_id2arr(ids):
    """ Converts action from id to array format (as understood by the environment) """
    return actions[ids]

def one_hot(labels):
    """ One hot encodes a set of actions """
    one_hot_labels = np.zeros(labels.shape + (n_actions,))
    for c in range(n_actions):
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def unhot(one_hot_labels):
    """ One hot Decodes a set of actions """
    return np.argmax(one_hot_labels, axis=1)

def transl_action_env2agent(acts):
    """ Translate actions from environment's format to agent's format """
    act_ids = action_arr2id(acts)
    return one_hot(act_ids)

def transl_action_agent2env(acts):
    """ Translate actions from agent's format to environment's format """
    act_arr = action_id2arr(acts)
    return act_arr[0]

def check_invalid_actions(y):
    """ Check if there is any forbidden actions in the expert database """
    inval_actions = [
        [ 0.0, 1.0, 0.4],  # ACCEL_BRAKE
        [ 1.0, 1.0, 0.4],  # RIGHT_ACCEL_BRAKE
        [-1.0, 1.0, 0.4],  # LEFT_ACCEL_BRAKE
        [ 1.0, 1.0, 0.0],  # RIGHT_ACCEL
        [-1.0, 1.0, 0.0],  # LEFT_ACCEL
    ]
    ia_count = 0
    for ia in inval_actions:
        ia_count += np.sum(np.all(y == ia, axis=1))
    if ia_count > 0:
        raise Exception('Invalid actions. Do something developer!')

def balance_actions(X, y, drop_prob):
    """ Balance samples. Gets hide of a share of the most common action (accelerate) """
    # Enconding of the action accelerate
    acceler = np.zeros(7)
    acceler[0] = 1. #changed to straight, since this is the most used action
    # Find out what samples are labeled as accelerate
    is_accel = np.all(y==acceler, axis=1)
    # Get the index of all other samples (not accelerate)
    other_actions_index = np.where(np.logical_not(is_accel))
    # Randomly pick drop some accelerate samples. Probabiliy of dropping is given by drop_prob
    drop_mask = np.random.rand(len(is_accel)) > drop_prob
    accel_keep = drop_mask * is_accel
    # Get the index of accelerate samples that were kept
    accel_keep_index = np.where(accel_keep)
    # Put all actions that we want to keep together
    final_keep = np.squeeze(np.hstack((other_actions_index, accel_keep_index)))
    final_keep = np.sort(final_keep)
    X_bal, y_bal = X[final_keep], y[final_keep]

    return X_bal, y_bal



def preprocess_state(states):
    """ Preprocess the images (states) of the expert dataset before feeding them to agent """
    states_pp = np.copy(states)

    # Paint black over the sum of rewards
    states_pp[:, 85:, :15] = [0.0, 0.0, 0.0]

    # Replace the colors defined bellow
    def replace_color(old_color, new_color):
        mask = np.all(states_pp == old_color, axis=3)
        states_pp[mask] = new_color

    # Black bar
    replace_color([000., 000., 000.], [120.0, 120.0, 120.0])

    # Road
    #new_road_color = [255.0, 255.0, 255.0]
    new_road_color = [102.0, 102.0, 102.0]
    replace_color([102., 102., 102.], new_road_color)
    replace_color([105., 105., 105.], new_road_color)
    replace_color([107., 107., 107.], new_road_color)
    # Curbs
    replace_color([255., 000., 000.], new_road_color)
    replace_color([255., 255., 255.], new_road_color)
    # Grass
    #new_grass_color = [0.0, 0.0, 0.0]
    new_grass_color = [102., 229., 102.]
    replace_color([102., 229., 102.], new_grass_color)
    replace_color([102., 204., 102.], new_grass_color)

    # Float RGB represenattion
    states_pp /= 255.

    # Converting to gray scale
    states_pp = rgb2gray(states_pp)

    return states_pp

def stack_history(X, y, N, shuffle=True):
    """ Stack states from the expert database into volumes of depth=history_length """
    x_stack = [X[i - N : i] for i in range(N, len(X)+1)]
    x_stack = np.moveaxis(x_stack, 1, -1)
    y_stack = y[N-1:]
    if shuffle:
        order = np.arange(len(x_stack))
        np.random.shuffle(order)
        x_stack = x_stack[order]
        y_stack = y_stack[order]
    return x_stack, y_stack


def vstack(arr, is_nr = False):
    """
    Expert database is divided by episodes.
    This function stack all those episodes together but discarding the
    first dead_start samples of every episode.
    """
    #determine smallest set
    # min = len(arr[0])
    # for i in range(0, len(arr)):
    #     if len(arr[i]) < min:
    #         min=len(arr[i])
    if is_nr:
        for i in range(0, len(arr)):
            arr[i] = np.reshape(np.asarray(arr[i]),(len(arr[i]),1))

    #print(np.asarray(arr[0]).shape)
    stack = np.array(arr[0][dead_start:], dtype=np.float32)
    for i in range(1, len(arr)):
        stack = np.vstack((stack, arr[i][dead_start:]))
    return stack

def conc_states(state_list):
    """ Assuming state shape (Channels,Width,Height)
        Returning single state element of form (1,Channels*len(state_list),Width,Height)
    """
    c_s = np.concatenate(np.asarray(state_list), axis=0)
    c_s = np.reshape(c_s,(1,len(state_list),state_list[0].shape[1],state_list[0].shape[2]))
    return c_s
