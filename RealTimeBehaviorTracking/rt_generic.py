"""
Provided by Lindsay Willmore, Princeton University
Based on DeepLabCut and Spinnaker-Python API

Everywhere "your" is found in a commentbelow, 
please change file paths or parameters to suit your experiment.
"""

import os.path
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
import time
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from deeplabcut.utils import auxiliaryfunctions
import cv2
from skimage.util import img_as_ubyte
import PySpin
import matplotlib.pyplot as plt
import serial
import pickle

## CAMERA HELPER FUNCTIONS
class TriggerType:
    SOFTWARE = 1
    HARDWARE = 2


CHOSEN_TRIGGER = TriggerType.SOFTWARE


def configure_trigger(cam):
    """
    This function configures the camera to use a trigger. First, trigger mode is
    set to off in order to select the trigger source. Once the trigger source
    has been selected, trigger mode is then enabled, which has the camera
    capture only a single image upon the execution of the chosen trigger.

     :param cam: Camera to configure trigger for.
     :type cam: CameraPtr
     :return: True if successful, False otherwise.
     :rtype: bool
    """
    result = True

    print('*** CONFIGURING TRIGGER ***\n')

    if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
        print('Software trigger chosen ...')
    elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
        print('Hardware trigger chose ...')

    try:
        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        nodemap = cam.GetNodeMap()
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

        print('Trigger mode disabled...')

        # Select trigger source
        # The trigger source must be set to hardware or software while trigger
        # mode is off.
        node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        if not PySpin.IsAvailable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
            print('Unable to get trigger source (node retrieval). Aborting...')
            return False

        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            node_trigger_source_software = node_trigger_source.GetEntryByName('Software')
            if not PySpin.IsAvailable(node_trigger_source_software) or not PySpin.IsReadable(
                    node_trigger_source_software):
                print('Unable to set trigger source (enum entry retrieval). Aborting...')
                return False
            node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())

        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            node_trigger_source_hardware = node_trigger_source.GetEntryByName('Line0')
            if not PySpin.IsAvailable(node_trigger_source_hardware) or not PySpin.IsReadable(
                    node_trigger_source_hardware):
                print('Unable to set trigger source (enum entry retrieval). Aborting...')
                return False
            node_trigger_source.SetIntValue(node_trigger_source_hardware.GetValue())

        # Turn trigger mode on
        # Once the appropriate trigger source has been set, turn trigger mode
        # on in order to retrieve images using the trigger.
        node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
        if not PySpin.IsAvailable(node_trigger_mode_on) or not PySpin.IsReadable(node_trigger_mode_on):
            print('Unable to enable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())
        print('Trigger mode turned back on...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def grab_next_image_by_trigger(nodemap, cam):
    """
    This function acquires an image by executing the trigger node.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        # Use trigger to capture image
        # The software trigger only feigns being executed by the Enter key;
        # what might not be immediately apparent is that there is not a
        # continuous stream of images being captured; in other examples that
        # acquire images, the camera captures a continuous stream of images.
        # When an image is retrieved, it is plucked from the stream.

        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            # Get user input
            # input('Press the Enter key to initiate software trigger.')

            # Execute software trigger
            node_softwaretrigger_cmd = PySpin.CCommandPtr(nodemap.GetNode('TriggerSoftware'))
            if not PySpin.IsAvailable(node_softwaretrigger_cmd) or not PySpin.IsWritable(node_softwaretrigger_cmd):
                print('Unable to execute trigger. Aborting...')
                return False

            node_softwaretrigger_cmd.Execute()

            # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger

        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            print('Use the hardware to trigger image acquisition.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def acquire_images(cam, nodemap, nodemap_tldevice):
    """
    This function acquires and saves 10 images from a device.
    Please see Acquisition example for more in-depth comments on acquiring images.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Set acquisition mode to continuous
        # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        #  Begin acquiring images
        cam.BeginAcquisition()

        print('Acquiring images...')

        #  Retrieve device serial number for filename
        #
        #  *** NOTES ***
        #  The device serial number is retrieved in order to keep cameras from
        #  overwriting one another. Grabbing image IDs could also accomplish
        #  this.
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Retrieve, convert, and save images
        for i in range(NUM_IMAGES):
            try:

                #  Retrieve the next image from the trigger
                result &= grab_next_image_by_trigger(nodemap, cam)

                #  Retrieve next received image
                image_result = cam.GetNextImage()

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:

                    #  Print image information; height and width recorded in pixels
                    #
                    #  *** NOTES ***
                    #  Images have quite a bit of available metadata including
                    #  things such as CRC, image status, and offset values, to
                    #  name a few.
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    print('Grabbed Image %d, width = %d, height = %d' % (i, width, height))

                    #  Convert image to mono 8
                    #
                    #  *** NOTES ***
                    #  Images can be converted between pixel formats by using
                    #  the appropriate enumeration value. Unlike the original
                    #  image, the converted one does not need to be released as
                    #  it does not affect the camera buffer.
                    #
                    #  When converting images, color processing algorithm is an
                    #  optional parameter.
                    image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

                    # Create a unique filename
                    if device_serial_number:
                        filename = 'Trigger-%s-%d.jpg' % (device_serial_number, i)
                    else:  # if serial number is empty
                        filename = 'Trigger-%d.jpg' % i

                    # Save image
                    #
                    #  *** NOTES ***
                    #  The standard practice of the examples is to use device
                    #  serial numbers to keep images of one device from
                    #  overwriting those of another.
                    image_converted.Save(filename)
                    print('Image saved at %s\n' % filename)

                    #  Release image
                    #
                    #  *** NOTES ***
                    #  Images retrieved directly from the camera (i.e. non-converted
                    #  images) need to be released in order to keep from filling the
                    #  buffer.
                    image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        # End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def reset_trigger(nodemap):
    """
    This function returns the camera to a normal state by turning off trigger mode.
  
    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

        print('Trigger mode disabled...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** DEVICE INFORMATION ***\n')

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not available.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        err = False

        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        result &= print_device_info(nodemap_tldevice)

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Configure trigger
        if configure_trigger(cam) is False:
            return False

        # Acquire images
        result &= acquire_images(cam, nodemap, nodemap_tldevice)

        # Reset trigger
        result &= reset_trigger(nodemap)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result



## LOAD YOUR DLC NETWORK
config='config.yaml' ### YOUR CONFIG PATH 
shuffle=1 #YOUR shuffle index
trainingsetindex=0 #YOUR training index


if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
    del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training

tf.reset_default_graph()
start_path=os.getcwd() #record cwd to return to this directory in the end

cfg = auxiliaryfunctions.read_config(config)
trainFraction = cfg['TrainingFraction'][trainingsetindex]
modelfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'

dlc_cfg = load_config(str(path_test_config))

Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(modelfolder , 'train'))if "index" in fn])

if cfg['snapshotindex'] == 'all':
    print("Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
    snapshotindex = -1
else:
    snapshotindex=cfg['snapshotindex']
    
increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
Snapshots = Snapshots[increasing_indices]

print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

# Check if data already was generated:
dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]

#update batchsize (based on parameters in config.yaml)
dlc_cfg['batch_size']=1#cfg['batch_size'] 

# Name for scorer:
DLCscorer = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations=trainingsiterations)
sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
pdindex = pd.MultiIndex.from_product([[DLCscorer], dlc_cfg['all_joints_names'], ['x', 'y', 'likelihood']],names=['scorer', 'bodyparts', 'coords'])



## LOAD CAMERA
system = PySpin.System.GetInstance()

# Get camera list
cam_list = system.GetCameras()

cam = cam_list.GetByIndex(0)

# Initialize camera
cam.Init()

# Test with single frame
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

configure_trigger(cam)

# Start acquisition
cam.BeginAcquisition()

nodemap = cam.GetNodeMap()


## LOAD ARDUINO
arduino=serial.Serial('/dev/ttyACM0',9600)


## MAIN FUNCTIONAL LOOP!

NFRAMES=3000 # your choice
t=time.time()
for i in range(NFRAMES):
    result=True

    result &= grab_next_image_by_trigger(nodemap, cam)

    #  Retrieve next received image
    image_primary = cam.GetNextImage()    
    width = image_primary.GetWidth()
    height = image_primary.GetHeight()

    image_array = image_primary.GetData().reshape(height,width)
    image_primary.Release()

    rgb = np.zeros((height,width,3),dtype=np.uint8)
    rgb[...,1] = image_array
    rgb[...,2] = image_array
    rgb[...,0] = image_array

    frame = img_as_ubyte(rgb)

    ## DLC prediction happens here
    pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)

    # Write a 1 to arduino to trigger laser
    if False: ### REPLACE WITH YOUR STIMULATION CONDITION
        arduino.write(b'1')
        print ('TRIGGER')
    else:
        arduino.write(b'0')


    for p in pose:
         cv2.circle(frame,(int(p[0]),int(p[1])),7,(0,0,255))
    cv2.circle(frame,(int(pose[0,0]),int(pose[0,1])),7,(0,0,255))    
    cv2.circle(frame,(int(pose[4,0]),int(pose[4,1])),7,(0,255,0))
    cv2.circle(frame,(int(pose[6,0]),int(pose[6,1])),7,(255,0,255))    
    cv2.circle(frame,(int(pose[9,0]),int(pose[9,1])),7,(255,255,0))        
    # cv2.imwrite('test0000.png', frame)
    plt.imshow(frame)
    plt.draw()
    plt.pause(0.0000001)
    plt.clf()


print ('%d frames in %f. '%(NFRAMES,time.time()-t))

