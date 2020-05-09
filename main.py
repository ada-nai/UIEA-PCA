"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

### Environment Variable for OpenVINO
### source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001 # Set the port for MQTT
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser('Counter App')
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default='/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so',
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-im", "--input_mode", type=str, default='VID',
                        help="Input mode selection \
                        (VID by default).\
                       Do not include `ffmpeg` arguments if input is only image")
    args = parser.parse_args()
    
    return args


def connect_mqtt():
    
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    return client

def bounding_boxes(frame, output, args):
    """
    Draw bounding boxes on out frame depending on inference output
    """
    width = int(frame.shape[1]) 
    height = int(frame.shape[0])
    op_count = 0 # Number of objects detected in the frame
    
    for box in output: # Output is squeezed here
        output_id = box[0]
        label = box[1]
        conf = box[2]
        
        # Break loop if first output in batch has id -1,
        # indicating no object further detected
        if output_id == -1:
            break
        
        # Draw box if object detected is person with conf>threshold
        elif (label == 1 and conf >= args.prob_threshold):
            x_min = int(box[3] * width)
            y_min = int(box[4] * height)
            x_max = int(box[5] * width)
            y_max = int(box[6] * height)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
            op_count += 1
            
    return frame, op_count


def infer_on_stream(args, client): 
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # SkipFrame counter
    frame_count = 1 
    temp_count = 0
    ref_count = 0
    
    # Stats counter
    prev_count = 0
    curr_count = 0
    total_count = 0
    start_time = 0
    duration = 0
    total_delta = 0
    
    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    
    
    ### TODO: Write an output image if `single_image_mode` ###
    if args.input_mode == 'IMG':
        frame = cv2.imread(args.input)
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]) )
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        infer_network.exec_net(p_frame)
        if infer_network.wait() == 0:
            result = infer_network.get_output()
            result = np.squeeze(result)
        frame, curr_count = bounding_boxes(frame, result, args)
        total_count = curr_count
        client.publish("person", json.dumps({"count": curr_count}))
        client.publish("person", json.dumps({"total": total_count}))
        client.publish("person/duration", json.dumps({"duration": duration}))
        cv2.imshow('out', frame)
        return
    
    ### TODO: Handle the input stream ###
    if args.input_mode == 'VID':
        cap = cv2.VideoCapture(args.input)
        cap.open(args.input)
    
    if args.input_mode == 'CAM':
        cap = cv2.VideoCapture(0)
        cap.open(0)
    
    ### Note width and height of original frame of video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame= cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]) )
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        
        assert net_input_shape == list(p_frame.shape)
        
        infer_start = time.time()
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)
        
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            
            ### Calculate average inference time
            delta = time.time() - infer_start
            total_delta = total_delta + delta
            avg_delta = total_delta / frame_count
            
            ### Squeeze result to remove unwanted dimensions
            result = np.squeeze(result)
        
        ### Draw bounding box for each frame
            frame, curr_count = bounding_boxes(frame, result, args)
            
            ### Publish average inference time on image
            check = "Average inference time(ms): " + "{0:.4g}".format(avg_delta*1000)
            frame = cv2.putText(frame, check, org = (100,75), fontScale=1, fontFace = cv2.FONT_HERSHEY_SIMPLEX, color= (255, 0, 0))
            

            ### TODO: Extract any desired stats from the results ###
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            if curr_count > prev_count:
                start_time = time.time()
            
                
                # Publish current_count
                client.publish("person", json.dumps({"count": curr_count}))
                
                prev_count = curr_count
                
                
            # SkipFrame implementation    
            if curr_count < prev_count:
                temp_count = frame_count
                if ref_count == 0:
                    ref_count = frame_count
                    
                    # Publish current_count
                    client.publish("person", json.dumps({"count": prev_count}))
                    
                if (temp_count - ref_count) <= 50:
                    
                    # Publish current_count
                    client.publish("person", json.dumps({"count": prev_count}))
                    
                    
                else:
                    # Calculate duration
                    duration = int(time.time() - start_time)

                    # Publish duration
                    client.publish("person/duration", json.dumps({"duration": duration}))
            
                    # Publish current_count
                    client.publish("person", json.dumps({"count": curr_count}))
                    
                    # Publish total_count
                    total_count = total_count + curr_count - prev_count
                    client.publish("person", json.dumps({"total": total_count}))    
                    
                    prev_count = curr_count
                    
                    # Reset ref_count
                    ref_count = 0
        
        ### Update frame_count
        frame_count += 1
            
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        
    
    ### Release the capture and destroy any cv2 frames
    client.publish("person", json.dumps({"count": 0}))
    cap.release()
    cv2.destroyAllWindows()
    



def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser()
    
    # Connect to the MQTT server
    client = connect_mqtt()
    
    # Perform inference on the input stream
    infer_on_stream(args, client) #, client
    
    ### Disconnect MQTT
    client.disconnect()

if __name__ == '__main__':
    main()
