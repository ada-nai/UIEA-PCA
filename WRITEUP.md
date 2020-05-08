# Project Write-Up: Deploy a Counter App at the Edge

## Project Workflow
- The model used for inference for the app was the `SSD MobileNet V2 COCO`, which is a `TensorFlow` model with supported frozen topologies from the TensorFlow Object Detection Models Zoo
- [Here's the link to the model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) 
- PS: Create a folder `ssd_ir` and use `wget` to download the original model there. The model will then be stored in the `ssd_ir` folder
- The following command was used in the terminal to convert it to an Intermediate Representation with the Model Optimizer (NB: Go to the `ssd_ir` folder first and then execute this command): 
`python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_ir/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config ssd_ir/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
- The model will be converted into an IR and the `bin`, `xml` and `mapping` files will be stored in the `ssd_ir` folder
- I then setup the inference mechanism along with the UI server, MQTT server and ffmpeg server
- Once everything was up and running, one final problem was that the model could not detect person 2 for few frames/instances. This caused the app to send an incorrect count of the people counted to the UI server
- To solve this problem, I implemented a skip-frames logic where I ignored a certain number of frames when the model detects no person in the frame. This is exploiting the fact that there is an ample frame gap between a person leaving the frame and the next person entering the frame
- The skip-frame logic involves using 3 variables keeping track of the frames, when there are no detections by the model for a given frame. The variables used are:
a. `frame_count`, which keeps counting frames throughout the video feed
b. `ref_count`, which is used to set a reference frame when the output detected by model is zero. This variable is constant for every case when there is no output detected by the model
c. `temp_count`, which is used to keep track of the frames being skipped/ignored. Its range is [ref_count, ref_count + threshold], where threshold is a fixed value. `temp_count` is like `frame_count`, but it has a range for being incremented for each case.

- After implementing the skip-frames logic, even though the model did not detect a person for a few frames; for the permissible frame limit, the app simply ignores the model's inference to avoid incorrect counting/updation of people in the feed
- The app is now ready to make inferences, minimizing any room for false negatives
- The following command should be entered on the terminal to execute the app:
    ```
     python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_ir/frozen_inference_graph.xml -pt 0.25 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768*432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    ```
![Project output image](https://raw.githubusercontent.com/ada-nai/nd131-openvino-fundamentals-project-starter/master/write-up_images/demo_output.png)


## Explaining Custom Layers

 The model used for inference for the app was the `SSD MobileNet V2 COCO`, which is a `TensorFlow` model with supported frozen topologies from the TensorFlow Object Detection Models Zoo. Also, the CPU extension for Linux was added for further support while working with the IR. Thus, no custom layers were required to be handled manually.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were:

### Size of the model
- The size of the original model was _66.5 MB_
- The size of the model post conversion to its Intermediate Representation was reduced to _~64.2 MB_
- The resultant model size was **reduced by ~3.5%**

### Inference time of the model (/frame)
- Personally, I do not have much exposure working with TensorFlow v.1, so I referred articles from the Knowledge section
- I came across a [post](https://knowledge.udacity.com/questions/129841), where an [article](https://www.dlology.com/blog/how-to-run-tensorflow-object-detection-model-on-jetson-nano/) was mentioned regarding making inferences on frames using TensorFlow v1
- I executed the notebook on Google Colab with 'CPU' settings and made inference on frames extracted from the sample video provided  

![Average inference time for non-OpenVINO model](https://raw.githubusercontent.com/ada-nai/nd131-openvino-fundamentals-project-starter/master/write-up_images/TF%20inference.png)

- The average time of inference was _~150 ms_
- In case of the inferences using the OpenVINO toolkit, I simply averaged out the inference time of each frame using `time.time()` function  

![Average inference time for OpenVINO model](https://raw.githubusercontent.com/ada-nai/nd131-openvino-fundamentals-project-starter/master/write-up_images/OpenVINO%20inference.png)

- The average time of inference using OpenVINO toolkit was _~70 ms_
- The inference time when using OpenVINO was **reduced by 53%**

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

- Attendance Management systems
    Such a system can be deployed in classrooms, where the attendance procedures can be automated after integrating the current app with some face recognition functionality
- Polling booth centers
    The polling centers can make use of such a system during elections and the app can be further enhanced to ensure that there is no duplicacy in casting of votes, i.e. the same person does not get to vote twice, wherein an alarm would ring in such a case
- Stores
    Most stores, regardless of size, have CCTV cameras installed in their premises. The counter app can be used to calculate the total footfall in a day, and other statistics like when the footfall is maximum, to improve customer experience and and appropriately manage staff to meet the customer needs according to footfall

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:

- Too much / too less lighting could result in inaccuracies during the model inference. It is preferred that the room lighting be continuous and of a consistent nature
- The conversion of the original model to the IR may be advantageous in terms of size and inference time, but the model accuracy could sometimes be a trade-off. Even here in this case, there were problems faced in detecting person 2 in the sample video, however appropriate logic can be applied and implemented via code, such that the impact of lower accuracy is made minimal (done here using the skip-frames logic)
- Shorter focal lengths capture more area of the premise, thus making the objects more distinct. As the focal length increases, the object appears to be larger and the model might not be able to distinguish between objects in the frame. Thus, shorter focal lengths are preferred. Thus, camera focal lengths of 30-40 mm are preferred

