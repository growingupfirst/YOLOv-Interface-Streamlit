import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
from ultralytics import YOLO
import supervision as sv

st.set_page_config(layout="wide")

cfg_model_path = 'models/yolov5s.pt'
model = None
previous_detections = None
empty_image_counter = 0
split_height = 320
split_width = 320

def image_input(data_src, confidence, classes, device='cpu', sahi=False, full_sahi_retrain=False, split_height=320, split_width=320):
    img_file = None
    if data_src == 'Sample data':
        # get all sample images
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            ready_img = cv2.imread(img_file)
            img = infer_image(img_file, ready_img, confidence, classes, device, sahi, full_sahi_retrain=full_sahi_retrain)
            st.image(img, caption="Model prediction")

def batched_video_input(data_src, confidence, classes, device='cpu', sahi=False, full_sahi_retrain=False, skip_image=0, split_height=320, split_width=320): 
    output = st.empty()
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "data/sample_videos/sample.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        batch_size = 4
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)

            if len(frames) == batch_size:
                output_imgs = infer_batch_frame(frames, frames, confidence, classes)
                for output_img in output_imgs:
                    output.image(output_img)
                frames = []

            

def video_input(data_src, confidence, classes, device='cpu', sahi=False, full_sahi_retrain=False, skip_image=0, split_height=320, split_width=320):
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "data/sample_videos/sample.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        placeholder = st.sidebar.empty()
        custom_size = placeholder.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width_placeholder = st.sidebar.empty()
            height_placeholder = st.sidebar.empty()
            width = width_placeholder.number_input("Width", min_value=120, step=20, value=width)
            height = height_placeholder.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.resize(frame, (width, height))
            print(f'EMPTY IMAGES:{empty_image_counter}')
            print(previous_detections)
            #should boost the performance of the video prediction        
            if ((counter>0) and (previous_detections is not None)):
                print('Not training now')                
                output_img = infer_video_frame(frame, frame, confidence, classes, device,
                                                sahi=sahi, full_sahi_retrain=full_sahi_retrain,
                                                  no_skip=False, split_height=split_height, split_width=split_width)
                counter+=1
                #SET COUNTER HERE TO CHANGE THE DISAPPEARANCE SPEED OF OBJECT WHICH LEFT THE PAGE 
                if (counter==skip_image) or (counter > 54):
                    counter = 0
            elif (counter == 0) or (previous_detections is None):
                output_img = infer_video_frame(frame, frame, confidence, classes, device,
                                                sahi=sahi, full_sahi_retrain=full_sahi_retrain,
                                                  split_height=split_height, split_width=split_width)
                counter+=1
                if skip_image==0:
                    counter = 0
                
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()


def infer_image(img, img_file, confidence, classes, device='cpu', sahi='False', full_sahi_retrain = False, detect_large=False, split_height=320, split_width=320):
    if sahi:
        def callback(image):
            print(f'Sahi confidence:{confidence}')
            model.conf = confidence
            result = model(image, conf=confidence, classes=classes)[0]
            return sv.Detections.from_ultralytics(result)
        slicer = sv.InferenceSlicer(callback=callback, slice_wh=(split_height,split_width), iou_threshold=0.01) #if you want to detect more small objects make it high.
        detections = slicer(image=img_file)
        if full_sahi_retrain:
            lol = sv.Detections.from_ultralytics(model(img_file, conf=confidence, classes=classes)[0]) #run on full image to get detections of large things
            detections = sv.Detections.merge([detections, lol]) #merge two detections
        detections = detections.with_nms(threshold=0.01) #the higher value, the higher chance of double detections
        #enable confidence mark
        try:
            labels = [
                    f"{class_name} {confidence:.2f}"
                    for class_name, confidence
                    in zip(detections['class_name'], detections.confidence)
                ]
        except:
            bounding_box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator(text_scale=1, border_radius=10, text_thickness=1)
            rgb_img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
            annotated_image = bounding_box_annotator.annotate(
            scene=rgb_img, detections=detections)
            annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)
            return annotated_image
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_scale=1, border_radius=10, text_thickness=1)
        rgb_img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
        annotated_image = bounding_box_annotator.annotate(
        scene=rgb_img, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)
        return annotated_image
    else:
        model.conf = confidence
        result = model(img, conf=confidence, classes=classes)
        print(model.conf)
        print(model.classes)
        detections = sv.Detections.from_ultralytics(result[0])
        bounding_box_annotator = sv.BoxAnnotator(thickness=1)
        label_annotator = sv.LabelAnnotator(text_scale=1, border_radius=10, text_thickness=4)
        rgb_img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
        annotated_image = bounding_box_annotator.annotate(
            scene=rgb_img, detections=detections)
        
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)

        return annotated_image

def infer_video_frame(img, img_file, confidence, classes, device='cpu', sahi='False', full_sahi_retrain = False, no_skip=True, split_height=320, split_width=320):
    global previous_detections, empty_image_counter
    if sahi:
        if (no_skip == True):
            def callback(image):
                print(f'Sahi confidence:{confidence}')
                model.conf = confidence
                result = model(image, conf=confidence, classes=classes)[0]
                return sv.Detections.from_ultralytics(result)
            slicer = sv.InferenceSlicer(callback=callback, slice_wh=(split_height,split_width), iou_threshold=0.01) #if you want to detect more small objects make it high.
            detections = slicer(image=img_file)
            if full_sahi_retrain:
                lol = sv.Detections.from_ultralytics(model(img_file, conf=confidence, classes=classes)[0]) #run on full image to get detections of large things
                detections = sv.Detections.merge([detections, lol]) #merge two detections
            detections = detections.with_nms(threshold=0.01) #the higher value, the higher chance of double detections
            #check if no detections for long time then use a new empty(or not) detection
            if (len(detections.xyxy) > 0) or (empty_image_counter>27):
                previous_detections = detections
                empty_image_counter = 0
            if len(detections.xyxy) == 0:
                empty_image_counter += 1

        else:
            print('\n IM USING PREVIOUS DETECTIONS\n')
            detections = previous_detections
            empty_image_counter += 1
        #enable confidence mark
        try:
            labels = [
                    f"{class_name} {confidence:.2f}"
                    for class_name, confidence
                    in zip(detections['class_name'], detections.confidence)
                ]
        except:
            bounding_box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator(text_scale=1, border_radius=10, text_thickness=1)
            rgb_img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
            annotated_image = bounding_box_annotator.annotate(
            scene=rgb_img, detections=detections)
            annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)
            return annotated_image

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_scale=1, border_radius=10, text_thickness=1)
        rgb_img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
        annotated_image = bounding_box_annotator.annotate(
        scene=rgb_img, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)
        return annotated_image
    else:
        model.conf = confidence
        result = model(img, conf=confidence, classes=classes)
        print(model.conf)
        print(model.classes)
        detections = sv.Detections.from_ultralytics(result[0])
        bounding_box_annotator = sv.BoxAnnotator(thickness=1)
        label_annotator = sv.LabelAnnotator(text_scale=1, border_radius=10, text_thickness=4)
        rgb_img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
        annotated_image = bounding_box_annotator.annotate(
            scene=rgb_img, detections=detections)
        
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)

        return annotated_image

def infer_batch_frame(imgs, img_files, confidence, classes, sahi='False', full_sahi_retrain = False, detect_large=False, split_height=320, split_width=320):
        if sahi:
            def callback(images):
                print(f'Sahi confidence:{confidence}')
                model.conf = confidence
                result = model(images, conf=confidence, classes=classes)[0]
                return sv.Detections.from_ultralytics(result)
            slicer = sv.InferenceSlicer(callback=callback, slice_wh=(split_height,split_width), iou_threshold=0.01) #if you want to detect more small objects make it high.
            detections = slicer(image=imgs)
            # if full_sahi_retrain:
            #     lol = sv.Detections.from_ultralytics(model(img_file, conf=confidence, classes=classes)[0]) #run on full image to get detections of large things
            #     detections = sv.Detections.merge([detections, lol]) #merge two detections
            # detections = detections.with_nms(threshold=0.01) #the higher value, the higher chance of double detections
        else:        
            annotated_images = []
            model.conf = confidence
            results = model(imgs, conf=confidence, classes=classes)

        for i, result in enumerate(results):
            rgb_img = cv2.cvtColor(img_files[i], cv2.COLOR_BGR2RGB)
            detections = sv.Detections.from_ultralytics(results[i])
            bounding_box_annotator = sv.BoxAnnotator(thickness=1)
            label_annotator = sv.LabelAnnotator(text_scale=1, border_radius=10, text_thickness=4)
            annotated_image = bounding_box_annotator.annotate(
                scene=rgb_img, detections=detections)
        
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections)
            annotated_images.append(annotated_image)
        return annotated_images



@st.cache_resource
def load_model(path, device):
    model_ = YOLO(path)
    print(device)
    if (device == 'cuda') and (torch.cuda.is_available()):
        model_.to(device)
    else:
        model_.to('cpu')
    return model_


@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file


def get_user_model():
    model_src = st.sidebar.radio("Model source", ["file upload", "url"])
    model_file = None
    if model_src == "file upload":
        model_bytes = st.sidebar.file_uploader("Upload a model file", type=['pt'])
        if model_bytes:
            model_file = "models/uploaded_" + model_bytes.name
            with open(model_file, 'wb') as out:
                out.write(model_bytes.read())
    else:
        url = st.sidebar.text_input("model url")
        if url:
            model_file_ = download_model(url)
            if model_file_.split(".")[-1] == "pt":
                model_file = model_file_

    return model_file

def main():
    # global variables
    global model, cfg_model_path, split_height, split_width

    st.title("Object Recognition Dashboard")

    st.sidebar.title("Settings")

    # upload model
    model_src = st.sidebar.radio("Select weight file", ["Use the demo YOLOV5s", "Use your own model"])
    # URL, upload file (max 200 mb)
    if model_src == "Use your own model":
        user_model_path = get_user_model()
        if user_model_path:
            cfg_model_path = user_model_path

        st.sidebar.text(cfg_model_path.split("/")[-1])
        st.sidebar.markdown("---")

    # check if model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please added to the model folder.", icon="⚠️")
    else:
        full_sahi=False
        sahi_option = st.sidebar.checkbox('Use Sahi')
        if sahi_option:
            st.sidebar.write('Retrain on whole image in the end? (Better for detection of large objects, but slower)')
            full_sahi = st.sidebar.checkbox('Enable predict on full image')
            col1, col2 = st.columns(2)
            with col1:
                split_height = st.number_input('Set image split height', value=320)
            with col2:
                split_width = st.number_input('Set image split width:', value=320)
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
        
        # load model
        model = load_model(cfg_model_path, device_option)
        classes = [list(model.names.values()).index(name) for name in list(model.names.values())]
        # confidence slider
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

        # custom classes
        if st.sidebar.checkbox("Custom Classes"):
            model_names = list(model.names.values())
            assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])
            classes = [model_names.index(name) for name in assigned_class]
            model.classes = classes
            print(model.classes)
        else:
            model.classes = list(model.names.keys())

        st.sidebar.markdown("---")

        # input options
        input_option = st.sidebar.radio("Select input type: ", ['image', 'video', 'batched_video'])

        # input src option
        data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])

        if input_option == 'image':
            image_input(data_src, confidence, classes, device_option,
                         sahi=sahi_option, full_sahi_retrain=full_sahi,
                         split_height=split_height,
                         split_width=split_width)
        elif input_option == 'video':
            skip_image = st.number_input('How many images to skip:', value=0)
            video_input(data_src, confidence, classes, device_option,
                         sahi=sahi_option, full_sahi_retrain=full_sahi,
                         skip_image=skip_image, split_height=split_height,
                         split_width=split_width)
        elif input_option == 'batched_video':
            batched_video_input(data_src, confidence, classes)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
