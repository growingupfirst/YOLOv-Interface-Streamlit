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
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

st.set_page_config(layout="wide")

cfg_model_path = 'models/yolov5s.pt'
model = None
confidence = .25


def image_input(data_src, device, sahi=False):
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
            img = infer_image(img_file, ready_img, device, sahi)
            st.image(img, caption="Model prediction")


def video_input(data_src, device, sahi=False):
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
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

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
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(frame, frame, device, sahi=sahi)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()


def infer_image(img, img_file, device='cpu', sahi='False'):
    if sahi:
        def callback(image):
            result = model(image)[0]
            return sv.Detections.from_ultralytics(result)
        
        slicer = sv.InferenceSlicer(callback=callback)
        detections = slicer(image=img_file)
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_scale=1, border_radius=10, text_thickness=4)

        rgb_img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
        annotated_image = bounding_box_annotator.annotate(
        scene=rgb_img, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)
        return annotated_image
    else:
        model.conf = confidence
        result = model(img, device=device) #if size else model(img,device=device)
        detections = sv.Detections.from_ultralytics(result[0])
        bounding_box_annotator = sv.BoxAnnotator(thickness=1)
        label_annotator = sv.LabelAnnotator(text_scale=1, border_radius=10, text_thickness=4)

        annotated_image = bounding_box_annotator.annotate(
            scene=img_file, detections=detections)
        
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)
        
        return annotated_image


@st.cache_resource
def load_model(path, device):
    model_ = YOLO(path)
    print(device)
    if (device == 'cuda') and (torch.cuda.is_available()):
        model_.to(device)
    else:
        model_.to('cpu')
    # model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    # detection_model = AutoDetectionModel.from_pretrained(model_path=path, device=device, model_type='yolov8')
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
    global model, confidence, cfg_model_path

    st.title("Object Recognition Dashboard")

    st.sidebar.title("Settings")

    # upload model
    model_src = st.sidebar.radio("Select yolov5 weight file", ["Use our demo model 5s", "Use your own model"])
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
        sahi_option = st.sidebar.checkbox('Use Sahi')
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)

        # load model
        model = load_model(cfg_model_path, device_option)

        # confidence slider
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

        # custom classes
        if st.sidebar.checkbox("Custom Classes"):
            model_names = list(model.names.values())
            assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])
            classes = [model_names.index(name) for name in assigned_class]
            model.classes = classes
        else:
            model.classes = list(model.names.keys())

        st.sidebar.markdown("---")

        # input options
        input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])

        # input src option
        data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])

        if input_option == 'image':
            image_input(data_src, device_option, sahi=sahi_option)
        else:
            video_input(data_src, device_option, sahi=sahi_option)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
