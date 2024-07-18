# Yolov Real-time Inference using Streamlit
A web interface for real-time yolo inference using streamlit. It supports CPU and GPU inference, supports both images and videos and uploading your own custom models.
It was based on [this repository](https://github.com/moaaztaha/Yolo-Interface-using-Streamlit) but the code was fully refurbished to add new features.

<img src="output.gif" alt="demo of the dashboard" width="800"/>

### [Streamlit Cloud Demo](https://yolov-interface-app-d2ijmoaxhszz2iyshshxnz.streamlit.app/)


## Features
- **Caches** the model for faster inference on both CPU and GPU. Now updated for the latest version of Streamlit.
- Supports uploading model files (<200MB) and downloading models from URL (any size)
- Supports both images and videos.
- Supports both CPU and GPU inference.
- Supports the **latest** YOLO models (v10 n/s/m/l/x)


## How to run
After cloning the repo:
1. Install requirements
   - `pip install -r requirements.txt`
2. Add sample images to `data/sample_images`
3. Add sample video to `data/sample_videos` and call it `sample.mp4` or change name in the code.
4. Add the model file to `models/` and change `cfg_model_path` to its path.
```bash
git clone https://github.com/growingupfirst/Yolov-Interface-Streamlit
cd Yolov-Interface-Streamlit
streamlit run app.py
```

### To do (not working at the moment):
- Supports:
  - Custom Classes
  - Changing Confidence
  - Changing input/frame size for videos
  - Batch processing

## References
- https://discuss.streamlit.io/t/deploy-yolov5-object-detection-on-streamlit/27675
- https://github.com/moaaztaha/Yolo-Interface-using-Streamlit
