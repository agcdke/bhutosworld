import threading

import av
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
from ultralytics import YOLO

st.set_page_config(
    page_title="Live Pose Estimation",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        h1 { text-align: center; }
        .stAlert p { margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏃 Live Body Pose Estimation")
st.caption("Real-time 17-keypoint detection powered by YOLOv8 · click **START** to begin")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "YOLOv8 Pose Model",
        options=["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt"],
        index=0,
        help="Nano → fastest; Small → balanced; Medium → most accurate",
    )

    confidence = st.slider("Confidence Threshold", 0.10, 0.95, 0.50, 0.05)

    st.subheader("Display")
    show_bbox = st.checkbox("Bounding Box", value=True)
    show_labels = st.checkbox("Labels", value=True)
    show_conf = st.checkbox("Confidence Score", value=True)

    st.divider()
    st.markdown(
        """
        **17 COCO Keypoints**
        - Nose · L/R Eye · L/R Ear
        - L/R Shoulder · Elbow · Wrist
        - L/R Hip · Knee · Ankle
        """
    )
    st.divider()
    st.markdown(
        """
        **Tips for best results**
        - Stand 1.5 – 3 m from camera
        - Good, even lighting
        - Keep full body in frame
        """
    )

# ── Model (cached per model name) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_model(name: str) -> YOLO:
    return YOLO(name)


model = load_model(model_choice)

# ── Thread-safe runtime config ────────────────────────────────────────────────
_lock = threading.Lock()
_cfg: dict = {}

with _lock:
    _cfg.update(
        confidence=confidence,
        show_bbox=show_bbox,
        show_labels=show_labels,
        show_conf=show_conf,
    )

# ── Video processor ───────────────────────────────────────────────────────────
class PoseEstimator:
    """WebRTC video processor: runs YOLOv8 pose on every frame."""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        with _lock:
            cfg = dict(_cfg)

        results = model(img, conf=cfg["confidence"], verbose=False)

        annotated = results[0].plot(
            boxes=cfg["show_bbox"],
            labels=cfg["show_labels"],
            conf=cfg["show_conf"],
            kpt_radius=6,
            line_width=2,
        )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ── RTC configuration (public STUN servers) ───────────────────────────────────
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)

# ── Layout ────────────────────────────────────────────────────────────────────
video_col, info_col = st.columns([3, 1])

with video_col:
    ctx = webrtc_streamer(
        key=f"pose-{model_choice}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=PoseEstimator,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with info_col:
    st.subheader("Model info")
    model_info = {
        "yolov8n-pose.pt": ("Nano", "~6 MB", "~30+ FPS"),
        "yolov8s-pose.pt": ("Small", "~23 MB", "~20+ FPS"),
        "yolov8m-pose.pt": ("Medium", "~52 MB", "~12+ FPS"),
    }
    name, size, fps = model_info[model_choice]
    st.markdown(
        f"""
        | | |
        |---|---|
        | **Variant** | {name} |
        | **Size** | {size} |
        | **Est. FPS** | {fps} |
        | **Keypoints** | 17 |
        | **Task** | Pose |
        """
    )

    st.subheader("How it works")
    st.info(
        "Each video frame is sent to a Python backend via WebRTC. "
        "YOLOv8 detects people and regresses 17 skeletal keypoints, "
        "then draws the annotated frame back to your browser in real-time."
    )

    if ctx.state.playing:
        st.success("Stream is live")
    else:
        st.warning("Stream is stopped")
