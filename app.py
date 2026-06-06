import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import io

st.set_page_config(
    page_title="Volleyball YOLOv8 Detection",
    page_icon="🏐",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #bbbbbb;
    margin-bottom: 30px;
}
.info-card {
    background-color: #1e1e2f;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏐 Volleyball Object Detection using YOLOv8</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hệ thống phát hiện người chơi, trọng tài và bóng trong ảnh bóng chuyền</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception:
    st.error("Không thể tải mô hình. Vui lòng kiểm tra file model hoặc requirements.txt.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Cài đặt nhận diện")
    confidence = st.slider("Confidence threshold", 0.1, 1.0, 0.35, 0.05)
    iou = st.slider("IoU threshold", 0.1, 1.0, 0.45, 0.05)

    st.header("🔐 Bảo mật")
    st.write("✔ Chỉ hỗ trợ JPG, JPEG, PNG")
    st.write("✔ Giới hạn file tối đa 10MB")
    st.write("✔ Không lưu ảnh người dùng")
    st.write("✔ Không hiển thị lỗi hệ thống chi tiết")

tab1, tab2, tab3 = st.tabs(["📤 Nhận diện ảnh", "📊 Kết quả", "ℹ️ Giới thiệu"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📤 Tải ảnh lên")
        uploaded_file = st.file_uploader(
            "Chọn ảnh bóng chuyền",
            type=["jpg", "jpeg", "png"]
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📌 Thông tin hệ thống")
        st.write("**Mô hình:** YOLOv8")
        st.write("**Nhiệm vụ:** Object Detection")
        st.write("**Đầu vào:** Ảnh bóng chuyền")
        st.write("**Đầu ra:** Ảnh có bounding box")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File quá lớn. Vui lòng tải ảnh nhỏ hơn 10MB.")
            st.stop()

        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Đang xử lý ảnh..."):
            results = model(image, conf=confidence, iou=iou)
            result_img = results[0].plot()

        st.session_state["image"] = image
        st.session_state["result_img"] = result_img
        st.session_state["results"] = results

        st.success("Nhận diện hoàn tất!")

        left, right = st.columns(2)

        with left:
            st.subheader("Ảnh gốc")
            st.image(image, use_container_width=True)

        with right:
            st.subheader("Ảnh sau nhận diện")
            st.image(result_img, use_container_width=True)

        result_pil = Image.fromarray(result_img)
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")

        st.download_button(
            label="⬇️ Tải ảnh kết quả",
            data=buffer.getvalue(),
            file_name="ket_qua_nhan_dien.png",
            mime="image/png"
        )

with tab2:
    st.subheader("📊 Bảng kết quả nhận diện")

    if "results" not in st.session_state:
        st.warning("Bạn cần upload ảnh ở tab Nhận diện ảnh trước.")
    else:
        results = st.session_state["results"]
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            st.info("Không phát hiện đối tượng nào.")
        else:
            data = []
            names = results[0].names

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                data.append({
                    "Đối tượng": names[cls_id],
                    "Độ tin cậy": round(conf, 3),
                    "X1": round(x1, 1),
                    "Y1": round(y1, 1),
                    "X2": round(x2, 1),
                    "Y2": round(y2, 1)
                })

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            st.metric("Tổng số đối tượng phát hiện", len(df))

with tab3:
    st.subheader("ℹ️ Giới thiệu đề tài")
    st.write("""
    Ứng dụng này sử dụng mô hình YOLOv8 để phát hiện các đối tượng trong ảnh bóng chuyền.
    Hệ thống hỗ trợ người dùng tải ảnh lên, xử lý ảnh bằng mô hình AI và hiển thị kết quả
    dưới dạng bounding box.
    """)

    st.write("""
    Trong đồ án, hệ thống có thể được dùng để minh họa quy trình xây dựng một bài toán
    phát hiện đối tượng gồm: thu thập dữ liệu, gán nhãn, huấn luyện mô hình, kiểm thử
    và triển khai ứng dụng web demo.
    """)
