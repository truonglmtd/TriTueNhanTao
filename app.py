import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Volleyball AI Detection",
    page_icon="🏐",
    layout="wide"
)


# ================= CSS =================
st.markdown("""
<style>
.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    text-align: center;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 48px;
    font-weight: 900;
    color: white;
}
.hero p {
    font-size: 18px;
    color: #d1d5db;
}
.card {
    background: #111827;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #374151;
    margin-bottom: 15px;
}
.footer {
    text-align: center;
    color: #9ca3af;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="hero">
    <h1>🏐 Volleyball AI Detection</h1>
    <p>Hệ thống phát hiện người chơi, trọng tài và bóng trong ảnh bóng chuyền bằng YOLOv8</p>
</div>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception:
    st.error("Không thể tải mô hình. Vui lòng kiểm tra requirements.txt.")
    st.stop()

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🏐 Volleyball AI")
    st.caption("YOLOv8 Detection System")
    st.divider()

    conf = st.slider("Confidence", 0.10, 1.00, 0.35, 0.05)
    iou = st.slider("IoU", 0.10, 1.00, 0.45, 0.05)

    st.divider()
    st.subheader("🔐 Bảo mật")
    st.caption("✔ Chỉ nhận JPG/JPEG/PNG")
    st.caption("✔ Giới hạn file 10MB")
    st.caption("✔ Không lưu ảnh người dùng")


# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Dashboard",
    "📤 Detection",
    "📊 Analytics",
    "🔐 Security"
])

# ================= DASHBOARD =================
with tab1:
    c1, c2, c3 = st.columns(3)

    c1.metric("Model", "YOLOv8n")
    c2.metric("Task", "Detection")
    c3.metric("Max Upload", "10MB")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Tổng quan đề tài")
    st.write("""
    Ứng dụng sử dụng YOLOv8 để phát hiện đối tượng trong ảnh bóng chuyền.
    Người dùng có thể tải ảnh lên, hệ thống xử lý ảnh bằng mô hình AI và trả về
    ảnh kết quả có bounding box, độ tin cậy và bảng thống kê.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ================= DETECTION =================
with tab2:
    uploaded_file = st.file_uploader(
        "📤 Tải ảnh bóng chuyền lên",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File quá lớn. Vui lòng chọn ảnh nhỏ hơn 10MB.")
            st.stop()

        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("File ảnh không hợp lệ.")
            st.stop()

        with st.spinner("AI đang phân tích ảnh..."):
            results = model(image, conf=conf, iou=iou)
            result_img = results[0].plot()

        boxes = results[0].boxes
        names = results[0].names

        data = []

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                data.append({
                    "Class": names[cls_id],
                    "Confidence": round(score, 3),
                    "X1": round(x1, 1),
                    "Y1": round(y1, 1),
                    "X2": round(x2, 1),
                    "Y2": round(y2, 1)
                })

        df = pd.DataFrame(data)

        st.session_state["df"] = df
        st.session_state["result_img"] = result_img

        st.success("Phân tích hoàn tất!")

        m1, m2, m3 = st.columns(3)
        m1.metric("Objects", len(df))
        m2.metric("Confidence", conf)
        m3.metric("IoU", iou)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ảnh gốc")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Kết quả AI")
            st.image(result_img, use_container_width=True)

        result_pil = Image.fromarray(result_img)
        img_buffer = io.BytesIO()
        result_pil.save(img_buffer, format="PNG")

        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "⬇️ Tải ảnh kết quả",
                data=img_buffer.getvalue(),
                file_name="volleyball_detection_result.png",
                mime="image/png",
                use_container_width=True
            )

        with d2:
            st.download_button(
                "⬇️ Tải bảng CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="detection_results.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("Hãy tải ảnh lên để bắt đầu nhận diện.")

# ================= ANALYTICS =================
with tab3:
    st.subheader("📊 Bảng kết quả nhận diện")

    if "df" not in st.session_state:
        st.warning("Chưa có dữ liệu. Hãy upload ảnh ở tab Detection.")
    else:
        df = st.session_state["df"]

        if df.empty:
            st.info("Không phát hiện đối tượng nào.")
        else:
            st.dataframe(df, use_container_width=True)

            st.subheader("Biểu đồ thống kê đối tượng")
            count_df = df["Class"].value_counts().reset_index()
            count_df.columns = ["Class", "Count"]
            st.bar_chart(count_df.set_index("Class"))

# ================= SECURITY =================
with tab4:
    st.subheader("🔐 Bảo mật hệ thống")

    st.write("""
    Ứng dụng được bổ sung các kiểm soát bảo mật cơ bản:
    """)

    st.write("✔ Kiểm tra định dạng file đầu vào")
    st.write("✔ Giới hạn dung lượng upload")
    st.write("✔ Không lưu ảnh người dùng")
    st.write("✔ Không hiển thị lỗi kỹ thuật chi tiết")


st.markdown("""
<div class="footer">
    Developed for AI Course Project · YOLOv8 · Streamlit
</div>
""", unsafe_allow_html=True)
