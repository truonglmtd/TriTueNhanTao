import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io

st.set_page_config(
    page_title="Volleyball AI Detection",
    page_icon="🏐",
    layout="wide"
)

st.markdown("""
<style>
.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(135deg, #1f2937, #111827);
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
.metric-card {
    background: #0f172a;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #334155;
    text-align: center;
}
.footer {
    text-align: center;
    color: #9ca3af;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🏐 Volleyball AI Detection</h1>
    <p>Hệ thống phát hiện người chơi, trọng tài và bóng trong ảnh bóng chuyền bằng YOLOv8</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception:
    st.error("Không thể tải mô hình. Vui lòng kiểm tra requirements.txt hoặc file model.")
    st.stop()

with st.sidebar:
    st.title("⚙️ Control Panel")

    conf = st.slider("Confidence", 0.10, 1.00, 0.35, 0.05)
    iou = st.slider("IoU", 0.10, 1.00, 0.45, 0.05)

    st.divider()
    st.subheader("🔐 Security")
    st.caption("✔ Chỉ nhận JPG, JPEG, PNG")
    st.caption("✔ Giới hạn file 10MB")
    st.caption("✔ Không lưu ảnh người dùng")
    st.caption("✔ Ẩn lỗi hệ thống")

    st.divider()
    st.subheader("📌 Model")
    st.caption("YOLOv8n")
    st.caption("Object Detection")

tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Detection",
    "📊 Analytics",
    "🧠 Model Info",
    "🔐 Security"
])

with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Tải ảnh bóng chuyền lên",
            type=["jpg", "jpeg", "png"]
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📋 Hướng dẫn")
        st.write("1. Tải ảnh bóng chuyền lên")
        st.write("2. Điều chỉnh Confidence và IoU nếu cần")
        st.write("3. Xem kết quả nhận diện")
        st.write("4. Tải ảnh hoặc bảng kết quả")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Project Overview")
        st.write("**Tên đề tài:** Xây dựng hệ thống phát hiện đối tượng trong ảnh bóng chuyền")
        st.write("**Mô hình:** YOLOv8")
        st.write("**Nhiệm vụ:** Object Detection")
        st.write("**Đầu ra:** Bounding box + confidence")
        st.markdown('</div>', unsafe_allow_html=True)

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

        img_col1, img_col2 = st.columns(2)

        with img_col1:
            st.subheader("Ảnh gốc")
            st.image(image, use_container_width=True)

        with img_col2:
            st.subheader("Kết quả AI")
            st.image(result_img, use_container_width=True)

        result_pil = Image.fromarray(result_img)
        img_buffer = io.BytesIO()
        result_pil.save(img_buffer, format="PNG")

        c1, c2 = st.columns(2)

        with c1:
            st.download_button(
                "⬇️ Tải ảnh kết quả",
                data=img_buffer.getvalue(),
                file_name="volleyball_detection_result.png",
                mime="image/png",
                use_container_width=True
            )

        with c2:
            st.download_button(
                "⬇️ Tải bảng CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="detection_results.csv",
                mime="text/csv",
                use_container_width=True
            )

with tab2:
    st.subheader("📊 Detection Analytics")

    if "df" not in st.session_state:
        st.warning("Chưa có dữ liệu. Hãy upload ảnh ở tab Detection.")
    else:
        df = st.session_state["df"]

        if df.empty:
            st.info("Không phát hiện đối tượng nào.")
        else:
            st.dataframe(df, use_container_width=True)

            st.subheader("Thống kê theo lớp đối tượng")
            count_df = df["Class"].value_counts().reset_index()
            count_df.columns = ["Class", "Count"]
            st.bar_chart(count_df.set_index("Class"))

with tab3:
    st.subheader("🧠 Thông tin mô hình")

    st.markdown("""
    **YOLOv8** là mô hình phát hiện đối tượng theo thời gian thực.
    Mô hình có khả năng xác định vị trí đối tượng trong ảnh bằng bounding box
    và trả về độ tin cậy cho từng đối tượng phát hiện được.
    """)

    st.markdown("""
    Trong đề tài này, hệ thống được xây dựng nhằm minh họa quy trình triển khai
    một ứng dụng AI từ mô hình học sâu sang giao diện web demo.
    """)

    st.info("Nếu có file best.pt sau khi train, có thể thay YOLOv8n bằng model đã huấn luyện riêng.")

with tab4:
    st.subheader("🔐 Chính sách bảo mật demo")

    st.write("Ứng dụng áp dụng một số kiểm soát bảo mật cơ bản:")

    st.write("✔ Kiểm tra định dạng file đầu vào")
    st.write("✔ Giới hạn dung lượng ảnh tải lên")
    st.write("✔ Không lưu ảnh người dùng trên server")
    st.write("✔ Không hiển thị traceback kỹ thuật cho người dùng")
    st.write("✔ Không yêu cầu thông tin cá nhân")

st.markdown("""
<div class="footer">
    Developed for AI Course Project · YOLOv8 · Streamlit
</div>
""", unsafe_allow_html=True)
