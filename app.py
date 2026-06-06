import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="YOLOv8 Volleyball Detection",
    page_icon="🏐",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 46px;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
    margin-bottom: 30px;
}
.card {
    background-color: #1e1e2f;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏐 Hệ thống phát hiện đối tượng bóng chuyền bằng YOLOv8</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Phát hiện người chơi, trọng tài và bóng trong ảnh bóng chuyền</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
   return YOLO("yolov8n.pt") # nếu có best.pt thì đổi thành YOLO("best.pt")

model = load_model()

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Tải ảnh lên")
    uploaded_file = st.file_uploader(
        "Chọn ảnh bóng chuyền",
        type=["jpg", "jpeg", "png"]
    )
    st.info("Hỗ trợ định dạng JPG, JPEG, PNG")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Thông tin mô hình")
    st.write("**Mô hình:** YOLOv8")
    st.write("**Chức năng:** Phát hiện đối tượng trong ảnh")
    st.write("**Đối tượng:** Người chơi, trọng tài, bóng")
    st.write("**Ứng dụng:** Hỗ trợ phân tích hình ảnh thể thao")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ảnh gốc")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Kết quả phát hiện")
        with st.spinner("Đang xử lý ảnh..."):
            results = model(image)
            result_img = results[0].plot()
            st.image(result_img, use_container_width=True)

    st.success("Phát hiện hoàn tất!")
else:
    st.warning("Vui lòng tải ảnh lên để bắt đầu phát hiện.")
