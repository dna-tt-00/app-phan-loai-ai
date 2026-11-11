import streamlit as st
import numpy as np

st.set_page_config(page_title="Phân loại Hành vi Sử dụng AI", layout="centered")

st.title("🎓 Ứng Dụng Dự Đoán & Tư Vấn Hành Vi Sử Dụng Generative AI")
st.write("Nhập thông tin học tập của sinh viên để kiểm tra nhóm hành vi và nhận tư vấn tương ứng.")

# Thông số Chuẩn hóa (Mean, Std) và Tâm cụm K-Means thu được từ mô hình đã huấn luyện
MEAN = np.array([3.12, 3.15, 6.20, 10.50, 2.80])
STD = np.array([0.45, 0.48, 4.50, 5.20, 1.80])

# 3 Tâm cụm K-Means chuẩn (Cluster Centers) trong không gian chuẩn hóa Z-score
CENTERS = np.array([
    [0.10,  0.15, -0.20,  0.10, -0.10],  # Cụm 0: Moderate AI Users (Cân bằng)
    [0.45,  0.50, -1.10,  1.20, -1.20],  # Cụm 1: Non/Low AI Users (Tự học cao, dùng AI rất ít)
    [-0.50, -0.60,  1.80, -1.10,  1.60]   # Cụm 2: Heavy AI Users (Dùng AI rất nhiều, lạm dụng)
])

# Form nhập liệu trên Web
with st.form("student_form"):
    st.subheader("📋 Nhập thông tin sinh viên")
    pre_gpa = st.number_input("GPA kỳ trước (Pre_Semester_GPA):", 0.0, 4.0, 3.10)
    post_gpa = st.number_input("GPA kỳ này (Post_Semester_GPA):", 0.0, 4.0, 3.30)
    ai_hours = st.number_input("Số giờ dùng AI/tuần (Weekly_GenAI_Hours):", 0.0, 50.0, 5.00)
    trad_hours = st.number_input("Số giờ tự học truyền thống/tuần (Traditional_Study_Hours):", 0.0, 50.0, 11.00)
    tools = st.slider("Số lượng công cụ AI sử dụng (Tool_Diversity):", 0, 10, 3)
    
    submit_btn = st.form_submit_button("🔍 Phân Loại Cụm Hành Vi")

if submit_btn:
    # 1. Thu thập và chuẩn hóa dữ liệu Z-score: (X - Mean) / Std
    user_val = np.array([pre_gpa, post_gpa, ai_hours, trad_hours, tools])
    user_scaled = (user_val - MEAN) / STD
    
    # 2. Tính khoảng cách Euclidean từ vector sinh viên đến 3 tâm cụm K-Means
    distances = np.linalg.norm(CENTERS - user_scaled, axis=1)
    cluster = np.argmin(distances) # Chọn cụm có khoảng cách ngắn nhất
    
    st.markdown("---")
    if cluster == 0:
        st.success("📌 Kết quả: **Cụm 0 - Nhóm Sử Dụng AI Hiệu Quả (Moderate AI Users)**")
        st.info("💡 **Lời khuyên:** Bạn đang kết hợp rất tốt giữa AI và tự học truyền thống. Hãy duy trì tần suất này!")
    elif cluster == 1:
        st.warning("📌 Kết quả: **Cụm 1 - Nhóm Ít/Không Sử Dụng AI (Non/Low AI Users)**")
        st.info("💡 **Lời khuyên:** Bạn nên thử nghiệm thêm các công cụ AI để tối ưu hóa thời gian làm bài và tìm kiếm tài liệu.")
    else:
        st.error("📌 Kết quả: **Cụm 2 - Nhóm Lạm Dụng AI (Heavy AI Users)**")
        st.info("💡 **Lời khuyên:** Cảnh báo nguy cơ lệ thuộc AI! Bạn nên tăng thời gian tự học truyền thống để nắm vững kiến thức gốc.")
