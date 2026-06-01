"""
🔬 FaceNet Demo App - Đồ án KHDL
Ứng dụng demo nhận diện khuôn mặt với FaceNet & FaceNet512
"""

import streamlit as st
import os
import sys
import time
import numpy as np
from PIL import Image

# --- FIX PATH: đảm bảo Python luôn tìm thấy src/ và deepface ---
# Streamlit có thể chạy từ thư mục khác, cần thêm project root vào sys.path
_APP_DIR = os.path.dirname(os.path.abspath(__file__))   # .../project_001/
_PROJECT_ROOT = os.path.dirname(_APP_DIR)               # .../deep_face/
for _p in [_APP_DIR, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- THEME SESSION STATE ---
if "theme" not in st.session_state:
    st.session_state.theme = "dark"   # mặc định là dark

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FaceNet Demo - Nhận Diện Khuôn Mặt",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DYNAMIC THEME CSS ---
is_dark = st.session_state.theme == "dark"

if is_dark:
    _bg        = "linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%)"
    _sidebar   = "rgba(15,12,41,0.97)"
    _text      = "#e8e8f0"
    _subtext   = "#a0a0c0"
    _card      = "rgba(255,255,255,0.05)"
    _border    = "rgba(255,255,255,0.1)"
    _upload    = "rgba(102,126,234,0.25)"
    _table_td  = "#d0d0e0"
    _toggle_icon = "🌙"   # moon
    _toggle_label = "Light Mode"
else:
    _bg        = "linear-gradient(135deg, #f0f4ff 0%, #e8eeff 50%, #f5f0ff 100%)"
    _sidebar   = "rgba(235,238,255,0.98)"
    _text      = "#1a1a3e"
    _subtext   = "#5a5a7a"
    _card      = "rgba(0,0,0,0.04)"
    _border    = "rgba(0,0,0,0.12)"
    _upload    = "rgba(102,126,234,0.15)"
    _table_td  = "#2a2a4a"
    _toggle_icon = "☀️"   # sun
    _toggle_label = "Dark Mode"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * {{ font-family: 'Inter', sans-serif; }}

    .stApp {{
        background: {_bg};
        transition: background 0.4s ease;
    }}

    /* text color cho toàn bộ app */
    .stApp, .stApp p, .stApp span, .stApp div {{
        color: {_text};
    }}

    .main-header {{
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }}
    .main-header h1 {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }}
    .main-header p {{
        color: {_subtext};
        font-size: 0.95rem;
    }}

    .result-card {{
        background: {_card};
        border: 1px solid {_border};
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }}
    .match-card {{
        border-color: #00e676 !important;
        box-shadow: 0 0 20px rgba(0,230,118,0.2);
    }}
    .no-match-card {{
        border-color: #ff5252 !important;
        box-shadow: 0 0 20px rgba(255,82,82,0.2);
    }}

    section[data-testid="stSidebar"] {{
        background: {_sidebar} !important;
        border-right: 1px solid {_border};
        transition: background 0.4s ease;
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }}

    [data-testid="stFileUploader"] {{
        border: 2px dashed {_upload};
        border-radius: 12px;
        padding: 0.5rem;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .badge-match {{
        background: rgba(0,230,118,0.15);
        color: #00e676;
        border: 1px solid rgba(0,230,118,0.3);
    }}
    .badge-no-match {{
        background: rgba(255,82,82,0.15);
        color: #ff5252;
        border: 1px solid rgba(255,82,82,0.3);
    }}

    /* Theme toggle button */
    .theme-toggle-btn {{
        width: 100%;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid {_border};
        background: {_card};
        color: {_text};
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        justify-content: center;
    }}

    /* Khung định dạng khuôn mặt kiểu App Ngân Hàng (eKYC) */
    [data-testid="stCameraInput"] > div {{
        position: relative;
        overflow: hidden;
        border-radius: 16px;
    }}
    [data-testid="stCameraInput"] > div::after {{
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 260px;
        height: 360px;
        border-radius: 50%;
        border: 4px dashed #00e676;
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.75);
        pointer-events: none;
        z-index: 10;
    }}
    [data-testid="stCameraInput"]::before {{
        content: 'Vui lòng đưa khuôn mặt khớp với khung hình Oval bên dưới';
        display: block;
        text-align: center;
        color: #00e676;
        font-weight: bold;
        margin-bottom: 10px;
    }}
</style>
""", unsafe_allow_html=True)


# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🔬 FaceNet Performance Demo</h1>
    <p>Đánh giá hiệu năng FaceNet (128-D) và FaceNet512 (512-D) trên framework DeepFace</p>
</div>
""", unsafe_allow_html=True)


# --- SIDEBAR ---
with st.sidebar:
    # ==== THEME TOGGLE (đặt lên đầu sidebar) ====
    st.markdown("### 🎨 Giao diện")
    t_col1, t_col2 = st.columns([3, 1])
    with t_col1:
        current_label = "🌙 Dark Mode" if is_dark else "☀️ Light Mode"
        st.markdown(f"**Hiện tại:** {current_label}")
    with t_col2:
        pass

    if st.button(
        f"{_toggle_icon} Đổi sang {_toggle_label}",
        use_container_width=True,
        key="theme_btn"
    ):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()

    st.divider()
    st.markdown("### ⚙️ Cấu hình")
    
    model_choice = st.radio(
        "**Chọn mô hình:**",
        ["Facenet (128-D)", "Facenet512 (512-D)", "So sánh cả hai"],
        index=1,
    )
    
    # Map selection to model name
    MODEL_MAP = {
        "Facenet (128-D)": ["Facenet"],
        "Facenet512 (512-D)": ["Facenet512"],
        "So sánh cả hai": ["Facenet", "Facenet512"],
    }
    selected_models = MODEL_MAP[model_choice]
    
    st.divider()
    
    st.markdown("### 📋 Thông tin đề tài")
    st.markdown("""
    **Đề tài:** Đánh giá hiệu năng FaceNet & FaceNet512 trong DeepFace trên LFW
    
    **GVHD:** Vũ Phú Lộc  
    **Nhóm SV:**  
    - Nguyễn Gia Bảo  
    - Võ Bạch Kim Thịnh  
    - Phan Trọng Nguyên
    
    **Trường:** ĐH Công Thương TP.HCM
    """)
    
    st.divider()
    st.markdown("### 🧠 Cấu hình Pipeline")
    st.code("""
Detector: MTCNN
Alignment: Enabled
Distance: Cosine Similarity
Threshold:
  Facenet:    0.40
  Facenet512: 0.35
    """, language="yaml")


# --- IMPORT APP UTILS ---
# Các hàm KHÔNG cần deepface (đăng ký, liệt kê) - import trước
try:
    from app_utils import (
        save_uploaded_file,
        register_face_image, list_registered_people,
        is_valid_name, is_name_duplicate,
        load_evaluation_metrics,
    )
    basic_loaded = True
except ImportError as e:
    basic_loaded = False
    import_error = str(e)

# Các hàm CẦN deepface (verify, search, recognize) - import sau
try:
    from app_utils import (
        verify_faces, search_face, recognize_from_camera,
    )
    from src.config import THRESHOLD
    utils_loaded = True
except ImportError as e:
    utils_loaded = False
    import_error = str(e)


# --- MAIN TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Face Verification",
    "🔍 Face Search",
    "👤 Đăng Ký Khuôn Mặt",
    "📷 Nhận Diện Webcam",
    "📊 Kết Quả Đánh Giá",
])


# ==============================
# TAB 1: FACE VERIFICATION
# ==============================
with tab1:
    st.markdown("#### So sánh 2 ảnh khuôn mặt")
    st.caption("Upload 2 ảnh để kiểm tra xem có phải cùng một người hay không")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Ảnh thứ 1:**")
        file1 = st.file_uploader("Chọn ảnh 1", type=["jpg", "jpeg", "png"], key="img1", label_visibility="collapsed")
        if file1:
            st.image(file1, caption="Ảnh 1", use_container_width=True)
    
    with col2:
        st.markdown("**Ảnh thứ 2:**")
        file2 = st.file_uploader("Chọn ảnh 2", type=["jpg", "jpeg", "png"], key="img2", label_visibility="collapsed")
        if file2:
            st.image(file2, caption="Ảnh 2", use_container_width=True)
    
    # Button verify
    if file1 and file2:
        if st.button("🚀 **SO SÁNH NGAY**", type="primary", use_container_width=True):
            
            if not utils_loaded:
                st.error(f"Lỗi import module: {import_error}")
            else:
                # Lưu file tạm
                path1 = save_uploaded_file(file1)
                path2 = save_uploaded_file(file2)
                
                # Chạy verification cho từng model đã chọn
                all_results = {}
                
                for model_name in selected_models:
                    with st.spinner(f"⏳ Đang xử lý với **{model_name}**..."):
                        try:
                            result = verify_faces(path1, path2, model_name)
                            all_results[model_name] = result
                        except Exception as e:
                            st.error(f"Lỗi khi xử lý {model_name}: {str(e)}")
                
                # Hiển thị kết quả
                if len(all_results) == 1:
                    # Hiển thị kết quả 1 model
                    model_name = list(all_results.keys())[0]
                    r = all_results[model_name]
                    
                    card_class = "match-card" if r["is_match"] else "no-match-card"
                    badge_class = "badge-match" if r["is_match"] else "badge-no-match"
                    verdict = "✅ CÙNG MỘT NGƯỜI" if r["is_match"] else "❌ KHÁC NGƯỜI"
                    badge_text = "MATCH" if r["is_match"] else "NO MATCH"
                    
                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                            <h3 style="margin:0; color:white;">{verdict}</h3>
                            <span class="badge {badge_class}">{badge_text}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metric boxes
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Cosine Similarity", f"{r['similarity']:.4f}")
                    with m2:
                        st.metric("Confidence", f"{r['confidence']:.1f}%")
                    with m3:
                        st.metric("Inference Time", f"{r['inference_time']:.2f}s")
                    with m4:
                        st.metric("Threshold", f"{r['threshold']}")
                    
                    # Progress bar cho similarity
                    st.markdown(f"**Similarity Score:** {r['similarity']:.4f}")
                    st.progress(min(1.0, max(0.0, r['similarity'])))
                    
                    st.info(f"🧠 Model: **{model_name}** | Embedding: **{r['embedding_dim']}-D** | Detector: **MTCNN** | Align: **True**")
                
                elif len(all_results) == 2:
                    # Hiển thị so sánh 2 models
                    st.markdown("### 🔄 So sánh kết quả giữa 2 mô hình")
                    
                    c1, c2 = st.columns(2)
                    
                    for i, (model_name, r) in enumerate(all_results.items()):
                        col = c1 if i == 0 else c2
                        with col:
                            card_class = "match-card" if r["is_match"] else "no-match-card"
                            verdict = "✅ MATCH" if r["is_match"] else "❌ NO MATCH"
                            st.markdown(f"""
                            <div class="result-card {card_class}">
                                <h4 style="color:#667eea; margin:0 0 0.5rem 0;">{model_name} ({r['embedding_dim']}-D)</h4>
                                <h3 style="color:white; margin:0;">{verdict}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.metric("Similarity", f"{r['similarity']:.4f}")
                            st.metric("Confidence", f"{r['confidence']:.1f}%")
                            st.metric("Time", f"{r['inference_time']:.2f}s")
                            st.progress(min(1.0, max(0.0, r['similarity'])))
                    
                    # Bảng so sánh
                    st.markdown("#### 📊 Bảng so sánh chi tiết")
                    compare_data = {
                        "Chỉ số": ["Similarity", "Threshold", "Kết quả", "Confidence", "Inference Time", "Embedding Dim"],
                    }
                    for model_name, r in all_results.items():
                        compare_data[model_name] = [
                            f"{r['similarity']:.4f}",
                            f"{r['threshold']}",
                            "✅ Match" if r['is_match'] else "❌ No Match",
                            f"{r['confidence']:.1f}%",
                            f"{r['inference_time']:.2f}s",
                            f"{r['embedding_dim']}-D",
                        ]
                    
                    import pandas as pd
                    st.dataframe(
                        pd.DataFrame(compare_data).set_index("Chỉ số"),
                        use_container_width=True,
                    )
                
                # Cleanup temp files
                try:
                    os.remove(path1)
                    os.remove(path2)
                except:
                    pass
    
    elif file1 or file2:
        st.info("📤 Vui lòng upload đủ 2 ảnh để so sánh.")


# ==============================
# TAB 2: FACE SEARCH
# ==============================
with tab2:
    st.markdown("#### Tìm kiếm khuôn mặt trong database LFW")
    st.caption("Upload 1 ảnh → Tìm những khuôn mặt giống nhất trong bộ dữ liệu LFW (13,000+ ảnh)")
    
    query_file = st.file_uploader("Chọn ảnh truy vấn", type=["jpg", "jpeg", "png"], key="query")
    
    top_k = st.slider("Số kết quả hiển thị (Top-K)", min_value=3, max_value=15, value=5)
    
    if query_file:
        st.image(query_file, caption="Ảnh truy vấn", width=200)
        
        # Chỉ dùng 1 model cho search (model đầu tiên được chọn)
        search_model = selected_models[0]
        
        if st.button(f"🔍 **TÌM KIẾM** (dùng {search_model})", type="primary", use_container_width=True):
            if not utils_loaded:
                st.error(f"Lỗi import: {import_error}")
            else:
                query_path = save_uploaded_file(query_file)
                
                with st.spinner(f"⏳ Đang trích xuất đặc trưng và tìm kiếm trong {search_model} database..."):
                    try:
                        results = search_face(query_path, search_model, top_k=top_k)
                        
                        if results:
                            st.success(f"Tìm thấy {len(results)} kết quả tương đồng nhất!")
                            
                            threshold = THRESHOLD[search_model]
                            
                            for i, r in enumerate(results):
                                sim_pct = r["similarity"] * 100
                                is_above = r["similarity"] >= threshold
                                icon = "🟢" if is_above else "🔴"
                                
                                with st.container():
                                    rc1, rc2, rc3 = st.columns([1, 3, 2])
                                    with rc1:
                                        st.markdown(f"### #{i+1}")
                                        # Hiển thị ảnh nếu tồn tại
                                        if os.path.exists(r["path"]):
                                            st.image(r["path"], width=100)
                                    with rc2:
                                        st.markdown(f"**{r['name']}**")
                                        st.caption(os.path.basename(r["path"]))
                                    with rc3:
                                        st.markdown(f"{icon} Similarity: **{r['similarity']:.4f}**")
                                        st.progress(min(1.0, max(0.0, r["similarity"])))
                                    
                                    st.divider()
                        else:
                            st.warning("Không tìm thấy file embeddings. Hãy chạy `precompute_embeddings.py` trước!")
                    
                    except Exception as e:
                        st.error(f"Lỗi: {str(e)}")
                
                try:
                    os.remove(query_path)
                except:
                    pass


# ==============================
# TAB 3: FACE REGISTRATION
# ==============================
with tab3:
    st.markdown("#### 👤 Đăng Ký Khuôn Mặt Vào Hệ Thống")
    st.caption("Nhập tên và upload ảnh khuôn mặt để đăng ký vào cơ sở dữ liệu nhận diện")

    if not basic_loaded:
        st.error(f"Lỗi import module: {import_error}")
    else:
        reg_col1, reg_col2 = st.columns([1, 1])

        with reg_col1:
            st.markdown("##### 📝 Đăng ký mới")

            person_name = st.text_input(
                "Nhập tên (không dấu, ví dụ: 'Gia Bao'):",
                placeholder="Gia Bao",
                key="reg_name",
            )

            # Validate tên ngay khi nhập
            if person_name:
                valid, msg = is_valid_name(person_name)
                if not valid:
                    st.error(f"❌ {msg}")
                elif is_name_duplicate(person_name):
                    st.warning(f"⚠️ Tên '{person_name}' đã tồn tại! Bạn có thể thêm ảnh vào thư mục này.")
                else:
                    st.success(f"✅ Tên '{person_name}' hợp lệ và chưa tồn tại.")

            face_file = st.file_uploader(
                "Upload ảnh khuôn mặt",
                type=["jpg", "jpeg", "png"],
                key="reg_face",
                accept_multiple_files=False,
            )

            if face_file:
                st.image(face_file, caption="Ảnh khuôn mặt", width=250)

            if person_name and face_file:
                valid, msg = is_valid_name(person_name)
                if valid:
                    if st.button("💾 **ĐĂNG KÝ KHUÔN MẶT**", type="primary", use_container_width=True):
                        with st.spinner("Đang lưu..."):
                            try:
                                saved_path = register_face_image(person_name.strip(), face_file)
                                st.success(f"✅ Đã đăng ký thành công! Ảnh lưu tại: `{saved_path}`")
                                st.balloons()
                            except Exception as e:
                                st.error(f"❌ Lỗi: {str(e)}")

        with reg_col2:
            st.markdown("##### 📋 Danh sách người đã đăng ký")

            people = list_registered_people()

            if people:
                for p in people:
                    with st.container():
                        pc1, pc2 = st.columns([1, 2])
                        with pc1:
                            if os.path.exists(p["first_photo"]):
                                st.image(p["first_photo"], width=100)
                        with pc2:
                            st.markdown(f"**{p['name']}**")
                            st.caption(f"{p['photo_count']} ảnh đã đăng ký")
                        st.divider()
            else:
                st.info("Chưa có ai đăng ký. Hãy thêm khuôn mặt đầu tiên!")


# ==============================
# TAB 4: WEBCAM RECOGNITION
# ==============================
with tab4:
    st.markdown("#### 📷 Nhận Diện Khuôn Mặt Qua Webcam")
    st.caption("Bật camera, chụp ảnh và hệ thống sẽ nhận diện bạn là ai")

    if not utils_loaded:
        st.error(f"Lỗi import module: {import_error}")
    else:
        # Kiểm tra xem đã có người đăng ký chưa
        people = list_registered_people()
        if not people:
            st.warning("⚠️ Chưa có khuôn mặt nào trong cơ sở dữ liệu! Hãy vào tab **Đăng Ký Khuôn Mặt** để thêm trước.")
        else:
            st.info(f"📦 Cơ sở dữ liệu hiện có **{len(people)} người**: {', '.join(p['name'] for p in people)}")

            cam_model = selected_models[0]  # Dùng model đầu tiên được chọn
            st.markdown(f"🧠 Đang dùng model: **{cam_model}**")

            camera_photo = st.camera_input("📸 Chụp ảnh từ webcam")

            if camera_photo:
                st.image(camera_photo, caption="Ảnh vừa chụp", width=300)

                if st.button("🔍 **NHẬN DIỆN NGAY**", type="primary", use_container_width=True, key="recognize_btn"):
                    # Lưu ảnh webcam tạm
                    cam_path = save_uploaded_file(camera_photo, save_dir="app/temp_uploads")

                    with st.spinner(f"⏳ Đang nhận diện với {cam_model}..."):
                        results = recognize_from_camera(cam_path, cam_model)

                    # Cleanup ảnh tạm
                    try:
                        os.remove(cam_path)
                    except:
                        pass

                    if not results:
                        st.warning("Không tìm thấy kết quả phù hợp.")
                    elif "error" in results[0]:
                        st.error(f"❌ Lỗi: {results[0]['error']}")
                    else:
                        # Hiển thị kết quả tốt nhất
                        best = results[0]

                        if best["is_match"]:
                            st.markdown(f"""
                            <div class="result-card match-card">
                                <h2 style="color:#00e676; margin:0;">✅ Xin chào, {best['name']}!</h2>
                                <p style="color:#a0a0c0; margin:0.5rem 0 0 0;">Hệ thống đã nhận diện bạn thành công.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-card no-match-card">
                                <h2 style="color:#ff5252; margin:0;">❌ Không nhận diện được</h2>
                                <p style="color:#a0a0c0; margin:0.5rem 0 0 0;">Khuôn mặt không khớp với ai trong cơ sở dữ liệu.</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Hiển thị metrics
                        rm1, rm2, rm3 = st.columns(3)
                        with rm1:
                            st.metric("Người khớp nhất", best["name"])
                        with rm2:
                            st.metric("Cosine Similarity", f"{best['similarity']:.4f}")
                        with rm3:
                            st.metric("Confidence", f"{best['confidence']:.1f}%")

                        st.progress(min(1.0, max(0.0, best["similarity"])))

                        # Hiển thị ảnh so sánh
                        cmp1, cmp2 = st.columns(2)
                        with cmp1:
                            st.markdown("**Ảnh webcam:**")
                            st.image(camera_photo, use_container_width=True)
                        with cmp2:
                            st.markdown(f"**Ảnh đăng ký ({best['name']}):**")
                            if os.path.exists(best["photo_path"]):
                                st.image(best["photo_path"], use_container_width=True)

                        # Bảng tất cả kết quả
                        if len(results) > 1:
                            st.markdown("#### 📋 Tất cả kết quả so sánh")
                            import pandas as pd
                            res_df = pd.DataFrame([
                                {
                                    "Tên": r["name"],
                                    "Similarity": f"{r['similarity']:.4f}",
                                    "Confidence": f"{r['confidence']:.1f}%",
                                    "Kết quả": "✅ Match" if r["is_match"] else "❌ No Match",
                                }
                                for r in results
                            ])
                            st.dataframe(res_df, use_container_width=True, hide_index=True)


# ==============================
# TAB 5: EVALUATION DASHBOARD
# ==============================
with tab5:
    st.markdown("#### 📊 Kết quả đánh giá trên bộ dữ liệu LFW")
    st.caption("Bảng so sánh hiệu năng giữa FaceNet (128-D) và FaceNet512 (512-D) sau 3 lần tối ưu hóa")
    
    if utils_loaded:
        metrics = load_evaluation_metrics()
    else:
        # Fallback: hardcode metrics
        metrics = {
            "Facenet": {
                "embedding_dim": 128, "accuracy": 0.9351, "precision": 0.9745,
                "recall": 0.8936, "f1": 0.9323, "far": 0.0233, "frr": 0.1064,
                "threshold": 0.40, "inference_time": 0.6, "ram_mb": 362,
            },
            "Facenet512": {
                "embedding_dim": 512, "accuracy": 0.9396, "precision": 0.9523,
                "recall": 0.9256, "f1": 0.9388, "far": 0.0464, "frr": 0.0744,
                "threshold": 0.35, "inference_time": 0.6, "ram_mb": 175,
            },
        }
    
    # === Bảng so sánh chính ===
    st.markdown("##### 🏆 Bảng So Sánh Trực Diện (Lần 3 - Sau Tối Ưu Hóa)")
    
    import pandas as pd
    
    table_data = []
    for model_name, m in metrics.items():
        table_data.append({
            "Mô hình": model_name,
            "Embedding": f"{m['embedding_dim']}-D",
            "Accuracy": f"{m['accuracy']:.2%}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1']:.4f}",
            "FAR": f"{m['far']:.2%}",
            "FRR": f"{m['frr']:.2%}",
            "Threshold": m['threshold'],
            "Time (s/ảnh)": f"{m['inference_time']:.1f}",
            "RAM (MB)": m['ram_mb'],
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df.set_index("Mô hình"), use_container_width=True)
    
    # === Metric highlight cards ===
    st.markdown("##### 📈 Điểm nổi bật")
    
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.metric(
            "Best Accuracy",
            "93.96%",
            "Facenet512",
        )
    with h2:
        st.metric(
            "Best F1-Score", 
            "0.9388",
            "Facenet512",
        )
    with h3:
        st.metric(
            "Lowest FRR",
            "7.44%",
            "Facenet512 ↓",
        )
    with h4:
        st.metric(
            "Lowest RAM",
            "175 MB",
            "Facenet512 (-52%)",
        )
    
    # === Bar charts ===
    st.markdown("##### 📊 Biểu đồ trực quan so sánh")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        chart_df = pd.DataFrame({
            "Mô hình": ["Facenet", "Facenet512"],
            "Accuracy": [93.51, 93.96],
        }).set_index("Mô hình")
        st.bar_chart(chart_df, color="#667eea")
        st.caption("Accuracy (%)")
    
    with chart_col2:
        chart_df2 = pd.DataFrame({
            "Mô hình": ["Facenet", "Facenet512"],
            "FAR (%)": [2.33, 4.64],
            "FRR (%)": [10.64, 7.44],
        }).set_index("Mô hình")
        st.bar_chart(chart_df2)
        st.caption("FAR vs FRR (%)")
    
    # === ROC Curve (nếu có file ảnh) ===
    roc_path = "reports/FAR_FRR_ROC_Diagram.png"
    if os.path.exists(roc_path):
        st.markdown("##### 📉 Biểu đồ ROC & FAR/FRR Trade-off")
        st.image(roc_path, caption="ROC Curve & FAR/FRR Diagram", use_container_width=True)
    
    # === Nhật ký tối ưu hóa ===
    st.markdown("##### 📝 Nhật Ký Tối Ưu Hóa (3 Lần Thực Nghiệm)")
    
    with st.expander("🔴 Lần 1: Baseline (69% Accuracy)", expanded=False):
        st.markdown("""
        - **Cấu hình:** Đọc trực tiếp ảnh toàn cảnh, không Detection, không Alignment
        - **Kết quả:** Facenet: **69%** | Facenet512: **66%**
        - **Vấn đề:** Background noise gây nhiễu vector đặc trưng nghiêm trọng
        """)
    
    with st.expander("🟡 Lần 2: Thêm MTCNN + Alignment (72% Accuracy)", expanded=False):
        st.markdown("""
        - **Cấu hình:** Thêm `detector_backend="mtcnn"`, `align=True`
        - **Kết quả:** Facenet: **72.3%** | Facenet512: **69.7%** | FAR = **0%**
        - **Vấn đề:** Threshold quá khắt khe (0.7) → FRR lên tới **55-60%**
        """)
    
    with st.expander("🟢 Lần 3: Threshold Tuning (94% Accuracy) ✅", expanded=True):
        st.markdown("""
        - **Cấu hình:** Hạ Threshold: Facenet → **0.40**, Facenet512 → **0.35**
        - **Kết quả:**
            - Facenet: ACC = **93.5%** | F1 = 0.9323 | FAR = 2.3% | FRR = 10.6%
            - Facenet512: ACC = **93.9%** | F1 = 0.9388 | FAR = 4.6% | FRR = 7.4%
        - **Kết luận:** Pipeline **MTCNN + FaceNet512 + Cosine (Threshold=0.35)** là giải pháp tối ưu!
        """)

    # === Kết luận ===
    st.success("""
    **🏁 Kết luận:** FaceNet512 (512-D) vượt trội hơn Facenet (128-D) ở hầu hết các chỉ số, 
    đặc biệt là Accuracy (93.9%), FRR thấp (7.4%), và tiết kiệm RAM hơn 52%. 
    Pipeline MTCNN + FaceNet512 + Cosine Similarity (Threshold = 0.35) là giải pháp tối ưu nhất.
    """)
