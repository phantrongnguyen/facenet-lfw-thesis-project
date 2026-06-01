"""
App3 - FaceNet Realtime Attendance Demo
White-theme Streamlit app with cached registration embeddings, static image
recognition, and local OpenCV realtime recognition.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
for _p in [_APP_DIR, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

st.set_page_config(
    page_title="App3 Realtime Face Recognition",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(88, 166, 255, .20), transparent 30%),
            radial-gradient(circle at top right, rgba(158, 119, 237, .18), transparent 32%),
            linear-gradient(135deg, #f8fbff 0%, #eef5ff 48%, #ffffff 100%);
        color: #172033;
    }
    .main-header {
        border: 1px solid rgba(120, 144, 180, .22);
        background: rgba(255,255,255,.82);
        box-shadow: 0 24px 70px rgba(36, 72, 120, .12);
        backdrop-filter: blur(18px);
        border-radius: 28px;
        padding: 2rem 2.2rem;
        margin: .5rem 0 1.4rem 0;
    }
    .main-header h1 {
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.5rem);
        font-weight: 900;
        letter-spacing: -0.06em;
        background: linear-gradient(135deg, #2457ff 0%, #8b5cf6 48%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p { color: #526178; font-size: 1.05rem; margin: .65rem 0 0; }
    .soft-card {
        border: 1px solid rgba(120, 144, 180, .20);
        background: rgba(255,255,255,.86);
        box-shadow: 0 18px 45px rgba(36, 72, 120, .10);
        border-radius: 24px;
        padding: 1.25rem;
        margin: .75rem 0;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        border-radius: 999px;
        padding: .45rem .8rem;
        font-weight: 800;
        border: 1px solid rgba(36,87,255,.16);
        background: #eef4ff;
        color: #2457ff;
    }
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,.92) !important;
        border-right: 1px solid rgba(120,144,180,.22);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.82);
        border: 1px solid rgba(120,144,180,.18);
        padding: 1rem;
        border-radius: 18px;
        box-shadow: 0 12px 28px rgba(36,72,120,.08);
    }
    .stButton > button {
        border-radius: 14px;
        font-weight: 800;
        box-shadow: 0 12px 26px rgba(36,87,255,.14);
    }
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(36,87,255,.25);
        border-radius: 18px;
        padding: .5rem;
        background: rgba(255,255,255,.72);
    }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    import cv2
    import app3_utils as utils
    basic_loaded = True
except Exception as e:
    basic_loaded = False
    import_error = str(e)

st.markdown(
    """
    <div class="main-header">
        <span class="status-pill">⚡ App3 • Cached • Realtime</span>
        <h1>Nhận diện khuôn mặt nhanh</h1>
        <p>Đăng ký tạo embedding ngay, nhận diện ảnh tĩnh kèm vector và realtime bằng camera local.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not basic_loaded:
    st.error(f"Không tải được dependency App3: {import_error}")
    st.code("python -m pip install opencv-python streamlit pandas pillow", language="powershell")
    st.stop()

thresholds = utils.get_available_thresholds()
models = [m for m in utils.ALL_RECOGNITION_MODELS if m in thresholds] or utils.ALL_RECOGNITION_MODELS

with st.sidebar:
    st.markdown("### ⚙️ Cấu hình App3")
    default_idx = models.index("Facenet512_mtcnn") if "Facenet512_mtcnn" in models else 0
    model_name = st.selectbox("Model nhận diện", models, index=default_idx)
    threshold = st.slider(
        "Ngưỡng nhận diện",
        min_value=0.10,
        max_value=0.90,
        value=float(thresholds.get(model_name, 0.35)),
        step=0.01,
        help="Similarity >= ngưỡng thì hiện tên, thấp hơn thì Không xác định.",
    )
    frame_interval = st.slider("Realtime: xử lý mỗi N frame", 3, 60, 15, 1)
    st.caption("Tăng N nếu máy bị lag. Giảm N nếu muốn phản hồi nhanh hơn.")
    st.divider()
    cache_status = utils.get_embedding_cache_status()
    st.metric("Ảnh đã đăng ký", cache_status["registered_photos"])
    st.metric("Vector cache", cache_status["by_model"].get(model_name, 0))
    if st.button("⚡ Rebuild cache model này", use_container_width=True):
        with st.spinner("Đang rebuild embedding cache..."):
            utils.build_registered_embeddings_cache([model_name], force=True)
        st.success("Đã rebuild cache.")
        st.rerun()

reg_tab, static_tab, live_tab, manage_tab = st.tabs([
    "👤 Đăng ký",
    "🖼️ Nhận diện ảnh tĩnh",
    "🎥 Realtime camera",
    "🧠 Database & cache",
])

with reg_tab:
    st.markdown("### 👤 Đăng ký khuôn mặt")
    st.caption("Nên đăng ký 3–8 ảnh/người nếu ảnh rõ và đúng cùng một người. App sẽ chặn ảnh mới nếu không đủ giống user đã có.")
    col1, col2 = st.columns([1, 1])
    with col1:
        person_name = st.text_input("Tên user không dấu", placeholder="Gia Bao", key="app3_reg_name")
        if person_name:
            valid, msg = utils.is_valid_name(person_name)
            if not valid:
                st.error(msg)
            elif utils.is_name_duplicate(person_name):
                st.warning("Tên đã tồn tại. Ảnh mới sẽ được kiểm tra similarity trước khi thêm vào user này.")
            else:
                st.success("Tên hợp lệ.")

        uploaded = st.file_uploader("Upload ảnh khuôn mặt", type=["jpg", "jpeg", "png"], key="app3_upload")
        camera_capture = st.camera_input("Hoặc chụp nhanh bằng camera", key="app3_capture")
        face_file = uploaded or camera_capture
        enforce_gate = st.checkbox(
            "Chặn ảnh nếu không giống user đã đăng ký",
            value=True,
            help="Áp dụng khi thêm ảnh vào tên đã tồn tại. App so embedding ảnh mới với ảnh cũ của user này.",
        )

        if face_file and person_name:
            if st.button("💾 Đăng ký và tạo embedding ngay", type="primary", use_container_width=True):
                valid, msg = utils.is_valid_name(person_name)
                if not valid:
                    st.error(msg)
                else:
                    with st.spinner("Đang kiểm tra ảnh và tạo embedding cache..."):
                        try:
                            reg_result = utils.register_face_image_checked(
                                person_name.strip(),
                                face_file,
                                model_name,
                                threshold,
                                enforce_consistency=enforce_gate,
                            )
                        except Exception as exc:
                            reg_result = {"saved": False, "path": None, "gate": {"message": f"Lỗi đăng ký: {exc}"}}

                    gate = reg_result.get("gate", {})
                    if reg_result.get("saved"):
                        st.success(f"Đã đăng ký: `{reg_result['path']}`")
                        if gate.get("best_similarity") is not None:
                            st.info(f"Similarity với ảnh cũ gần nhất: {gate['best_similarity']:.4f} / ngưỡng {gate['threshold']:.2f}")
                        else:
                            st.info(gate.get("message", "Ảnh đầu tiên của user mới."))
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(gate.get("message", "Không thể đăng ký ảnh này."))
                        if gate.get("best_similarity") is not None:
                            st.warning(f"Similarity tốt nhất: {gate['best_similarity']:.4f} < ngưỡng {gate['threshold']:.2f}")
    with col2:
        if face_file:
            st.image(face_file, caption="Ảnh dùng để đăng ký", use_column_width=True)
        st.markdown(
            "<div class='soft-card'><b>Gợi ý:</b><br>"
            "Nhiều ảnh/người là tốt nếu ảnh rõ, khác góc/ánh sáng nhẹ. "
            "Không nên thêm ảnh người khác vào cùng tên vì sẽ làm vector cache bị nhiễu.</div>",
            unsafe_allow_html=True,
        )

with static_tab:
    st.markdown("### 🖼️ Nhận diện ảnh tĩnh")
    st.caption("Upload/chụp một ảnh, App3 sẽ xuất vector query và cho biết vector này gần nhất với vector nào trong database.")
    people = utils.list_registered_people()
    if not people:
        st.warning("Chưa có user trong database. Hãy đăng ký khuôn mặt trước.")
    else:
        col1, col2 = st.columns([1, 1.25])
        with col1:
            query_file = st.file_uploader("Upload ảnh cần nhận diện", type=["jpg", "jpeg", "png"], key="app3_static_query")
            query_capture = st.camera_input("Hoặc chụp ảnh cần nhận diện", key="app3_static_camera")
            query_source = query_file or query_capture
            if query_source:
                st.image(query_source, caption="Ảnh query", use_column_width=True)
            show_query_vector = st.checkbox("Hiển thị vector embedding query", value=False, key="app3_show_query_vector")
            run_static = st.button("🔍 Nhận diện khuôn mặt", type="primary", disabled=not query_source, use_container_width=True)

        with col2:
            if run_static and query_source:
                with st.spinner("Đang tạo vector cho ảnh query và so khớp với cache đã lưu..."):
                    temp_path = utils.save_uploaded_file(query_source, "app/temp_uploads")
                    result = utils.recognize_image_path_cached(temp_path, model_name, threshold)
                    crop = utils.extract_face_crop(temp_path, model_name)

                if result is None:
                    st.error("Cache rỗng hoặc chưa có embedding hợp lệ cho model này. Hãy rebuild cache.")
                else:
                    if result["is_match"]:
                        st.success(f"✅ Nhận diện: **{result['name']}**")
                    else:
                        st.warning("❓ Không xác định")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Similarity", f"{result['similarity']:.4f}")
                    m2.metric("Threshold", f"{result['threshold']:.2f}")
                    m3.metric("Vector dim", result.get("embedding_dim", "-"))
                    m4.metric("Thời gian", f"{result['elapsed']:.2f}s")

                    st.markdown(
                        f"Vector query gần nhất với vector database của **{result.get('raw_name', '-')}** "
                        f"từ file `{os.path.basename(result.get('photo_path', ''))}`."
                    )
                    if crop:
                        st.image(crop, caption="Khuôn mặt đã detect/align từ ảnh query", width=220)

                    top_matches = result.get("top_matches", [])
                    if top_matches:
                        st.markdown("#### Top vector gần nhất trong database")
                        df = pd.DataFrame(top_matches)
                        df["photo"] = df["photo_path"].apply(lambda p: os.path.basename(str(p)))
                        st.dataframe(
                            df[["rank", "person_name", "similarity", "photo"]],
                            use_container_width=True,
                            hide_index=True,
                        )

                    emb = np.asarray(result.get("query_embedding", []), dtype=np.float32)
                    if show_query_vector and emb.size:
                        st.markdown(f"#### Vector embedding query ({emb.size} chiều)")
                        st.caption("Hiển thị toàn bộ vector. Có thể copy hoặc tải CSV nếu cần báo cáo.")
                        st.code(np.array2string(emb, precision=6, separator=", ", max_line_width=120), language="text")
                        csv = pd.DataFrame({"index": np.arange(emb.size), "value": emb}).to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️ Tải vector CSV", csv, file_name=f"query_embedding_{model_name}.csv", mime="text/csv")

with live_tab:
    st.markdown("### 🎥 Realtime camera")
    st.caption("Realtime mới ưu tiên camera browser/WebRTC. Video không dùng vòng lặp OpenCV blocking của Streamlit nữa.")
    people = utils.list_registered_people()
    if not people:
        st.warning("Chưa có khuôn mặt trong database. Hãy đăng ký ít nhất 1 user trước.")
    else:
        st.success(f"Database có {len(people)} user: {', '.join(p['name'] for p in people)}")
        recognize_every = st.slider("Nhận diện mỗi N giây", 1.0, 5.0, 2.0, 0.5)
        st.info("Nếu realtime vẫn nặng, tăng N lên 3–5 giây hoặc dùng model nhẹ hơn. Camera chạy trong browser nên ổn định hơn OpenCV loop.")

        try:
            from streamlit_webrtc import WebRtcMode, webrtc_streamer
            rtc_available = True
            rtc_error = ""
        except Exception as exc:
            rtc_available = False
            rtc_error = str(exc)

        if rtc_available:
            processor_factory = utils.make_webrtc_processor(model_name, threshold, recognize_every)
            webrtc_ctx = webrtc_streamer(
                key=f"app3-webrtc-{model_name}",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=processor_factory,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            if webrtc_ctx.video_processor:
                state = webrtc_ctx.video_processor.get_state()
                last_result = state.get("last_result")
                if last_result:
                    if last_result.get("is_match"):
                        st.success(f"✅ {last_result['name']} • similarity {last_result['similarity']:.4f}")
                    elif last_result.get("error"):
                        st.warning(last_result["error"])
                    else:
                        st.warning(f"❓ Không xác định • gần nhất {last_result.get('raw_name', '-')} • similarity {last_result.get('similarity', 0):.4f}")
                else:
                    st.caption("Bấm START, cho phép camera trong trình duyệt, rồi nhìn vào vùng giữa khung hình.")
        else:
            st.error("Thiếu dependency realtime WebRTC nên không mở được realtime browser.")
            st.code("python -m pip install streamlit-webrtc av", language="powershell")
            st.caption(f"Chi tiết lỗi: {rtc_error}")

        st.divider()
        st.markdown("#### Fallback: chụp một lần bằng browser")
        browser_frame = st.camera_input("Chụp một ảnh để nhận diện", key="app3_realtime_snapshot")
        if browser_frame:
            with st.spinner("Đang nhận diện ảnh vừa chụp..."):
                temp_path = utils.save_uploaded_file(browser_frame, "app/temp_uploads")
                result = utils.recognize_image_path_cached(temp_path, model_name, threshold)
                crop = utils.extract_face_crop(temp_path, model_name)
            if result is None:
                st.error("Cache rỗng hoặc chưa có embedding hợp lệ. Hãy rebuild cache.")
            elif result["is_match"]:
                st.success(f"✅ {result['name']} • similarity {result['similarity']:.4f}")
            else:
                st.warning(f"❓ Không xác định • gần nhất {result.get('raw_name', '-')} • similarity {result.get('similarity', 0):.4f}")
            if crop:
                st.image(crop, caption="Khuôn mặt detect", width=220)

with manage_tab:
    st.markdown("### 🧠 Database & embedding cache")
    people = utils.list_registered_people()
    cache_status = utils.get_embedding_cache_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("User", len(people))
    c2.metric("Ảnh đăng ký", cache_status["registered_photos"])
    c3.metric("Tổng vector hợp lệ", cache_status["total_records"])

    if people:
        selected_people = []
        for p in people:
            with st.container():
                cols = st.columns([.3, 1, 2, 2])
                with cols[0]:
                    if st.checkbox("Chọn user", key=f"app3_select_{p['name']}", label_visibility="collapsed"):
                        selected_people.append(p["name"])
                with cols[1]:
                    if os.path.exists(p["first_photo"]):
                        st.image(p["first_photo"], width=90)
                with cols[2]:
                    st.markdown(f"**{p['name']}**")
                    st.caption(f"{p['photo_count']} ảnh")
                with cols[3]:
                    count = cache_status["by_model"].get(model_name, 0)
                    st.caption(f"Cache cho `{model_name}`: {count}/{cache_status['registered_photos']}")
                st.divider()

        ac1, ac2 = st.columns(2)
        with ac1:
            if st.button("⚡ Rebuild cache người đã chọn", disabled=not selected_people, use_container_width=True):
                with st.spinner("Đang rebuild..."):
                    utils.rebuild_embeddings_for_people(selected_people, [model_name], force=True)
                st.success("Đã rebuild cache.")
                st.rerun()
        with ac2:
            confirm = st.checkbox("Xác nhận xóa", key="app3_confirm_delete")
            if st.button("🗑️ Xóa người đã chọn", disabled=(not selected_people or not confirm), use_container_width=True):
                result = utils.delete_registered_people(selected_people)
                if result["deleted"]:
                    st.success("Đã xóa: " + ", ".join(result["deleted"]))
                if result["errors"]:
                    st.error("; ".join(result["errors"]))
                st.rerun()
    else:
        st.info("Database hiện đang trống.")
