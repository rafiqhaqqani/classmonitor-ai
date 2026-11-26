from datetime import datetime
import os
import tempfile
import threading
import time
import uuid
import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# Pengaman agar aplikasi tetap berjalan mulus meski av / streamlit-webrtc belum terinstal di Windows
try:
    import av
    from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
except ImportError:
    WEBRTC_AVAILABLE = False
    RTC_CONFIG = None

st.set_page_config(
    page_title="ClassMonitor AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HISTORY_FILE = "history_monitoring.csv"
TRACKER_FILE = "custom_bytetrack.yaml"
SNAPSHOT_DIR = "snapshots"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ================= CUSTOM DARK THEME CSS & FONTS =================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .stApp {
        background-color: #090d16;
        color: #f1f5f9;
    }

    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1280px;
    }

    .navbar-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 16px 22px;
        margin-bottom: 1rem;
    }
    .app-logo {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .logo-accent {
        background: linear-gradient(90deg, #10b981, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .status-chip {
        font-family: 'Fira Code', monospace !important;
        font-size: 0.78rem;
        padding: 6px 14px;
        border-radius: 8px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-left: 4px solid #10b981;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-card.cyan { border-left-color: #06b6d4; }
    .metric-card.amber { border-left-color: #f59e0b; }
    .metric-card.rose { border-left-color: #f43f5e; }

    .m-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        font-weight: 600;
    }
    .m-num {
        font-family: 'Fira Code', monospace !important;
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 2px 0;
    }
    .m-desc {
        font-size: 0.8rem;
        color: #cbd5e1;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: #111827;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important;
        color: #10b981 !important;
    }

    [data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid #1e293b;
    }

    .snap-box {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================= HELPER FUNCTIONS =================
def ensure_tracker_config():
    if not os.path.exists(TRACKER_FILE):
        yaml_text = """tracker_type: bytetrack
track_high_thresh: 0.30
track_low_thresh: 0.10
new_track_thresh: 0.35
track_buffer: 150
match_thresh: 0.80
fuse_score: True
"""
        with open(TRACKER_FILE, "w", encoding="utf-8") as f:
            f.write(yaml_text)


def compute_sustained_peak(counts, window=3):
    if not counts:
        return 0
    if len(counts) < window:
        return int(np.median(counts))
    rolling_medians = [
        np.median(counts[i : i + window]) for i in range(len(counts) - window + 1)
    ]
    return int(np.round(np.max(rolling_medians)))


def save_snapshot(bgr_image, existing_path=None):
    if existing_path and os.path.exists(existing_path):
        path = existing_path
    else:
        filename = f"peak_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}.jpg"
        path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, bgr_image)
    return path


def upsert_history(matkul, pertemuan, kapasitas, terdeteksi, sumber, snap_path="", row_idx=None):
    selisih = kapasitas - terdeteksi
    persentase = round((terdeteksi / kapasitas) * 100, 1) if kapasitas > 0 else 0

    if selisih > 0:
        status = f"Kurang {selisih}"
    elif selisih == 0:
        status = "Sesuai"
    else:
        status = f"Lebih {abs(selisih)}"

    row_data = {
        "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Mata Kuliah": matkul,
        "Pertemuan": f"P{pertemuan:02d}",
        "Kapasitas": kapasitas,
        "Terdeteksi": terdeteksi,
        "Selisih": selisih,
        "Kehadiran (%)": min(persentase, 100.0),
        "Status": status,
        "Sumber": sumber,
        "Snapshot": snap_path,
    }

    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        if "Snapshot" not in df.columns:
            df["Snapshot"] = ""
        if row_idx is not None and 0 <= row_idx < len(df):
            for k, v in row_data.items():
                df.at[row_idx, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
            row_idx = len(df) - 1
    else:
        df = pd.DataFrame([row_data])
        row_idx = 0

    df.to_csv(HISTORY_FILE, index=False)
    return row_idx


def render_telemetry_panel(terdeteksi, kapasitas, live_now=None):
    selisih = kapasitas - terdeteksi
    persentase = round((terdeteksi / kapasitas) * 100, 1) if kapasitas > 0 else 0

    if selisih > 0:
        status_txt = f"Kurang {selisih} dari target"
        accent = "rose"
    elif selisih == 0:
        status_txt = "Kapasitas kelas penuh"
        accent = ""
    else:
        status_txt = f"Melebihi +{abs(selisih)} orang"
        accent = "amber"

    sub_peak = (
        f"Terdeteksi saat ini: {live_now} Mhs"
        if live_now is not None
        else "Hasil deteksi aktif ruangan"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="m-title">Terdeteksi (Puncak Stabil)</div>
            <div class="m-num">{terdeteksi} <span style="font-size:1rem;color:#94a3b8;">Mhs</span></div>
            <div class="m-desc">{sub_peak}</div>
        </div>
        <div class="metric-card cyan">
            <div class="m-title">Target Kapasitas</div>
            <div class="m-num">{kapasitas} <span style="font-size:1rem;color:#94a3b8;">Mhs</span></div>
            <div class="m-desc">Mahasiswa terdaftar</div>
        </div>
        <div class="metric-card {accent}">
            <div class="m-title">Selisih & Okupansi</div>
            <div class="m-num">{persentase}%</div>
            <div class="m-desc">{status_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_yolo(model_path):
    return YOLO(model_path)


ensure_tracker_config()

# ================= TOP CONTROL BAR =================
with st.container():
    c_top1, c_top2, c_top3 = st.columns([2.6, 0.9, 0.9])
    with c_top1:
        preset_matkul = [
            "Kecerdasan Buatan dan Analitika Data",
            "Teknologi Internet Untuk Segala",
            "Sistem Komunikasi",
            "Sistem Tenaga Listrik",
            "Sistem Instrumentasi",
            "Tata Tulis Karya Ilmiah",
            "Input Manual...",
        ]
        pilihan = st.selectbox("Mata Kuliah", preset_matkul)
        if pilihan == "Input Manual...":
            matkul = st.text_input("Nama Mata Kuliah", "Mata Kuliah Umum")
        else:
            matkul = pilihan
    with c_top2:
        pertemuan = st.number_input("Pertemuan", min_value=1, max_value=16, value=1)
    with c_top3:
        kapasitas = st.number_input("Kapasitas", min_value=1, value=30)

    with st.expander("Kalibrasi Parameter Deteksi (Model / Confidence / IoU / Resolusi)", expanded=False):
        p0, p1, p2, p3 = st.columns(4)
        with p0:
            opsi_model = ["yolov8m.pt"] + (["best.pt"] if os.path.exists("best.pt") else [])
            selected_model_path = st.selectbox("Model AI", opsi_model, index=0)
        with p1:
            conf_thresh = st.slider("Confidence", 0.10, 0.80, 0.25, step=0.05)
        with p2:
            iou_thresh = st.slider("IoU Threshold", 0.30, 0.90, 0.50, step=0.05)
        with p3:
            img_size = st.select_slider("Resolusi (px)", options=[640, 800, 1024, 1280], value=1280)

st.markdown(
    f"""
    <div class="navbar-box">
        <div class="app-logo">CLASSMONITOR <span class="logo-accent">// AI VISION</span></div>
        <div class="status-chip">{matkul} • P{pertemuan:02d} • Target {kapasitas} • {selected_model_path}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_yolo(selected_model_path)
except Exception as e:
    st.error(f"Gagal memuat model AI ({selected_model_path}): {e}")
    st.stop()

# ================= NAVIGATION TABS =================
tab_cam, tab_foto, tab_video, tab_history = st.tabs([
    "Kamera Live",
    "Unggah Foto",
    "Unggah Video",
    "Riwayat & Analitik",
])

# ----------------- TAB 1: KAMERA LIVE -----------------
with tab_cam:
    if WEBRTC_AVAILABLE:
        if "live_shared" not in st.session_state:
            st.session_state["live_shared"] = {
                "lock": threading.Lock(),
                "current_count": 0,
                "counts": [],
                "best_frame_count": -1,
                "best_bgr": None,
                "conf": conf_thresh,
                "iou": iou_thresh,
                "imgsz": img_size,
            }

        shared = st.session_state["live_shared"]
        with shared["lock"]:
            shared["conf"] = conf_thresh
            shared["iou"] = iou_thresh
            shared["imgsz"] = img_size

        def live_video_callback(frame):
            img_bgr = frame.to_ndarray(format="bgr24")

            with shared["lock"]:
                c_th = shared["conf"]
                i_th = shared["iou"]
                sz = shared["imgsz"]

            results = model.predict(
                img_bgr,
                classes=[0],
                conf=c_th,
                iou=i_th,
                imgsz=sz,
                verbose=False,
            )[0]

            curr = len(results.boxes) if results.boxes is not None else 0
            plotted_bgr = results.plot(line_width=2, font_size=0.5)

            with shared["lock"]:
                shared["current_count"] = curr
                shared["counts"].append(curr)
                if len(shared["counts"]) > 120:
                    shared["counts"].pop(0)
                if curr >= shared["best_frame_count"] or shared["best_bgr"] is None:
                    shared["best_frame_count"] = curr
                    shared["best_bgr"] = plotted_bgr.copy()

            return av.VideoFrame.from_ndarray(plotted_bgr, format="bgr24")

        col_cam_main, col_cam_stat = st.columns([2.4, 1])

        with col_cam_main:
            ctx = webrtc_streamer(
                key="classmonitor-live-stream",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIG,
                video_frame_callback=live_video_callback,
                media_stream_constraints={
                    "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
                    "audio": False,
                },
                async_processing=True,
            )

        with col_cam_stat:
            live_kpi_box = st.empty()
            with shared["lock"]:
                init_peak = compute_sustained_peak(shared["counts"])
                init_curr = shared["current_count"]
            with live_kpi_box.container():
                render_telemetry_panel(init_peak, kapasitas, live_now=init_curr)

        if ctx.state.playing:
            session_key = f"live_{matkul}_{pertemuan}_{kapasitas}_{selected_model_path}"
            if st.session_state.get("active_live_session") != session_key:
                st.session_state["active_live_session"] = session_key
                st.session_state["live_row_idx"] = None
                st.session_state["live_snap_path"] = None
                with shared["lock"]:
                    shared["counts"] = []
                    shared["best_frame_count"] = -1
                    shared["best_bgr"] = None

            while ctx.state.playing:
                with shared["lock"]:
                    curr_live = shared["current_count"]
                    peak_live = compute_sustained_peak(shared["counts"])
                    best_frame_live = (
                        shared["best_bgr"].copy() if shared["best_bgr"] is not None else None
                    )

                if best_frame_live is not None and peak_live > 0:
                    st.session_state["live_snap_path"] = save_snapshot(
                        best_frame_live, existing_path=st.session_state.get("live_snap_path")
                    )
                    st.session_state["live_row_idx"] = upsert_history(
                        matkul,
                        pertemuan,
                        kapasitas,
                        peak_live,
                        "Kamera Live",
                        snap_path=st.session_state["live_snap_path"],
                        row_idx=st.session_state.get("live_row_idx"),
                    )

                with live_kpi_box.container():
                    render_telemetry_panel(peak_live, kapasitas, live_now=curr_live)

                time.sleep(0.5)
    else:
        # Mode OpenCV Webcam HD Real-Time jika av / webrtc belum terinstal di lokal Windows
        col_cam_main, col_cam_stat = st.columns([2.4, 1])
        with col_cam_main:
            c_ctrl1, c_ctrl2 = st.columns([1.5, 1])
            with c_ctrl1:
                run_webcam = st.toggle("Aktifkan Kamera Live", value=False, key="cam_toggle")
            with c_ctrl2:
                cam_idx = st.selectbox(
                    "Device Kamera", [0, 1, 2], index=0, label_visibility="collapsed"
                )
            cam_frame_box = st.empty()

        with col_cam_stat:
            cam_kpi_box = st.empty()
            render_telemetry_panel(0, kapasitas, live_now=0)

        if run_webcam:
            cap = cv2.VideoCapture(cam_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not cap.isOpened():
                st.error("Kamera tidak ditemukan pada perangkat ini.")
            else:
                counts = []
                row_idx = None
                snap_path = None
                best_count = -1
                try:
                    while st.session_state.get("cam_toggle", False):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        res = model.predict(
                            frame,
                            classes=[0],
                            conf=conf_thresh,
                            iou=iou_thresh,
                            imgsz=img_size,
                            verbose=False,
                        )[0]
                        curr = len(res.boxes) if res.boxes is not None else 0
                        counts.append(curr)
                        if len(counts) > 120:
                            counts.pop(0)
                        peak = compute_sustained_peak(counts)
                        plotted_bgr = res.plot(line_width=2, font_size=0.5)

                        if curr >= best_count and peak > 0:
                            best_count = curr
                            snap_path = save_snapshot(plotted_bgr, existing_path=snap_path)
                            row_idx = upsert_history(
                                matkul,
                                pertemuan,
                                kapasitas,
                                peak,
                                "Kamera Live",
                                snap_path=snap_path,
                                row_idx=row_idx,
                            )

                        cam_frame_box.image(
                            cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB), use_container_width=True
                        )
                        with cam_kpi_box.container():
                            render_telemetry_panel(peak, kapasitas, live_now=curr)
                        time.sleep(0.03)
                finally:
                    cap.release()

# ----------------- TAB 2: UNGGAH FOTO -----------------
with tab_foto:
    col_f_main, col_f_stat = st.columns([2.4, 1])
    with col_f_main:
        up_img = st.file_uploader(
            "Pilih Foto", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )
    with col_f_stat:
        foto_kpi = st.empty()
        render_telemetry_panel(0, kapasitas)

    if up_img is not None:
        img_bgr = cv2.imdecode(np.asarray(bytearray(up_img.read()), dtype=np.uint8), 1)
        res = model.predict(
            img_bgr,
            classes=[0],
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=img_size,
            verbose=False,
        )[0]
        count = len(res.boxes) if res.boxes is not None else 0
        plotted_bgr = res.plot(line_width=2, font_size=0.5)

        img_sig = (
            f"{up_img.name}_{up_img.size}_{matkul}_{pertemuan}_{kapasitas}_"
            f"{selected_model_path}_{conf_thresh}_{iou_thresh}_{img_size}"
        )
        if st.session_state.get("last_saved_img") != img_sig:
            snap_file = save_snapshot(plotted_bgr)
            upsert_history(matkul, pertemuan, kapasitas, count, "Foto", snap_path=snap_file)
            st.session_state["last_saved_img"] = img_sig
            st.toast(f"Terekam ke Riwayat: {matkul} (P{pertemuan:02d})")

        with col_f_main:
            st.image(cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col_f_stat:
            foto_kpi.empty()
            render_telemetry_panel(count, kapasitas)

# ----------------- TAB 3: UNGGAH VIDEO -----------------
with tab_video:
    col_v_main, col_v_stat = st.columns([2.4, 1])
    with col_v_main:
        up_vid = st.file_uploader(
            "Pilih Video", type=["mp4", "avi", "mov"], label_visibility="collapsed"
        )
        frame_placeholder = st.empty()
    with col_v_stat:
        vid_kpi = st.empty()
        render_telemetry_panel(st.session_state.get("last_vid_count", 0), kapasitas)

    if up_vid is not None:
        vid_sig = (
            f"{up_vid.name}_{up_vid.size}_{matkul}_{pertemuan}_{kapasitas}_"
            f"{selected_model_path}_{conf_thresh}_{iou_thresh}_{img_size}"
        )

        if st.session_state.get("processed_vid_sig") != vid_sig:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(up_vid.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stride = max(1, total_frames // 30) if total_frames > 0 else 5

            counts = []
            frame_idx = 0
            row_idx = None
            snap_path = None
            best_bgr = None
            best_frame_count = -1
            use_tracker = True

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % stride == 0:
                        if use_tracker:
                            try:
                                res = model.track(
                                    frame,
                                    classes=[0],
                                    conf=conf_thresh,
                                    iou=iou_thresh,
                                    imgsz=img_size,
                                    tracker=TRACKER_FILE,
                                    persist=True,
                                    verbose=False,
                                )[0]
                            except Exception:
                                use_tracker = False
                                res = model.predict(
                                    frame,
                                    classes=[0],
                                    conf=conf_thresh,
                                    iou=iou_thresh,
                                    imgsz=img_size,
                                    verbose=False,
                                )[0]
                        else:
                            res = model.predict(
                                frame,
                                classes=[0],
                                conf=conf_thresh,
                                iou=iou_thresh,
                                imgsz=img_size,
                                verbose=False,
                            )[0]

                        curr = len(res.boxes) if res.boxes is not None else 0
                        counts.append(curr)
                        peak_count = compute_sustained_peak(counts)

                        plotted_bgr = res.plot(line_width=2, font_size=0.5)
                        if curr >= best_frame_count or best_bgr is None:
                            best_frame_count = curr
                            best_bgr = plotted_bgr
                            snap_path = save_snapshot(best_bgr, existing_path=snap_path)

                        row_idx = upsert_history(
                            matkul,
                            pertemuan,
                            kapasitas,
                            peak_count,
                            "Video",
                            snap_path=snap_path,
                            row_idx=row_idx,
                        )

                        with vid_kpi.container():
                            render_telemetry_panel(peak_count, kapasitas)
                        frame_placeholder.image(
                            cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB), use_container_width=True
                        )

                    frame_idx += 1
            finally:
                cap.release()
                try:
                    os.remove(tfile.name)
                except OSError:
                    pass

            final_peak = compute_sustained_peak(counts)
            st.session_state["processed_vid_sig"] = vid_sig
            st.session_state["last_vid_count"] = final_peak
            st.session_state["last_vid_frame"] = (
                cv2.cvtColor(best_bgr, cv2.COLOR_BGR2RGB) if best_bgr is not None else None
            )

            with vid_kpi.container():
                render_telemetry_panel(final_peak, kapasitas)
            if st.session_state["last_vid_frame"] is not None:
                frame_placeholder.image(st.session_state["last_vid_frame"], use_container_width=True)
            st.toast(f"Terekam ke Riwayat: {matkul} (P{pertemuan:02d})")
        else:
            if st.session_state.get("last_vid_frame") is not None:
                frame_placeholder.image(st.session_state["last_vid_frame"], use_container_width=True)

# ----------------- TAB 4: RIWAYAT, KLIK SNAPSHOT & GRAFIK PER PERTEMUAN -----------------
with tab_history:
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        if "Snapshot" not in df.columns:
            df["Snapshot"] = ""

        f1, f2, f3 = st.columns([2.6, 1.1, 0.9])
        with f1:
            pilih_matkul = st.selectbox(
                "Filter Mata Kuliah",
                ["Semua Mata Kuliah"] + sorted(df["Mata Kuliah"].unique().tolist()),
                label_visibility="collapsed",
            )
        df_show = (
            df.reset_index(drop=True)
            if pilih_matkul == "Semua Mata Kuliah"
            else df[df["Mata Kuliah"] == pilih_matkul].reset_index(drop=True)
        )

        with f2:
            st.download_button(
                "Export CSV",
                df_show.drop(columns=["Snapshot"], errors="ignore").to_csv(index=False).encode("utf-8"),
                file_name=f"riwayat_{pilih_matkul.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with f3:
            if st.button("Reset Data", use_container_width=True):
                os.remove(HISTORY_FILE)
                for f_name in os.listdir(SNAPSHOT_DIR):
                    try:
                        os.remove(os.path.join(SNAPSHOT_DIR, f_name))
                    except OSError:
                        pass
                st.rerun()

        if len(df_show) > 0:
            df_chart = (
                df_show.drop_duplicates(subset=["Mata Kuliah", "Pertemuan"], keep="last")
                .sort_values("Pertemuan")
                .reset_index(drop=True)
            )
            pertemuan_order = sorted(df_chart["Pertemuan"].unique().tolist())

            g1, g2 = st.columns(2)

            if pilih_matkul != "Semua Mata Kuliah":
                with g1:
                    df_melt = df_chart.melt(
                        id_vars=["Pertemuan", "Mata Kuliah"],
                        value_vars=["Kapasitas", "Terdeteksi"],
                        var_name="Kategori",
                        value_name="Jumlah",
                    )
                    chart_bar = (
                        alt.Chart(df_melt)
                        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                        .encode(
                            x=alt.X(
                                "Pertemuan:N",
                                sort=pertemuan_order,
                                axis=alt.Axis(labelAngle=0, title="Pertemuan"),
                            ),
                            y=alt.Y("Jumlah:Q", title="Mahasiswa"),
                            color=alt.Color(
                                "Kategori:N",
                                scale=alt.Scale(
                                    domain=["Kapasitas", "Terdeteksi"],
                                    range=["#334155", "#10b981"],
                                ),
                                legend=alt.Legend(orient="top", title=None),
                            ),
                            xOffset="Kategori:N",
                            tooltip=["Mata Kuliah", "Pertemuan", "Kategori", "Jumlah"],
                        )
                        .properties(height=240, title="Kapasitas vs Terdeteksi per Pertemuan")
                    )
                    st.altair_chart(chart_bar, use_container_width=True)

                with g2:
                    chart_line = (
                        alt.Chart(df_chart)
                        .mark_area(
                            line={"color": "#06b6d4", "width": 3},
                            color=alt.Gradient(
                                gradient="linear",
                                stops=[
                                    alt.GradientStop(color="rgba(6, 182, 212, 0.35)", offset=0),
                                    alt.GradientStop(color="rgba(6, 182, 212, 0.0)", offset=1),
                                ],
                                x1=1,
                                x2=1,
                                y1=1,
                                y2=0,
                            ),
                            point=alt.OverlayMarkDef(color="#06b6d4", size=70),
                        )
                        .encode(
                            x=alt.X(
                                "Pertemuan:N",
                                sort=pertemuan_order,
                                axis=alt.Axis(labelAngle=0, title="Pertemuan"),
                            ),
                            y=alt.Y(
                                "Kehadiran (%):Q",
                                scale=alt.Scale(domain=[0, 100]),
                                title="Okupansi (%)",
                            ),
                            tooltip=[
                                "Mata Kuliah",
                                "Pertemuan",
                                "Terdeteksi",
                                "Kapasitas",
                                "Kehadiran (%)",
                                "Status",
                            ],
                        )
                        .properties(height=240, title="Tren Persentase Okupansi (%) per Pertemuan")
                    )
                    st.altair_chart(chart_line, use_container_width=True)
            else:
                with g1:
                    chart_bar_all = (
                        alt.Chart(df_chart)
                        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                        .encode(
                            x=alt.X(
                                "Pertemuan:N",
                                sort=pertemuan_order,
                                axis=alt.Axis(labelAngle=0, title="Pertemuan"),
                            ),
                            y=alt.Y("Terdeteksi:Q", title="Mahasiswa Terdeteksi"),
                            color=alt.Color(
                                "Mata Kuliah:N",
                                legend=alt.Legend(orient="top", title=None, columns=2),
                            ),
                            xOffset="Mata Kuliah:N",
                            tooltip=[
                                "Pertemuan",
                                "Mata Kuliah",
                                "Terdeteksi",
                                "Kapasitas",
                                "Kehadiran (%)",
                            ],
                        )
                        .properties(height=240, title="Mahasiswa Terdeteksi per Pertemuan")
                    )
                    st.altair_chart(chart_bar_all, use_container_width=True)

                with g2:
                    chart_line_all = (
                        alt.Chart(df_chart)
                        .mark_line(
                            strokeWidth=3,
                            point=alt.OverlayMarkDef(size=70),
                        )
                        .encode(
                            x=alt.X(
                                "Pertemuan:N",
                                sort=pertemuan_order,
                                axis=alt.Axis(labelAngle=0, title="Pertemuan"),
                            ),
                            y=alt.Y(
                                "Kehadiran (%):Q",
                                scale=alt.Scale(domain=[0, 100]),
                                title="Okupansi (%)",
                            ),
                            color=alt.Color(
                                "Mata Kuliah:N",
                                legend=alt.Legend(orient="top", title=None, columns=2),
                            ),
                            tooltip=[
                                "Pertemuan",
                                "Mata Kuliah",
                                "Terdeteksi",
                                "Kapasitas",
                                "Kehadiran (%)",
                                "Status",
                            ],
                        )
                        .properties(height=240, title="Tren Okupansi (%) per Pertemuan")
                    )
                    st.altair_chart(chart_line_all, use_container_width=True)

        col_tbl, col_snap = st.columns([1.8, 1.2])

        with col_tbl:
            st.caption("Klik salah satu baris (kolom Sumber) untuk melihat frame puncak stabil.")
            df_table = df_show.drop(columns=["Snapshot"], errors="ignore").copy()
            df_table["Sumber"] = "🔍 " + df_table["Sumber"].astype(str)

            event = st.dataframe(
                df_table,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "Kehadiran (%)": st.column_config.ProgressColumn(
                        "Okupansi",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )

        with col_snap:
            selected_rows = event.selection.rows if event and event.selection else []
            active_idx = selected_rows[0] if selected_rows else (len(df_show) - 1)
            selected_record = df_show.iloc[active_idx]
            snap_file_path = str(selected_record.get("Snapshot", ""))

            st.markdown('<div class="snap-box">', unsafe_allow_html=True)
            st.markdown(
                f"**Frame Puncak Stabil** • `{selected_record['Mata Kuliah']}` ({selected_record['Pertemuan']})"
            )
            if snap_file_path and os.path.exists(snap_file_path):
                st.image(snap_file_path, use_container_width=True)
                st.caption(
                    f"Sumber: {selected_record['Sumber']} | Terdeteksi: {selected_record['Terdeteksi']} / "
                    f"{selected_record['Kapasitas']} Mhs ({selected_record['Kehadiran (%)']}%)"
                )
            else:
                st.info("Snapshot tidak tersedia untuk sesi ini.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="text-align:center; padding: 3.5rem 1rem; color: #64748b; border: 1px dashed #1e293b; border-radius: 12px; margin-top: 1rem;">
                <div style="font-weight: 600; font-size: 1rem; color: #94a3b8;">Belum ada data riwayat</div>
                <div style="font-size: 0.85rem; margin-top: 4px;">Gunakan tab Kamera Live, Unggah Foto, atau Unggah Video untuk mulai merekam.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )