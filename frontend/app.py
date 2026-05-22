"""Virtual Try-On System - Streamlit Frontend
AI-Based Virtual Try-On and Fit Recommendation System
"""

import io
import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

from frontend.auth import (
    authenticate_user,
    create_user,
    init_auth_db,
    initialize_auth_session,
    is_session_valid,
    login_session,
    logout_session,
    request_password_reset,
    reset_password_with_token,
    update_user_profile,
)
from ml_ai.core.garment_manager import list_available_garments, load_garment_image, load_garment_metadata
from ml_ai.core.image_utils import load_image
from ml_ai.core.measurement_inference import infer_measurements, validate_measurements
from ml_ai.core.model_layer import load_models
from ml_ai.core.pose_detection import detect_pose
from ml_ai.core.segmentation import segment_body
from ml_ai.core.size_recommendation import explain_recommendation, recommend_size
from ml_ai.core.validation import validate_image


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Virtual Try-On",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_interactive_background():
    import streamlit.components.v1 as components
    js_code = """
    <script>
    const parentDoc = window.parent.document;
    if (!parentDoc.getElementById("particle-bg")) {
        const canvas = parentDoc.createElement("canvas");
        canvas.id = "particle-bg";
        canvas.style.position = "fixed";
        canvas.style.top = "0";
        canvas.style.left = "0";
        canvas.style.width = "100vw";
        canvas.style.height = "100vh";
        canvas.style.zIndex = "-1";
        canvas.style.pointerEvents = "none";
        parentDoc.body.prepend(canvas);

        const ctx = canvas.getContext("2d");
        let width = canvas.width = parentDoc.documentElement.clientWidth;
        let height = canvas.height = parentDoc.documentElement.clientHeight;

        let particles = [];
        let mouse = { x: width/2, y: height/2 };

        parentDoc.addEventListener("mousemove", (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        });

        parentDoc.defaultView.addEventListener("resize", () => {
            width = canvas.width = parentDoc.documentElement.clientWidth;
            height = canvas.height = parentDoc.documentElement.clientHeight;
        });

        class Particle {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 1.5;
                this.vy = (Math.random() - 0.5) * 1.5;
                this.radius = Math.random() * 2 + 1;
            }
            update() {
                this.x += this.vx;
                this.y += this.vy;
                if(this.x < 0 || this.x > width) this.vx = -this.vx;
                if(this.y < 0 || this.y > height) this.vy = -this.vy;
            }
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(31, 119, 180, 0.4)"; 
                ctx.fill();
            }
        }

        for(let i=0; i<60; i++) particles.push(new Particle());

        function animate() {
            ctx.clearRect(0, 0, width, height);
            
            for(let i=0; i<particles.length; i++) {
                particles[i].update();
                particles[i].draw();
                
                let dx = particles[i].x - mouse.x;
                let dy = particles[i].y - mouse.y;
                let dist = Math.sqrt(dx*dx + dy*dy);
                if(dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(mouse.x, mouse.y);
                    ctx.strokeStyle = `rgba(31, 119, 180, ${0.5 * (1 - dist/150)})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
                
                for(let j=i+1; j<particles.length; j++) {
                    let dx2 = particles[i].x - particles[j].x;
                    let dy2 = particles[i].y - particles[j].y;
                    let dist2 = Math.sqrt(dx2*dx2 + dy2*dy2);
                    if(dist2 < 120) {
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = `rgba(31, 119, 180, ${0.2 * (1 - dist2/120)})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }
            parentDoc.defaultView.requestAnimationFrame(animate);
        }
        animate();
    }
    </script>
    """
    components.html(js_code, height=0, width=0)

inject_interactive_background()

st.markdown(
    """
    <style>
    /* Make backgrounds transparent to reveal the canvas */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    .main { padding: 0rem 1rem; }
    .title { font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem; }
    .subtitle { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .measurement-box { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .success-box { background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
    .error-box { background-color: #f8d7da; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# AUTH PAGE
# ============================================================================

def render_auth_page() -> None:
    st.markdown('<p class="title">Secure Access</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Login or create an account to use the Virtual Try-On system</p>',
        unsafe_allow_html=True,
    )
    login_tab, register_tab, forgot_tab = st.tabs(["Login", "Register", "Forgot Password"])

    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            login_id = st.text_input("Email or Username")
            password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login")
        if login_submit:
            success, message, user = authenticate_user(login_id, password)
            if success and user is not None:
                login_session(st.session_state, user)
                st.success("Login successful.")
                st.rerun()
            else:
                st.error(message)

    with register_tab:
        with st.form("register_form", clear_on_submit=True):
            new_username = st.text_input("Username")
            new_email    = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_pw   = st.text_input("Confirm Password", type="password")
            register_submit = st.form_submit_button("Create Account")
        if register_submit:
            if new_password != confirm_pw:
                st.error("Passwords do not match.")
            else:
                success, message = create_user(new_username, new_email, new_password)
                st.success(message) if success else st.error(message)

    with forgot_tab:
        st.caption("Development mode: reset token shown here instead of email.")
        with st.form("forgot_password_form", clear_on_submit=False):
            forgot_login_id = st.text_input("Email or Username", key="forgot_login_id")
            forgot_submit   = st.form_submit_button("Generate Reset Token")
        if forgot_submit:
            success, message, reset_token = request_password_reset(forgot_login_id)
            if success:
                st.info(message)
                if reset_token:
                    st.code(reset_token)
                    st.warning("This token expires in 15 minutes.")
            else:
                st.error(message)

        with st.form("reset_password_form", clear_on_submit=True):
            reset_token_input = st.text_input("Reset Token")
            new_password      = st.text_input("New Password", type="password")
            confirm_pw        = st.text_input("Confirm New Password", type="password")
            reset_submit      = st.form_submit_button("Reset Password")
        if reset_submit:
            if new_password != confirm_pw:
                st.error("Passwords do not match.")
            else:
                success, message = reset_password_with_token(reset_token_input, new_password)
                st.success(message) if success else st.error(message)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_ai_models():
    try:
        return load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


@st.cache_resource(show_spinner=False)
def get_tryon_engine():
    """Load TryOnEngine once per session."""
    from ml_ai.core.tryon_engine import TryOnEngine
    return TryOnEngine()


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert BGR/BGRA numpy array to PIL Image."""
    if len(image.shape) == 3 and image.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def image_to_bytes(image: np.ndarray) -> bytes:
    """Convert BGR numpy array to PNG bytes."""
    pil_img = bgr_to_pil(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def process_user_image(image_path, user_height_cm: float = 0.0):
    """Process user image: detect pose, segment body, infer measurements."""
    try:
        seg_model, pose_model = load_ai_models()
        if seg_model is None or pose_model is None:
            st.error("Models not loaded")
            return None

        image = load_image(image_path)
        image_height = image.shape[0]

        with st.spinner("Segmenting body..."):
            seg_result = segment_body(image, seg_model)

        with st.spinner("Detecting pose..."):
            try:
                pose_result = detect_pose(image, pose_model)
            except RuntimeError as e:
                st.error(f"Pose detection failed: {e}")
                return None

        with st.spinner("Inferring measurements..."):
            measurements = infer_measurements(
                pose_result, seg_result,
                image_height=image_height,
                user_height_cm=user_height_cm
            )

        is_valid, errors = validate_measurements(measurements)
        if not is_valid:
            st.error(f"Measurement validation failed: {errors}")
            return None

        return {
            "image":        image,
            "measurements": measurements,
            "pose":         pose_result,
            "segmentation": seg_result,
        }

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


# ============================================================================
# SIDEBAR
# ============================================================================

init_auth_db()
initialize_auth_session(st.session_state)
session_valid = is_session_valid(st.session_state)

st.sidebar.markdown("# Virtual Try-On System")
st.sidebar.markdown("---")

if session_valid:
    user_info = st.session_state.get("auth_user") or {}
    
    prof_pic = user_info.get("profile_picture_path")
    if prof_pic and Path(prof_pic).exists():
        try:
            st.sidebar.image(prof_pic, width=150)
        except Exception:
            pass
            
    st.sidebar.success(f"Logged in as {user_info.get('username', 'user')}")
    if st.sidebar.button("Logout"):
        logout_session(st.session_state)
        st.rerun()

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Upload & Measure"
        
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("Upload & Measure", use_container_width=True):
        st.session_state.current_page = "Upload & Measure"
    if st.sidebar.button("Try-On", use_container_width=True):
        st.session_state.current_page = "Try-On"
    if st.sidebar.button("Garments", use_container_width=True):
        st.session_state.current_page = "Garments"

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**How it works:**\n\n"
        "1. Upload a photo of yourself\n"
        "2. We detect your body and pose\n"
        "3. We infer your measurements\n"
        "4. We recommend clothing sizes\n"
        "5. Try on different garments!"
    )
else:
    st.session_state.current_page = None
    st.sidebar.info("Please login to use the application.")


# ============================================================================
# PAGE: Upload & Measure
# ============================================================================

if not session_valid:
    render_auth_page()
    page = None
else:
    def go_to_profile():
        st.session_state.current_page = "My Profile"
        
    page = st.session_state.current_page

    col_space, col_pic, col_btn = st.columns([7, 1, 2])
    user_info = st.session_state.get("auth_user") or {}
    prof_pic = user_info.get("profile_picture_path")
    
    with col_pic:
        if prof_pic and Path(prof_pic).exists():
            import base64
            with open(prof_pic, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode()
            st.markdown(f'<div style="display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{b64}" style="width:45px;height:45px;border-radius:50%;object-fit:cover; border: 2px solid #1f77b4; margin-left: auto;"></div>', unsafe_allow_html=True)
        else:
            first_letter = user_info.get("username", "U")[0].upper()
            st.markdown(f'<div style="display: flex; justify-content: center; align-items: center; width:45px; height:45px; border-radius:50%; background-color:#1f77b4; color:white; font-size:24px; font-weight:bold; margin-left: auto;">{first_letter}</div>', unsafe_allow_html=True)
            
    with col_btn:
        st.button("⚙️ Profile", on_click=go_to_profile, use_container_width=True)

# ============================================================================
# PAGE: My Profile
# ============================================================================

if page == "My Profile":
    st.markdown('<p class="title">My Profile</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Manage your personal info and body measurements</p>', unsafe_allow_html=True)
    
    user_info = st.session_state.get("auth_user") or {}
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Personal Info")
        with st.form("personal_info_form"):
            new_email = st.text_input("Email", value=user_info.get("email", ""))
            new_phone = st.text_input("Phone Number", value=user_info.get("phone_number") or "")
            gender_options = ["", "Male", "Female", "Non-binary", "Other", "Prefer not to say"]
            current_gender = user_info.get("gender") or ""
            new_gender = st.selectbox("Gender", gender_options, index=gender_options.index(current_gender) if current_gender in gender_options else 0)
            
            current_dob = user_info.get("date_of_birth")
            new_dob = st.date_input("Date of Birth", value=None if not current_dob else datetime.fromisoformat(current_dob).date(), min_value=datetime(1900, 1, 1).date(), max_value=datetime.now().date())
            
            save_personal = st.form_submit_button("Save Personal Info", type="primary")
            if save_personal:
                updates = {
                    "email": new_email,
                    "phone_number": new_phone,
                    "gender": new_gender,
                    "date_of_birth": new_dob.isoformat() if new_dob else None
                }
                success, msg = update_user_profile(user_info["id"], updates)
                if success:
                    for k, v in updates.items():
                        st.session_state["auth_user"][k] = v
                    st.success("✅ Personal info updated!")
                    st.rerun()
                else:
                    st.error(msg)
                    
        st.subheader("Profile Picture")
        with st.form("profile_pic_form"):
            pic_upload = st.file_uploader("Upload new profile picture", type=["jpg", "jpeg", "png"])
            save_pic = st.form_submit_button("Upload Picture")
            if save_pic and pic_upload:
                save_dir = Path("database/data/profiles")
                save_dir.mkdir(parents=True, exist_ok=True)
                ext = pic_upload.name.split('.')[-1]
                save_path = save_dir / f"user_{user_info['id']}_profile.{ext}"
                with open(save_path, "wb") as f:
                    f.write(pic_upload.getbuffer())
                
                success, msg = update_user_profile(user_info["id"], {"profile_picture_path": str(save_path)})
                if success:
                    st.session_state["auth_user"]["profile_picture_path"] = str(save_path)
                    st.success("✅ Profile picture updated!")
                    st.rerun()
                else:
                    st.error(msg)
                    
    with c2:
        st.subheader("Body Measurements (cm)")
        with st.form("measurements_form"):
            new_height = st.number_input("Height", value=float(user_info.get("height_cm") or 170.0), min_value=50.0, max_value=300.0, step=1.0)
            new_shoulder = st.number_input("Shoulder Width", value=float(user_info.get("shoulder_width_cm") or 40.0), min_value=10.0, max_value=100.0, step=0.1)
            new_chest = st.number_input("Chest Circumference", value=float(user_info.get("chest_circumference_cm") or 90.0), min_value=30.0, max_value=200.0, step=0.1)
            new_torso = st.number_input("Torso Length", value=float(user_info.get("torso_length_cm") or 50.0), min_value=20.0, max_value=150.0, step=0.1)
            
            save_measurements = st.form_submit_button("Save Measurements", type="primary")
            if save_measurements:
                updates = {
                    "height_cm": new_height,
                    "shoulder_width_cm": new_shoulder,
                    "chest_circumference_cm": new_chest,
                    "torso_length_cm": new_torso
                }
                success, msg = update_user_profile(user_info["id"], updates)
                if success:
                    for k, v in updates.items():
                        st.session_state["auth_user"][k] = v
                    st.success("✅ Measurements updated!")
                    st.rerun()
                else:
                    st.error(msg)

elif page == "Upload & Measure":
    st.markdown('<p class="title">Upload & Measure</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a photo to measure your body</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Photo")
        
        user_info = st.session_state.get("auth_user") or {}
        default_height = float(user_info.get("height_cm") or 170.0)

        # ── Height input for accurate calibration ────────────────────
        st.markdown("#### 📏 Your Height")
        user_height_cm = st.number_input(
            "Enter your height (cm)",
            min_value=100.0,
            max_value=250.0,
            value=default_height,
            step=1.0,
            help="Your real height is used to calibrate accurate body measurements from the photo."
        )
        if user_height_cm > 0:
            st.success(f"✅ Height set: {user_height_cm:.0f} cm — measurements will be calibrated to your body")
        else:
            st.warning("⚠️ Without your height, measurements will be rough estimates only")

        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear front-facing full-body photo of yourself",
        )

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            validation = validate_image(tmp_path)

            if not validation.is_valid:
                st.error("Image validation failed")
                for error in validation.errors:
                    st.write(f"- {error}")
            else:
                if validation.warnings:
                    st.warning("Warnings:")
                    for warning in validation.warnings:
                        st.write(f"- {warning}")

                image = load_image(tmp_path)
                st.image(bgr_to_pil(image), caption="Uploaded image", use_column_width=True)

                if st.button("Analyze Photo", key="analyze_btn", type="primary"):
                    result = process_user_image(tmp_path, user_height_cm=user_height_cm)
                    if result:
                        st.session_state.result    = result
                        st.session_state.temp_path = tmp_path
                        st.success("✅ Image processed! Go to **Try-On** page to try garments.")

    with col2:
        st.subheader("Your Measurements")
        if "result" in st.session_state:
            result       = st.session_state.result
            measurements = result["measurements"]
            pose         = result["pose"]

            # Calibration status badge
            if measurements.calibration_method == 'height':
                st.success(f"✅ **Calibrated** — using your height ({measurements.user_height_cm:.0f} cm)")
            else:
                st.warning("⚠️ **Estimated** — enter your height for accurate measurements")

            st.markdown("#### Body Measurements")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Shoulder Width",  f"{measurements.shoulder_width_cm:.1f} cm")
                st.metric("Torso Length",    f"{measurements.torso_length_cm:.1f} cm")
            with col_m2:
                st.metric("Chest Circumference", f"{measurements.chest_circumference_cm:.1f} cm")
                st.metric("Confidence",          f"{measurements.confidence * 100:.1f}%")

            st.markdown("#### Pose Analysis")
            st.write(f"**Is Frontal:** {'Yes' if pose.is_frontal else 'No'}")
            st.write(f"**Shoulder Width (px):** {pose.shoulder_width_px:.1f}")
            st.write(f"**Keypoints Detected:** {len(pose.keypoints)}")

            if pose.warnings:
                st.warning("**Pose Warnings:**")
                for w in pose.warnings:
                    st.write(f"- {w}")
                    
            st.markdown("---")
            if st.button("💾 Save these measurements to My Profile", type="primary"):
                user_info = st.session_state.get("auth_user") or {}
                if user_info:
                    new_data = {
                        "shoulder_width_cm": round(measurements.shoulder_width_cm, 1),
                        "chest_circumference_cm": round(measurements.chest_circumference_cm, 1),
                        "torso_length_cm": round(measurements.torso_length_cm, 1),
                        "height_cm": user_height_cm,
                    }
                    success, msg = update_user_profile(user_info["id"], new_data)
                    if success:
                        for k, v in new_data.items():
                            st.session_state["auth_user"][k] = v
                        st.success("✅ Measurements saved to your profile!")
                    else:
                        st.error(msg)
        else:
            st.info("Upload and analyze a photo to see measurements here.")


# ============================================================================
# PAGE: Try-On
# ============================================================================

elif page == "Try-On":
    st.markdown('<p class="title">Virtual Try-On</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">See how garments fit you</p>', unsafe_allow_html=True)

    user_info = st.session_state.get("auth_user") or {}
    has_photo_result = "result" in st.session_state and "temp_path" in st.session_state
    
    # ── Guard ────────────────────────────────────────────────────────
    if not has_photo_result and not user_info.get("chest_circumference_cm"):
        st.warning("⚠️ Please upload a photo on the **Upload & Measure** page or save your measurements in **My Profile**.")
        st.stop()
        
    if has_photo_result:
        result       = st.session_state.result
        measurements = result["measurements"]
        temp_path    = st.session_state.temp_path
    else:
        # Create dummy measurements object from profile
        from ml_ai.core.measurement_inference import BodyMeasurements
        measurements = BodyMeasurements(
            shoulder_width_cm=user_info.get("shoulder_width_cm", 0.0),
            chest_circumference_cm=user_info.get("chest_circumference_cm", 0.0),
            torso_length_cm=user_info.get("torso_length_cm", 0.0),
            user_height_cm=user_info.get("height_cm", 0.0),
            calibration_method="profile",
            confidence=1.0,
            warnings=[]
        )
        temp_path = None
        st.info("💡 Using your saved profile measurements for size recommendations. To see a visual try-on, please upload a photo.")

    # ── TEMP DEBUG: expand this to see keypoint coordinates ──────────
    if has_photo_result:
        pose = result["pose"]
        with st.expander("🔍 Debug: Detected Keypoints (expand to see)", expanded=False):
            st.write(f"**Total keypoints detected:** {len(pose.keypoints)}")
            st.write(f"**Is frontal:** {pose.is_frontal}")
            st.write(f"**Shoulder width (px):** {pose.shoulder_width_px:.1f}")
            st.markdown("---")
            for kp in pose.keypoints:
                st.write(f"**{kp.name}** → x={kp.x:.0f}, y={kp.y:.0f}, conf={kp.confidence:.2f}")

    garments = list_available_garments()
    if not garments:
        st.error("No garments available in the database.")
        st.stop()

    # ── Layout ───────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.subheader("🛍️ Select Garment")

        selected_garment = st.selectbox("Choose garment:", garments, key="tryon_garment_select")

        try:
            metadata   = load_garment_metadata(selected_garment)
            size_chart = metadata.get("size_chart", {})
        except FileNotFoundError:
            st.error(f"Garment not found: {selected_garment}")
            st.stop()

        # Garment thumbnail
        try:
            garment_img = load_garment_image(selected_garment)
            st.image(bgr_to_pil(garment_img), width=220, caption=metadata.get("name", selected_garment))
        except Exception:
            st.caption("(Preview unavailable)")

        # Garment details
        with st.expander("📋 Garment Details", expanded=False):
            st.write(f"**Name:** {metadata.get('name', 'N/A')}")
            st.write(f"**Brand:** {metadata.get('brand', 'N/A')}")
            st.write(f"**Category:** {metadata.get('category', 'N/A')}")
            st.write(f"**Material:** {metadata.get('material', 'N/A')}")
            st.write(f"**Price:** ${metadata.get('price_usd', 0):.2f}")
            colors = metadata.get("available_colors", [])
            if colors:
                st.write(f"**Colors:** {', '.join(colors)}")

        # Size recommendation
        if size_chart:
            recommendation = recommend_size(measurements, size_chart)
            st.success(
                f"📏 Recommended Size: **{recommendation.size}** "
                f"({recommendation.confidence * 100:.0f}% confidence)"
            )

            with st.expander("📊 All Size Fit Scores", expanded=False):
                for size in sorted(recommendation.fit_scores.keys()):
                    score = recommendation.fit_scores[size] * 100
                    st.write(f"**{size}:** {score:.1f}%")
                    st.progress(min(recommendation.fit_scores[size], 1.0))

            with st.expander("📐 Size Chart", expanded=False):
                size_chart_data = []
                for size in sorted(size_chart.keys()):
                    m = size_chart[size]
                    size_chart_data.append({
                        "Size":          size,
                        "Shoulder (cm)": m.get("shoulder_width_cm", "N/A"),
                        "Chest (cm)":    m.get("chest_circumference_cm", "N/A"),
                        "Torso (cm)":    m.get("torso_length_cm", "N/A"),
                    })
                st.dataframe(size_chart_data, use_container_width=True)

        # Fit controls
        st.subheader("⚙️ Fit Settings")
        blend_alpha    = st.slider("Garment opacity",  0.5,  1.0,  1.0, 0.01, key="blend_alpha")
        shoulder_scale = st.slider("Fit width",        0.85, 1.20, 1.00, 0.01, key="shoulder_scale",
                                   help="1.00 = exact fit | >1.00 = looser | <1.00 = tighter")

        st.markdown(" ")
        run_tryon = st.button("✨ Try It On", type="primary", use_container_width=True, key="tryon_btn")

    # ── Right column: result ─────────────────────────────────────────
    with col_right:
        st.subheader("🪞 Try-On Preview")

        result_key = f"tryon_result_{selected_garment}"

        if run_tryon:
            if not has_photo_result:
                st.error("❌ You must upload a photo on the **Upload & Measure** page before you can visually try on a garment!")
            else:
                with st.spinner("Warping garment to your body shape…"):
                    try:
                        person_img  = load_image(temp_path)
                        garment_img = load_garment_image(selected_garment)
                        category    = metadata.get("category", "tshirt").lower()

                        try:
                            from ml_ai.core.garment_manager import load_garment_mask
                            garment_mask_img = load_garment_mask(selected_garment)
                        except FileNotFoundError:
                            garment_mask_img = None

                        engine       = get_tryon_engine()
                        tryon_result = engine.run(
                            person_image=person_img,
                            garment_image=garment_img,
                            garment_category=category,
                            blend_alpha=blend_alpha,
                            shoulder_scale=shoulder_scale,
                            use_segmentation_mask=True,
                            garment_mask=garment_mask_img,
                        )

                        if tryon_result.success and tryon_result.composite_image is not None:
                            st.session_state[result_key] = {
                                "success":           True,
                                "composite_bytes":   image_to_bytes(tryon_result.composite_image),
                                "processing_time_s": tryon_result.processing_time_s,
                                "warnings":          tryon_result.warnings,
                            }
                        else:
                            st.session_state[result_key] = {
                                "success":  False,
                                "error":    tryon_result.error,
                                "warnings": tryon_result.warnings,
                            }

                    except Exception as e:
                        st.session_state[result_key] = {
                            "success":  False,
                            "error":    str(e),
                            "warnings": [],
                        }

        # Display result
        tryon_data = st.session_state.get(result_key)

        if tryon_data is None:
            try:
                person_img = load_image(temp_path)
                st.image(bgr_to_pil(person_img), caption="Your photo — press '✨ Try It On'", use_column_width=True)
            except Exception:
                st.info("Press **✨ Try It On** to see the result here.")

        elif tryon_data.get("success"):
            composite_bytes = tryon_data["composite_bytes"]
            st.image(composite_bytes, caption=f"Wearing: {metadata.get('name', selected_garment)}", use_column_width=True)

            proc_time = tryon_data.get("processing_time_s", 0)
            warnings  = tryon_data.get("warnings", [])
            st.caption(f"⏱️ Processed in {proc_time:.2f}s")

            if warnings:
                with st.expander("⚠️ Warnings", expanded=False):
                    for w in warnings:
                        st.warning(w)

            st.download_button(
                label="⬇️ Download Try-On Image",
                data=composite_bytes,
                file_name=f"tryon_{selected_garment}.png",
                mime="image/png",
                use_container_width=True,
                key=f"download_{selected_garment}"
            )

            with st.expander("🔍 Before / After Comparison", expanded=True):

                try:
                    # Original image
                    person_img = load_image(temp_path)

                    # Convert images to PIL
                    before_img = bgr_to_pil(person_img)
                    after_img = Image.open(io.BytesIO(composite_bytes))

                    # Interactive slider
                    image_comparison(
                        img1=before_img,
                        img2=after_img,
                        label1="Before",
                        label2="After",
                        width=700,
                        starting_position=50,
                        show_labels=True,
                        make_responsive=True,
                    )

                except Exception as e:
                    st.error(f"Comparison failed: {e}")

            # ── Debug visualization ──────────────────────────────────────
            with st.expander("🛠️ Debug: Pipeline Stage Outputs", expanded=False):
                debug_dir = Path("database/data/tryon_debug")
                if not debug_dir.exists():
                    debug_dir = Path("data/tryon_debug")
                if debug_dir.exists():
                    d1, d2, d3 = st.columns(3)
                    wg_path = debug_dir / "debug_warped_garment.png"
                    wm_path = debug_dir / "debug_warped_mask.png"
                    dc_path = debug_dir / "debug_composite.png"
                    if wg_path.exists():
                        with d1:
                            wg = cv2.imread(str(wg_path), cv2.IMREAD_UNCHANGED)
                            st.image(bgr_to_pil(wg), caption="Warped Garment", use_column_width=True)
                    if wm_path.exists():
                        with d2:
                            wm = cv2.imread(str(wm_path), cv2.IMREAD_GRAYSCALE)
                            st.image(wm, caption="Warped Mask", use_column_width=True)
                    if dc_path.exists():
                        with d3:
                            dc = cv2.imread(str(dc_path))
                            st.image(bgr_to_pil(dc), caption="Final Composite", use_column_width=True)
                else:
                    st.info("Debug images not available yet. Run a try-on first.")

        else:
            st.error(f"❌ Try-on failed: {tryon_data.get('error', 'Unknown error')}")
            if tryon_data.get("warnings"):
                for w in tryon_data["warnings"]:
                    st.warning(w)
            if st.button("🔄 Try Again", key="retry_btn"):
                st.session_state.pop(result_key, None)
                st.rerun()


# ============================================================================
# PAGE: Garments
# ============================================================================

elif page == "Garments":
    st.markdown('<p class="title">Browse Garments</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Explore available clothing items</p>', unsafe_allow_html=True)

    garments = list_available_garments()
    if not garments:
        st.error("No garments available")
        st.stop()

    st.subheader(f"Available Garments ({len(garments)})")
    cols = st.columns(3)

    for idx, garment_id in enumerate(garments):
        try:
            metadata = load_garment_metadata(garment_id)
            with cols[idx % 3]:
                st.markdown(f"### {metadata.get('name', garment_id)}")
                try:
                    garment_img = load_garment_image(garment_id)
                    st.image(bgr_to_pil(garment_img), use_column_width=True,
                             caption=metadata.get("name", garment_id))
                except Exception:
                    st.info("No image available")

                st.write(f"**Brand:** {metadata.get('brand', 'N/A')}")
                st.write(f"**Category:** {metadata.get('category', 'N/A')}")
                st.write(f"**Price:** ${metadata.get('price_usd', 0):.2f}")
                colors = metadata.get("available_colors", [])
                if colors:
                    st.write(f"**Colors:** {', '.join(colors)}")

                if st.button(f"Try {metadata.get('name', 'this')}", key=f"try_{garment_id}"):
                    st.session_state.selected_garment = garment_id
                    st.rerun()

        except Exception as e:
            st.error(f"Error loading {garment_id}: {e}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>AI-Based Virtual Try-On and Fit Recommendation System v0.1.0</p>
        <p>Built with Streamlit, OpenCV, MediaPipe and TPS Warping</p>
    </div>
    """,
    unsafe_allow_html=True,
)