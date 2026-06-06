# Virtual Try-On - Project Changelog

This document provides a detailed overview of all the major changes, fixes, and improvements made to the Virtual Try-On project, specifying *where* the changes were made, *why* they were necessary, and showing the *actual code changes*.

---

## IDM-VTON Colab Stability & VRAM Optimization Updates
**Where:** IDM_VTON_Worker.ipynb
**Why:** The Google Colab T4 environment was throwing Out-of-Memory (OOM) errors, Gradio Schema parsing errors, and corrupted file downloads via wget.
**What was done:** 
- **Gradio Monkey Patch**: Reverted an incompatible schema check for Gradio 4.24.0.
- **Model Download Fix**: Replaced wget with huggingface_hub downloads to prevent truncated .pkl and .onnx files.
- **PyTorch Device Micro-Manager**: Explicitly unloaded massive components (like 	ext_encoder) to the CPU during massive VRAM spikes, while tricking diffusers into thinking the pipeline was still purely CUDA-based, completely eliminating T4 OOM crashes.


## 1. Cloud-Based API Engine (Google Colab Integration)
**Where:** `ml_ai/core/cloud_engine.py` (New file) and `IDM_VTON_Worker.ipynb`
**Why:** The local machine was running out of VRAM (Out of Memory error) when trying to run the full IDM-VTON model. 
**What was done:** Wrote a robust API handler to send images from the local Streamlit app to the Ngrok URL exposed by Colab.

**Code Change Example (`ml_ai/core/cloud_engine.py` lines 15-30):**
```python
from gradio_client import Client, handle_file

def call_cloud_api(person_img_path, garment_img_path, category="upper_body"):
    client = Client("yisol/IDM-VTON")
    dict_img = {
        "background": handle_file(person_img_path),
        "layers": [],
        "composite": None
    }
    
    result = client.predict(
        dict=dict_img,
        garm_img=handle_file(garment_img_path),
        garment_des="A stylish " + category,
        is_checked=True,
        is_checked_crop=False,
        denoise_steps=30,
        seed=42,
        api_name="/tryon"
    )
    return result
```

## 2. Premium User Interface Redesign (Glassmorphism & Orbs)
**Where:** `frontend/app.py`
**Why:** The original Streamlit UI looked very basic. The goal was to create a premium, modern aesthetic using HTML5 canvas and advanced CSS.
**What was done:** Replaced the default CSS with a new Glassmorphism theme and glowing radial gradient orbs.

**Code Change Example (`frontend/app.py` lines 54-150):**
```javascript
// Replaced sharp particles with soft radial gradients to fix Chromium rendering bugs
const colors = [
    {r: 139, g: 92, b: 246, a: 0.4}, // Purple
    {r: 56, g: 189, b: 248, a: 0.4}, // Light Blue
    {r: 236, g: 72, b: 153, a: 0.3}, // Pink
];

// Inside the Orb class draw method:
const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius);
gradient.addColorStop(0, `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, ${this.color.a})`);
gradient.addColorStop(0.5, `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, ${this.color.a * 0.5})`);
gradient.addColorStop(1, `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, 0)`);
```

**Code Change Example (`frontend/app.py` lines 152-250 CSS Injection):**
```css
/* Glassmorphism sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.4) !important;
    backdrop-filter: blur(24px) saturate(150%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(150%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Micro-animations on buttons */
.stButton > button {
    background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5) !important;
}
```

## 3. Persistent Browser Login
**Where:** `frontend/auth_browser.py` (New file) and `frontend/app.py`
**Why:** Users were logged out automatically if they refreshed the tab. 
**What was done:** Hooked Streamlit into the browser's native `sessionStorage` API.

**Code Change Example (`frontend/auth_browser.py` lines 20-35):**
```python
def persist_login(session_state, user_data: dict):
    """Saves the user data to Streamlit session and browser sessionStorage."""
    session_state["user"] = user_data
    session_state["logged_in"] = True
    
    # Inject JS to store token in the browser
    js_code = f"""
    <script>
        window.parent.sessionStorage.setItem("vton_auth", '{user_data.get("username")}');
    </script>
    """
    st.components.v1.html(js_code, height=0, width=0)
```

**Code Change Example (`frontend/app.py` lines 192-198):**
```python
if login_submit:
    success, message, user = authenticate_user(login_id, password)
    if success and user is not None:
        # Replaced standard login with persistent login
        persist_login(st.session_state, user)
        st.success("Login successful.")
        st.rerun()
```

## 4. Removal of Studio Mode / Background Remover
**Where:** `frontend/app.py`
**Why:** The Studio Mode was adding unnecessary complexity.
**What was done:** Stripped out the sidebar toggles and background processing logic.

**Code Change Example (`frontend/app.py` - Lines Removed):**
```python
# The following lines were completely removed from the sidebar UI logic:
# st.sidebar.header("Processing Options")
# use_studio_mode = st.sidebar.checkbox(
#     "Studio Mode (Remove Background)",
#     value=True,
#     help="Removes the background from the person image"
# )
```

## 5. Git Repository Cleanup & Sync
**Where:** `.gitignore`
**Why:** Massive temporary files were blocking GitHub pushes.
**What was done:** Explicitly blocked temp directories.

**Code Change Example (`.gitignore` lines 110-120):**
```text
# Virtual Try-On specific ignores
/temp/
/temp_idm_vton/
/temp_idm_vton_hf/
/database/data/tryon_debug/
/logs/
*.png
*.jpg
```

## 6. Fix: UI Overlap Bug (Material Icons Overwritten)
**Where:** `frontend/app.py`
**Why:** The new `Outfit` font was aggressively overwriting the internal Google Material Icons font that Streamlit uses for arrows and UI symbols, causing literal text like `_arrow_right_` to overlap on the screen.
**What was done:** Replaced the global CSS font override with a targeted one that protects icon classes.

**Code Change Example (`frontend/app.py` lines 185-200):**
```css
/* Old (Caused Overlap): */
* { font-family: 'Outfit', sans-serif !important; }

/* New (Fixed): */
html, body, [class*="css"]  {
    font-family: 'Outfit', sans-serif;
}
/* Protect Material Icons */
.material-icons, .material-symbols-rounded, [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}
```

## 7. Fix: Persistent Login JavaScript Deadlock
**Where:** `frontend/auth_browser.py` and `frontend/app.py`
**Why:** Refreshing the page still logged users out, and occasionally caused the entire app to go blank. This happened because `st.stop()` and `st.rerun()` were aborting the Python script *before* the Javascript injection could reach the browser to save the token.
**What was done:** Removed the early abort commands, allowing the UI to finish rendering so the browser could execute the injected `sessionStorage` scripts natively.

**Code Change Example (`frontend/app.py` lines 441-449):**
```python
if login_submit:
    success, message, user = authenticate_user(login_id, password)
    if success and user is not None:
        persist_login(st.session_state, user)
        # BUG FIX: Removed st.rerun() here to prevent aborting the Javascript session injection!
        st.success("Login successful. Initializing...")
```
**Code Change Example (`frontend/auth_browser.py` lines 45-53):**
```python
    # 1. First execution returns 0 while JS evaluates
    token = st_javascript(f"sessionStorage.getItem('{AUTH_COOKIE_NAME}')")
    
    # BUG FIX: Allow the script to continue so the iframe can mount
    if token == 0:
        return 
```

## 8. Dynamic Button Interactions & Micro-Animations
**Where:** `frontend/app.py`
**Why:** The user interface felt slightly static when clicking buttons. To improve the tactile feel, dynamic animations were required.
**What was done:** Added a `click-pulse` keyframe animation and a physical "squish" scale transform (`scale(0.95)`) to the CSS `:active` pseudo-class for all Streamlit buttons.

**Code Change Example (`frontend/app.py` lines 265-290):**
```css
@keyframes click-pulse {
    0% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.7); }
    70% { box-shadow: 0 0 0 15px rgba(139, 92, 246, 0); }
    100% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0); }
}

.stButton > button:active {
    transform: translateY(2px) scale(0.95) !important;
    box-shadow: 0 2px 10px rgba(139, 92, 246, 0.4) !important;
    animation: click-pulse 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    filter: brightness(1.2);
}
```

## 9. Removed Custom Garment Uploads & Lower Garment Filtering
**Where:** `frontend/app.py`
**Why:** To streamline the UI and simplify the user flow, the complex tab system separating catalog garments and custom image uploads was stripped out.
**What was done:** Removed `st.tabs(["From Catalog", "Upload Custom"])` and the `st.radio("Clothing Category", ["Uppers", "Lowers"])` filter. The app now directly displays a single dropdown containing all catalog garments immediately.

**Code Change Example (`frontend/app.py` lines 941-1025 Removed):**
```python
# The following complex tab and filtering logic was completely removed:
# tab_catalog, tab_custom = st.tabs(["From Catalog", "Upload Custom"])
# ...
# catalog_category = st.radio("Clothing Category", ["Uppers", "Lowers"], horizontal=True)
# ...
# custom_upload = st.file_uploader("Upload clothing image", type=["jpg", "jpeg", "png"])
```


