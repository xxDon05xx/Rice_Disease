import os
import numpy as np
import cv2
import httpx
import urllib3
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
from supabase import create_client, Client
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms, models

app = Flask(__name__)

# --- CONFIGURATION ---
CHECKPOINT    = 'resnet_rice_best_fixed.pth'
DATA_DIR      = 'Rice_Leaf_AUG'
BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'rice_doctor_super_secret_key_123'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- SUPABASE ---
SUPABASE_URL = 'https://eclthmxchjqvhyodvaea.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVjbHRobXhjaGpxdmh5b2R2YWVhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE5NDI4NzEsImV4cCI6MjA4NzUxODg3MX0.-ayZRmOcT6bzEUYhQ8zFaMJt8TonRNaL88KSGLBTsWU'

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
try:
    from supabase.client import ClientOptions
    import httpcore
    # Force IPv4 to avoid IPv6 timeout issues on some networks
    transport = httpx.HTTPTransport(
        verify=False,
        local_address="0.0.0.0"   # forces IPv4
    )
    supabase: Client = create_client(
        SUPABASE_URL, SUPABASE_KEY,
        options=ClientOptions(
            httpx_client=httpx.Client(transport=transport, timeout=30.0)
        )
    )
except Exception:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print('Connected to Supabase!')

# ─────────────────────────────────────────────────────────────
# CLASS MAPPING  (8-class Flask app  →  6-class ResNet model)
#
#  Old 8 classes                   →  New 6 classes
#  ──────────────────────────────────────────────────────────
#  Bacterial Leaf Blight           →  Bacterial Leaf Blight  (same)
#  Brown Spot                      →  Brown Spot             (same)
#  Healthy Rice Leaf               →  Healthy Rice Leaf      (same)
#  Leaf Blast                      →  Leaf Blast             (same)
#  Leaf scald                      →  Leaf scald             (same)
#  Narrow Brown Leaf Spot          →  Brown Spot             (merged – visually similar)
#  Rice Hispa                      →  Leaf Blast             (merged – no separate class)
#  Sheath Blight                   →  Sheath Blight          (same)
# ─────────────────────────────────────────────────────────────

IMG_SIZE = 224
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The 6 classes our ResNet was trained on (must match checkpoint order)
CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight",
]

val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# --- LOAD PyTorch ResNet50 MODEL ---
def load_model():
    ckpt        = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    class_names = ckpt.get("classes", CLASS_NAMES)
    m           = models.resnet50(weights=None)
    m.fc        = nn.Linear(m.fc.in_features, len(class_names))
    m.load_state_dict(ckpt["model_state"])
    m.eval().to(DEVICE)
    return m, class_names


print("Loading AI model...")
model_obj   = None
class_names = CLASS_NAMES
try:
    model_obj, class_names = load_model()
    print(f"Model loaded. Classes: {class_names}")
except Exception as e:
    print(f"Error loading model: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_and_predict(image_path):
    """Run PyTorch inference; returns list of (class, prob) sorted desc."""
    img    = Image.open(image_path).convert("RGB")
    tensor = val_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model_obj(tensor), dim=1).squeeze().cpu().numpy()
    order = np.argsort(probs)[::-1]
    return [(class_names[i], float(probs[i])) for i in order]


# cv2 imported at top of file

def analyze_severity(image_path, disease_class):
    """
    Enhanced Severity Analysis.
    Calculates severity based on precise OpenCV contours:
    - Number of spots/lesions
    - Total affected area spread
    - Browning/Yellowing intensity
    """
    if disease_class == 'Healthy Rice Leaf':
        return 0.0, 'ആരോഗ്യകരം'
        
    try:
        # 1. Load the image with OpenCV and convert to different color spaces
        img = cv2.imread(image_path)
        if img is None:
            return 0.0, 'അജ്ഞാതം'
            
        img = cv2.resize(img, (256, 256))
        
        # Convert BGR to HSV for color masking
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Convert BGR to Grayscale for structural / edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Extract the leaf from the background (Green + Yellow/Brown spectrums)
        # Background is usually dark or bright white, so we filter by saturation and value
        lower_leaf = np.array([0, 25, 25])
        upper_leaf = np.array([180, 255, 255])
        leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)
        
        total_leaf_pixels = cv2.countNonZero(leaf_mask)
        
        if total_leaf_pixels < 500: # Increase threshold for a valid leaf
            return 0.0, 'അപര്യാപ്തം'

        # 3. Detect Diseased Areas (Browning, Yellowing, Dark Spots, Lesions)
        # Hue ranges: 
        # Brown/Yellow ~ 10-40
        # Dark Spots ~ Low Value
        # White/Gray Scalds ~ Low Saturation, High Value
        
        # Mask 1: Brown & Yellow (Rust / Spots / Blight)
        lower_brown = np.array([10, 40, 40])
        upper_brown = np.array([45, 255, 255])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Mask 2: Dark Lesions
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 70])
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Mask 3: Pale/White Scalds
        lower_pale = np.array([0, 0, 180])
        upper_pale = np.array([180, 40, 255])
        pale_mask = cv2.inRange(hsv, lower_pale, upper_pale)
        
        # Combine masks
        disease_mask = cv2.bitwise_or(brown_mask, dark_mask)
        disease_mask = cv2.bitwise_or(disease_mask, pale_mask)
        
        # Only consider disease that is actually ON the leaf
        disease_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=leaf_mask)

        # 4. Morphological Operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 5. Contour Analysis (Find spots and measure spread)
        contours, hierarchy = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        number_of_spots = 0
        total_disease_area_from_contours = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5: # Ignore tiny noise dots
                number_of_spots += 1
                total_disease_area_from_contours += area

        # Calculate exact spread
        area_spread_ratio = total_disease_area_from_contours / total_leaf_pixels
        
        # Calculate color intensity (browning) severity within the diseased areas
        # We look at the average 'Value' and 'Saturation' of the brown areas
        mean_val = cv2.mean(img, mask=disease_mask)
        
        # 6. Weighted Severity Score Calculation
        # - 70% weight to standard area spread ratio
        # - 30% weight to number of spots (a high number of distinct spots indicates rapid spreading like Brown Spot / Blast)
        
        # Normalize spot count impact (assuming 50 spots is extremely severe)
        spot_severity = min(number_of_spots / 50.0, 1.0) 
        
        raw_severity = (area_spread_ratio * 0.70) + (spot_severity * 0.30)
        severity_pct = round(min(max(raw_severity * 100, 0), 100), 1)

        if severity_pct < 15:
            label = 'കുറവ്'
        elif severity_pct < 35:
            label = 'മിതമായ'
        elif severity_pct < 60:
            label = 'ഉയർന്ന'
        else:
            label = 'ഗുരുതരം'

        return severity_pct, label
    except Exception as e:
        print(f"Severity analysis error: {e}")
        return 0.0, 'അജ്ഞാതം'


def get_weather_integrated_recommendation(disease, temp_c, humidity, is_rain):
    """Weather-aware recommendation (same logic as original, mapped to 6 classes)."""
    hot      = temp_c is not None and temp_c >= 32
    humid    = humidity is not None and humidity >= 80
    cool_dry = temp_c is not None and temp_c < 24 and (humidity is None or humidity < 60)
    if is_rain:           scenario = 'rainy'
    elif hot and humid:   scenario = 'hot_humid'
    elif humid:           scenario = 'humid'
    elif hot:             scenario = 'hot'
    elif cool_dry:        scenario = 'cool_dry'
    else:                 scenario = 'normal'
    t = round(temp_c) if temp_c else '?'
    h = int(humidity)  if humidity else '?'

    matrix = {
        'Bacterial Leaf Blight': {
            'rainy':     '🌧️ മഴ ഉള്ളതിനാൽ ഇന്ന് കെമിക്കൽ തളിക്കൽ ഒഴിവാക്കുക. ജലനിർഗ്ഗമനം ഉറപ്പാക്കുക, ബാധിത ചെടികൾ നീക്കം ചെയ്യുക. മഴ ശമിച്ചശേഷം കോപ്പർ ഓക്‌സിക്ലോറൈഡ് ഉപയോഗിക്കുക.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — ബാക്ടീരിയ ദ്രുതഗതിയിൽ പടരാൻ സാധ്യതയുണ്ട്. ഉടൻ കോപ്പർ ഓക്‌സിക്ലോറൈഡ് വൈകുന്നേരം തളിക്കുക.',
            'humid':     f'💧 {h}% ആർദ്രത — കോപ്പർ കുമിൾനാശിനി തളിക്കുക. നൈട്രജൻ വളം നിർത്തുക.',
            'hot':       f'🌡️ {t}°C — രാവിലെ 7–9 മണിക്ക് കോപ്പർ ഓക്‌സിക്ലോറൈഡ് തളിക്കുക.',
            'cool_dry':  f'❄️ {t}°C — ഭീഷണി കുറവ്. കോപ്പർ കുമിൾനാശിനി പ്രതിരോധമായി ഉപയോഗിക്കുക.',
            'normal':    'കോപ്പർ ഓക്‌സിക്ലോറൈഡ് തളിക്കുക. ജലനിർഗ്ഗമനം ഉറപ്പാക്കുക.',
        },
        'Brown Spot': {
            'rainy':     '🌧️ മഴ ശമിച്ചശേഷം മാൻകോസേബ് ഉപയോഗിക്കുക. ജലനിർഗ്ഗമനം ഉറപ്പാക്കുക.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — ഫംഗൽ ഭീഷണി കൂടുതലാണ്. ഉടൻ മാൻകോസേബ് തളിക്കുക.',
            'humid':     f'💧 {h}% — മാൻകോസേബ് (2.5 g/L) വൈകുന്നേരം തളിക്കുക.',
            'hot':       f'🌡️ {t}°C — ട്രൈസൈക്ലസോൾ ഉപയോഗിക്കുക. പൊട്ടാഷ്-സിലിക്കൺ വളങ്ങൾ നൽകുക.',
            'cool_dry':  f'❄️ {t}°C — ഒരു ഫംഗിസൈഡ് ഉപയോഗിക്കുക.',
            'normal':    'മാൻകോസേബ് അല്ലെങ്കിൽ ട്രൈസൈക്ലസോൾ ഉപയോഗിക്കുക.',
        },
        'Healthy Rice Leaf': {
            'rainy':     '🌧️ ചെടി ആരോഗ്യകരമാണ്! ജലനിർഗ്ഗമനം ഉറപ്പാക്കി ആഴ്ചതോറും ഇലകൾ നിരീക്ഷിക്കുക.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — ഈ കാലാവസ്ഥ ബ്ലാസ്റ്റ് ഭീഷണി ഉയർത്താം. ആഴ്ചതോറും പരിശോധിക്കുക.',
            'humid':     f'💧 {h}% — ഫംഗൽ ഭീഷണി ഉണ്ടാകാം; ഇലകൾ നിരീക്ഷിക്കുക.',
            'hot':       f'🌡️ {t}°C — ഉച്ചസമയത്ത് ജലസേചനം ഒഴിവാക്കുക.',
            'cool_dry':  f'✅ {t}°C — കൃഷി ഇതുപോലെ തുടരുക.',
            'normal':    '✅ ചെടി ആരോഗ്യകരമാണ്! കൃത്യമായ ജലം, വളം, കളനാശിനി തുടരുക.',
        },
        'Leaf Blast': {
            'rainy':     '🌧️ മഴ ശമിച്ചശേഷം ഉടൻ ട്രൈസൈക്ലസോൾ തളിക്കുക.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — ഏറ്റവും ഗുരുതരമായ അവസ്ഥ. ഉടൻ ട്രൈസൈക്ലസോൾ (0.6 g/L) തളിക്കുക.',
            'humid':     f'💧 {h}% — ഇന്ന് വൈകുന്നേരം ട്രൈസൈക്ലസോൾ തളിക്കുക.',
            'hot':       f'🌡️ {t}°C — രാവിലെ അല്ലെങ്കിൽ വൈകുന്നേരം ഉപയോഗിക്കുക.',
            'cool_dry':  f'❄️ {t}°C — ട്രൈസൈക്ലസോൾ ഒരു തവണ ഉപയോഗിക്കുക.',
            'normal':    'ട്രൈസൈക്ലസോൾ ഫംഗിസൈഡ് ഉടൻ ഉപയോഗിക്കുക. നൈട്രജൻ വളം കുറയ്ക്കുക.',
        },
        'Leaf scald': {
            'rainy':     '🌧️ ബാധിത ഇലകൾ നീക്കം ചെയ്യുക. മഴ ശമിച്ചശേഷം Propiconazole ഉപയോഗിക്കുക.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — Propiconazole അല്ലെങ്കിൽ Tebuconazole വൈകുന്നേരം തളിക്കുക.',
            'humid':     f'💧 {h}% — ബാധിത ഭാഗങ്ങൾ നീക്കം ചെയ്ത് Propiconazole തളിക്കുക.',
            'hot':       f'🌡️ {t}°C — Propiconazole ഉടൻ ഉപയോഗിക്കുക.',
            'cool_dry':  f'❄️ {t}°C — ബാധിത ഭാഗങ്ങൾ നീക്കം ചെയ്ത് ഒരു ഫംഗിസൈഡ് തളിക്കുക.',
            'normal':    'ബാധിത ഭാഗങ്ങൾ നീക്കം ചെയ്ത് Propiconazole ഉപയോഗിക്കുക.',
        },
        'Sheath Blight': {
            'rainy':     '🌧️ മഴ ശമിച്ചശേഷം Hexaconazole ഉപയോഗിക്കുക.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — ഉടൻ Hexaconazole അല്ലെങ്കിൽ Trichoderma ഉപയോഗിക്കുക.',
            'humid':     f'💧 {h}% — Hexaconazole ഉപയോഗിക്കുക. ചെടി നിബിഡത കുറയ്ക്കുക.',
            'hot':       f'🌡️ {t}°C — ഉടൻ Hexaconazole ഉപയോഗിക്കുക.',
            'cool_dry':  f'❄️ {t}°C — ഫംഗൽ ഭീഷണി കുറവ്. ഒരു ഫംഗിസൈഡ് തളിക്കുക.',
            'normal':    'Hexaconazole അല്ലെങ്കിൽ Trichoderma ഉപയോഗിക്കുക. നൈട്രജൻ വളം നിയന്ത്രിക്കുക.',
        },
    }
    dm = matrix.get(disease)
    if not dm:
        return None
    return dm.get(scenario, dm.get('normal'))


# ── ROUTES ────────────────────────────────────────────────────

@app.route('/')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/scan')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            result = supabase.table('users').select('*').eq('username', username).execute()
            user   = result.data[0] if result.data else None
            if user and check_password_hash(user['password'], password):
                session['username'] = username
                return redirect(url_for('index'))
            else:
                flash('തെറ്റായ യൂസർനെയിം അല്ലെങ്കിൽ പാസ്\u200cവേഡ് (Invalid Username or Password)')
        except Exception as e:
            print(f'Login DB error: {e}')
            flash('സെർവർ കണക്ഷൻ പിശക്. വീണ്ടും ശ്രമിക്കുക. (Server connection error)')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            existing = supabase.table('users').select('id').eq('username', username).execute()
            if existing.data:
                flash('യൂസർനെയിം നിലവിലുണ്ട് (Username already exists)')
                return render_template('signup.html')
            hashed_pw = generate_password_hash(password)
            supabase.table('users').insert({'username': username, 'password': hashed_pw}).execute()
            flash('അക്കൗണ്ട് വിജയകരമായി നിർമ്മിച്ചു! ദയവായി ലോഗിൻ ചെയ്യുക.')
            return redirect(url_for('login'))
        except Exception as e:
            print(f'Signup DB error: {e}')
            flash('സെർവർ കണക്ഷൻ പിശക്. വീണ്ടും ശ്രമിക്കുക. (Server connection error)')
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Invalid file type'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'})

    if model_obj is None:
        return jsonify({'error': 'Model not loaded'})

    try:
        results        = prepare_and_predict(filepath)
        predicted_class, confidence_raw = results[0]
        confidence     = round(confidence_raw * 100, 2)

        # Malayalam translations for 6 classes
        translations = {
            'Bacterial Leaf Blight': 'ബാക്ടീരിയൽ ലീഫ് ബ്ലൈറ്റ്',
            'Brown Spot':            'തവിട്ടു പുള്ളി രോഗം',
            'Healthy Rice Leaf':     'ആരോഗ്യമുള്ള ഇല',
            'Leaf Blast':            'ബ്ലാസ്റ്റ് രോഗം',
            'Leaf scald':            'ഇല കരിച്ചിൽ',
            'Sheath Blight':         'പോള രോഗം',
        }

        # Default recommendations for 6 classes
        recommendations = {
            'Bacterial Leaf Blight': 'ചെമ്പ് അടങ്ങിയ കുമിൾനാശിനികൾ (കോപ്പർ ഓക്‌സിക്ലോറൈഡ്) തളിക്കുക. രോഗബാധിത ചെടികൾ നീക്കം ചെയ്യുക. നൈട്രജൻ വളം ആവശ്യത്തിൽ കൂടുതൽ ഉപയോഗിക്കരുത്.',
            'Brown Spot':            'മാൻകോസേബ് അല്ലെങ്കിൽ ട്രൈസൈക്ലസോൾ കൊണ്ടുള്ള കുമിൾനാശിനി തളിക്കുക. സിലിക്കൺ, പൊട്ടാഷ് എന്നിവ അടങ്ങിയ വളങ്ങൾ ഉപയോഗിക്കുക.',
            'Healthy Rice Leaf':     'ചെടി ആരോഗ്യത്തോടെ ഇരിക്കുന്നു! പതിവ് നനയ്ക്കൽ, വളം ഇടൽ, കളനാശിനി ഉപയോഗം തുടരുക.',
            'Leaf Blast':            'ട്രൈസൈക്ലസോൾ (Tricyclazole) കൊണ്ടുള്ള കുമിൾനാശിനി ഉടൻ തളിക്കുക. നൈട്രജൻ വളം കുറയ്ക്കുക.',
            'Leaf scald':            'ബാധിത ഭാഗങ്ങൾ നീക്കം ചെയ്ത് നശിപ്പിക്കുക. Propiconazole കുമിൾനാശിനി തളിക്കുക.',
            'Sheath Blight':         'Hexaconazole അല്ലെങ്കിൽ Trichoderma ഉപയോഗിക്കുക. സസ്യനിബിഡത കുറയ്ക്കുക.',
        }

        translated_class = translations.get(predicted_class, predicted_class)

        # Weather-integrated recommendation
        weather_rec = None
        try:
            temp_c   = float(request.form.get('temp_c', 0)) or None
            humidity = float(request.form.get('humidity', 0)) or None
            is_rain  = request.form.get('is_rain', 'false').lower() == 'true'
            if temp_c is not None or is_rain:
                weather_rec = get_weather_integrated_recommendation(
                    predicted_class, temp_c, humidity, is_rain)
        except Exception:
            pass

        recommendation = weather_rec or recommendations.get(predicted_class, 'ഒരു കൃഷി വിദഗ്ദ്ധനെ സമീപിക്കുക.')
        severity_pct, severity_label = analyze_severity(filepath, predicted_class)

        # Save to Supabase
        try:
            pred_user    = supabase.table('users').select('id').eq('username', session['username']).execute()
            pred_user_id = pred_user.data[0]['id'] if pred_user.data else None
            
            if pred_user_id is None:
                print(f"Warning: User '{session['username']}' not found in 'users' table. Proceeding with user_id=None.")

            supabase.table('predictions').insert({
                'user_id':        pred_user_id,
                'username':       session['username'],
                'disease_class':  predicted_class,
                'disease_ml':     translated_class,
                'confidence':     confidence,
                'severity':       severity_pct,
                'severity_label': severity_label,
                'image_url':      f'/static/uploads/{filename}',
                'timestamp':      datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }).execute()
        except Exception as e:
            print(f"Database insertion error: {e}")
            import traceback; traceback.print_exc()

        return jsonify({
            'class':          translated_class,
            'confidence':     confidence,
            'image_url':      f'/static/uploads/{filename}',
            'recommendation': recommendation,
            'severity':       severity_pct,
            'severity_label': severity_label,
            # bonus: return all 6 class probabilities for a bar chart in frontend
            'all_predictions': [
                {'label': label, 'prob': round(p * 100, 1)}
                for label, p in results
            ]
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Error processing image: {str(e)}'})


# --- DASHBOARD API ---
@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        username = session['username']
        result   = supabase.table('predictions').select('*').eq('username', username).order('id', desc=True).execute()
        preds    = result.data if result.data else []

        total_scans    = len(preds)
        healthy_count  = sum(1 for p in preds if p['disease_class'] == 'Healthy Rice Leaf')
        diseased_count = total_scans - healthy_count

        disease_breakdown = {}
        for p in preds:
            d = p['disease_class']
            disease_breakdown[d] = disease_breakdown.get(d, 0) + 1

        timeline = {}
        for p in preds:
            date_str = p['timestamp'][:10] if p.get('timestamp') else 'Unknown'
            if date_str not in timeline:
                timeline[date_str] = {'total': 0, 'healthy': 0, 'diseased': 0, 'severities': []}
            timeline[date_str]['total'] += 1
            if p['disease_class'] == 'Healthy Rice Leaf':
                timeline[date_str]['healthy'] += 1
            else:
                timeline[date_str]['diseased'] += 1
                timeline[date_str]['severities'].append(p.get('severity', 0) or 0)

        sorted_timeline = sorted(timeline.items())
        health_score    = round((healthy_count / total_scans * 100), 1) if total_scans > 0 else 0

        diseased_preds  = [p for p in preds if p['disease_class'] != 'Healthy Rice Leaf']
        avg_severity    = 0
        severity_trend  = 'stable'
        severity_trend_pct = 0

        if diseased_preds:
            all_sevs    = [p.get('severity', 0) or 0 for p in diseased_preds]
            avg_severity = round(sum(all_sevs) / len(all_sevs), 1)
            if len(diseased_preds) >= 4:
                mid         = len(diseased_preds) // 2
                recent_avg  = sum(all_sevs[:mid]) / mid
                older_avg   = sum(all_sevs[mid:]) / (len(diseased_preds) - mid)
                severity_trend_pct = round(recent_avg - older_avg, 1)
                severity_trend = ('improving' if severity_trend_pct < -3
                                else 'worsening' if severity_trend_pct > 3 else 'stable')

        disease_severity = {}
        for p in diseased_preds:
            d = p['disease_class']
            disease_severity.setdefault(d, []).append(p.get('severity', 0) or 0)
        disease_avg_severity = {k: round(sum(v)/len(v), 1) for k, v in disease_severity.items()}

        recent = [{
            'disease_class':  p['disease_class'],
            'disease_ml':     p.get('disease_ml', p['disease_class']),
            'confidence':     p['confidence'],
            'severity':       p.get('severity', 0) or 0,
            'severity_label': p.get('severity_label', ''),
            'image_url':      p.get('image_url'),
            'timestamp':      p['timestamp']
        } for p in preds[:10]]

        most_common = max(disease_breakdown, key=disease_breakdown.get) if disease_breakdown else None

        return jsonify({
            'total_scans':         total_scans,
            'healthy_count':       healthy_count,
            'diseased_count':      diseased_count,
            'health_score':        health_score,
            'avg_severity':        avg_severity,
            'severity_trend':      severity_trend,
            'severity_trend_pct':  severity_trend_pct,
            'disease_breakdown':   disease_breakdown,
            'disease_avg_severity':disease_avg_severity,
            'timeline': [{'date': d, 'total': v['total'], 'healthy': v['healthy'],
                        'diseased': v['diseased'],
                        'avg_severity': round(sum(v['severities'])/len(v['severities']),1)
                        if v['severities'] else 0}
                        for d, v in sorted_timeline],
            'recent':      recent,
            'most_common': most_common
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Database connection error: {str(e)}'}), 503


# --- FORUM / CHAT ---
FORUM_UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads', 'forum')
os.makedirs(FORUM_UPLOAD_FOLDER, exist_ok=True)

def load_posts(current_username=None):
    try:
        posts_result = supabase.table('messages').select('*').is_('parent_id','null').order('id', desc=True).limit(50).execute()
        posts = posts_result.data if posts_result.data else []
        post_ids = [p['id'] for p in posts]
        replies, likes = [], []
        if post_ids:
            replies = (supabase.table('messages').select('*').in_('parent_id', post_ids).order('id').execute().data or [])
            likes   = (supabase.table('likes').select('*').in_('message_id', post_ids).execute().data or [])
        replies_map, likes_count_map, liked_by_me_map = {}, {}, {}
        for r in replies:
            replies_map.setdefault(r['parent_id'], []).append(
                {'id': r['id'], 'username': r['username'], 'text': r['text'], 'timestamp': r['timestamp']})
        for lk in likes:
            mid = lk['message_id']
            likes_count_map[mid] = likes_count_map.get(mid, 0) + 1
            if current_username and lk['username'] == current_username:
                liked_by_me_map[mid] = True
        return [{
            'id': p['id'], 'username': p['username'], 'text': p['text'],
            'image_url': p.get('image_url'), 'timestamp': p['timestamp'],
            'replies': replies_map.get(p['id'], []),
            'likes_count': likes_count_map.get(p['id'], 0),
            'liked_by_me': liked_by_me_map.get(p['id'], False)
        } for p in posts]
    except Exception as e:
        print(f'load_posts DB error: {e}')
        return []

def save_post(username, text, image_url=None, parent_id=None):
    try:
        user_result = supabase.table('users').select('id').eq('username', username).execute()
        user_id = user_result.data[0]['id'] if user_result.data else 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {'user_id': user_id, 'username': username, 'text': text, 'timestamp': now}
        if image_url:  row['image_url']  = image_url
        if parent_id:  row['parent_id']  = parent_id
        supabase.table('messages').insert(row).execute()
        return row
    except Exception as e:
        print(f'save_post DB error: {e}')
        return {'error': str(e)}

@app.route('/chat')
def chat():
    if 'username' not in session: return redirect(url_for('login'))
    return render_template('chat.html')

@app.route('/contact')
def contact():
    if 'username' not in session: return redirect(url_for('login'))
    return render_template('contact.html')

@app.route('/api/posts', methods=['GET'])
def get_posts():
    if 'username' not in session: return jsonify({'error': 'Unauthorized'}), 401
    try:
        return jsonify(load_posts(current_username=session['username']))
    except Exception as e:
        print(f'get_posts error: {e}')
        return jsonify({'error': 'Database connection error'}), 503

@app.route('/api/posts', methods=['POST'])
def create_post():
    if 'username' not in session: return jsonify({'error': 'Unauthorized'}), 401
    try:
        text = request.form.get('text', '').strip()
        file = request.files.get('image')
        image_url = None
        if file and file.filename and allowed_file(file.filename):
            filename  = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
            filepath  = os.path.join(FORUM_UPLOAD_FOLDER, filename)
            file.save(filepath)
            image_url = f'/static/uploads/forum/{filename}'
        if not text and not image_url:
            return jsonify({'error': 'Post cannot be empty'}), 400
        post = save_post(session['username'], text, image_url=image_url)
        return jsonify(post)
    except Exception as e:
        print(f'create_post error: {e}')
        return jsonify({'error': 'Database connection error'}), 503

@app.route('/api/posts/<int:post_id>/reply', methods=['POST'])
def reply_to_post(post_id):
    if 'username' not in session: return jsonify({'error': 'Unauthorized'}), 401
    try:
        data = request.json
        if not data or not data.get('text', '').strip():
            return jsonify({'error': 'Reply cannot be empty'}), 400
        reply = save_post(session['username'], data['text'].strip(), parent_id=post_id)
        return jsonify(reply)
    except Exception as e:
        print(f'reply_to_post error: {e}')
        return jsonify({'error': 'Database connection error'}), 503

@app.route('/api/posts/<int:post_id>/like', methods=['POST'])
def toggle_like(post_id):
    if 'username' not in session: return jsonify({'error': 'Unauthorized'}), 401
    try:
        username    = session['username']
        user_result = supabase.table('users').select('id').eq('username', username).execute()
        user_id     = user_result.data[0]['id'] if user_result.data else 0
        existing    = supabase.table('likes').select('id').eq('message_id', post_id).eq('user_id', user_id).execute()
        if existing.data:
            supabase.table('likes').delete().eq('message_id', post_id).eq('user_id', user_id).execute()
            return jsonify({'liked': False})
        supabase.table('likes').insert({'message_id': post_id, 'user_id': user_id, 'username': username}).execute()
        return jsonify({'liked': True})
    except Exception as e:
        print(f'toggle_like error: {e}')
        return jsonify({'error': 'Database connection error'}), 503


if __name__ == '__main__':
    app.run(debug=True)