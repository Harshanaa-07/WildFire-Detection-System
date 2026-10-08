import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import os
import base64
import io
import math
import wave
import struct
import streamlit.components.v1 as components

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Wildfire Detection Hub",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# GENERATE SIREN AUDIO
# ============================================================

@st.cache_data
def generate_siren():

    sample_rate = 44100
    duration = 3.0

    audio_data = []

    for i in range(int(sample_rate * duration)):

        t = i / sample_rate

        # Alternating siren frequencies
        cycle = t % 1.0

        if cycle < 0.5:
            freq = 700 + (1100 - 700) * (cycle / 0.5)
        else:
            freq = 1100 - (1100 - 700) * ((cycle - 0.5) / 0.5)

        # Main siren
        value = math.sin(2 * math.pi * freq * t)

        # Harmonics
        value += 0.35 * math.sin(
            2 * math.pi * freq * 2 * t
        )

        value += 0.15 * math.sin(
            2 * math.pi * freq * 3 * t
        )

        value *= 0.65

        # Convert to 16-bit PCM
        sample = int(
            max(-1, min(1, value)) * 32767
        )

        audio_data.append(sample)

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for sample in audio_data:
            wav.writeframes(
                struct.pack("<h", sample)
            )

    return base64.b64encode(
        buffer.getvalue()
    ).decode()


SIREN_BASE64 = generate_siren()


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_wildfire_model(weights_path):

    model = models.resnet50(
        weights=None
    )

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        1
    )

    # Check weights
    if not os.path.exists(weights_path):

        st.error(
            f"Model weights not found: {weights_path}"
        )

        st.stop()

    # Load trained weights
    state_dict = torch.load(
        weights_path,
        map_location=torch.device("cpu")
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    transform = transforms.Compose([

        transforms.Resize(
            (128, 128)
        ),

        transforms.ToTensor(),

    ])

    if image.mode != "RGB":

        image = image.convert(
            "RGB"
        )

    return transform(
        image
    ).unsqueeze(0)


# ============================================================
# LOAD MODEL
# ============================================================

weights_file = os.path.join(
    "models",
    "resnet50_model.pth"
)

model = load_wildfire_model(
    weights_file
)


# ============================================================
# SESSION STATE
# ============================================================

if "alarm_active" not in st.session_state:

    st.session_state.alarm_active = False


if "prediction_status" not in st.session_state:

    st.session_state.prediction_status = None


if "raw_score" not in st.session_state:

    st.session_state.raw_score = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

if "total_result" not in st.session_state:
    st.session_state.total_result = None

if "fire_count" not in st.session_state:
    st.session_state.fire_count = 0

if "no_fire_count" not in st.session_state:
    st.session_state.no_fire_count = 0


# ============================================================
# NORMAL WHITE UI
# ============================================================

if (
    not st.session_state.alarm_active
    and st.session_state.prediction_status != "No Fire"
):
    st.html(
        """
        <style>

        /* =====================================================
           NORMAL WHITE BACKGROUND
           ===================================================== */

        .stApp {

            background: #ffffff !important;

            background-image: none !important;

        }


        [data-testid="stAppViewContainer"] {

            background: #ffffff !important;

        }


        [data-testid="stHeader"] {

            background: #ffffff !important;

        }


        /* =====================================================
           NORMAL TITLE
           ===================================================== */

        h1 {

            color: #ff7a00 !important;

            font-weight: 900 !important;

            text-shadow:
                0 0 8px
                rgba(255,120,0,0.25);

        }


        h2,
        h3 {

            color: #ff7a00 !important;

        }


        p,
        label {

            color: #333333 !important;

        }


        /* =====================================================
           DESCRIPTION BOX
           ===================================================== */

        .normal-info-box {

            background: #fff7ed;

            border-left:
                5px solid #ff8c00;

            padding: 15px;

            border-radius: 10px;

            color: #333333;

            margin-bottom: 20px;

        }


        /* =====================================================
           UPLOAD BOX
           ===================================================== */

        [data-testid="stFileUploader"] {

            background:
                #ffffff !important;

            border:
                2px solid #ff8c00 !important;

            border-radius:
                15px !important;

            padding:
                10px !important;

            box-shadow:
                0 2px 12px
                rgba(255,120,0,0.15);

        }


        [data-testid="stFileUploader"] * {

            color:
                #333333 !important;

        }


        /* =====================================================
           NORMAL BUTTON
           ===================================================== */

        .stButton > button {

            width: 100%;

            border-radius: 10px;

            border:
                2px solid #ff8c00;

            background:
                linear-gradient(
                    135deg,
                    #ff8c00,
                    #ff5a00
                );

            color:
                #ffffff !important;

            font-weight:
                900;

            font-size:
                17px;

            padding:
                12px;

            box-shadow:
                0 0 10px
                rgba(255,80,0,0.25);

            transition:
                0.2s;

        }


        .stButton > button:hover {

            transform:
                scale(1.02);

            box-shadow:
                0 0 20px
                rgba(255,80,0,0.5);

        }


        /* =====================================================
           NORMAL ALERTS
           ===================================================== */

        [data-testid="stAlert"] {

            border-radius:
                10px;

        }


        /* =====================================================
           METRICS
           ===================================================== */

        [data-testid="stMetricLabel"] {

            color:
                #555555 !important;

        }


        [data-testid="stMetricValue"] {

            color:
                #222222 !important;

        }

        </style>
        """
    )
# ============================================================
# SAFE ENVIRONMENT MODE
# SHOWN ONLY AFTER NO FIRE IS DETECTED
# ============================================================

if st.session_state.prediction_status == "No Fire":

    st.html(
        """
        <style>

        /* =====================================================
           SAFE GREEN BACKGROUND
           ===================================================== */

        .stApp {
            background:
                linear-gradient(
                    180deg,
                    #dff7df 0%,
                    #bce8b8 45%,
                    #8dcc82 72%,
                    #5f9f55 100%
                ) !important;
        }

        [data-testid="stAppViewContainer"] {
            background: transparent !important;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }


        /* =====================================================
           FOREST SCENE
           ===================================================== */

        .safe-forest-scene {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100vw;
            height: 38vh;
            min-height: 250px;

            overflow: hidden;

            pointer-events: none;

            z-index: 0;
        }


        /* =====================================================
           SUN
           ===================================================== */

        .safe-sun {
            position: absolute;

            top: 8%;
            right: 12%;

            width: 75px;
            height: 75px;

            border-radius: 50%;

            background:
                radial-gradient(
                    circle,
                    #fff7a8 0%,
                    #ffe66d 45%,
                    #ffd43b 75%
                );

            box-shadow:
                0 0 25px rgba(255, 221, 75, 0.6),
                0 0 50px rgba(255, 221, 75, 0.3);

            animation:
                sunGlow 3s ease-in-out infinite alternate;
        }


        @keyframes sunGlow {

            from {
                transform: scale(0.96);
                opacity: 0.88;
            }

            to {
                transform: scale(1.05);
                opacity: 1;
            }

        }


        /* =====================================================
           CLOUDS
           ===================================================== */

        .safe-cloud {
            position: absolute;

            width: 130px;
            height: 38px;

            background: rgba(255,255,255,0.82);

            border-radius: 40px;

            filter: blur(0.2px);

            animation:
                cloudMove 24s linear infinite;
        }

        .safe-cloud::before,
        .safe-cloud::after {

            content: "";

            position: absolute;

            background: rgba(255,255,255,0.82);

            border-radius: 50%;

        }

        .safe-cloud::before {

            width: 55px;
            height: 55px;

            left: 20px;
            top: -25px;

        }

        .safe-cloud::after {

            width: 70px;
            height: 70px;

            right: 15px;
            top: -35px;

        }

        .cloud-one {

            top: 15%;
            left: -160px;

            animation-duration: 28s;

        }

        .cloud-two {

            top: 35%;
            left: -220px;

            transform: scale(0.7);

            animation-duration: 36s;

            animation-delay: 4s;

        }


        @keyframes cloudMove {

            from {
                transform:
                    translateX(0);
            }

            to {
                transform:
                    translateX(120vw);
            }

        }


        /* =====================================================
           DISTANT HILLS
           ===================================================== */

        .safe-hill {

            position: absolute;

            bottom: 20%;

            border-radius:
                50% 50% 0 0;

            background:
                linear-gradient(
                    180deg,
                    #76b96c,
                    #4f9650
                );

        }

        .hill-one {

            left: -10%;
            width: 65%;
            height: 42%;

        }

        .hill-two {

            right: -15%;
            width: 70%;
            height: 48%;

        }


        /* =====================================================
           FOREST TREES
           ===================================================== */

        .safe-tree {

            position: absolute;

            bottom: 14%;

            width: 55px;
            height: 145px;

            transform-origin:
                bottom center;

            animation:
                treeSway 4s ease-in-out infinite alternate;

        }


        .safe-tree::before {

            content: "";

            position: absolute;

            left: 23px;
            bottom: 0;

            width: 10px;
            height: 72px;

            border-radius: 5px;

            background:
                linear-gradient(
                    90deg,
                    #704214,
                    #8b5a2b,
                    #603813
                );

        }


        .safe-tree::after {

            content: "";

            position: absolute;

            left: 0;
            top: 0;

            width: 55px;
            height: 90px;

            border-radius:
                50% 50% 45% 45%;

            background:
                radial-gradient(
                    circle at 35% 35%,
                    #75c85c,
                    #3d8f43 55%,
                    #286b35 100%
                );

            box-shadow:
                25px 15px 0 #4c9f48,
                -20px 25px 0 #58aa4d;

        }


        @keyframes treeSway {

            from {
                transform:
                    rotate(-1deg);
            }

            to {
                transform:
                    rotate(1.5deg);
            }

        }


        /* Different tree positions */

        .tree-one {

            left: 5%;
            height: 170px;

            animation-duration: 4.2s;

        }

        .tree-two {

            left: 18%;
            height: 125px;

            transform: scale(0.85);

            animation-duration: 3.5s;

        }

        .tree-three {

            left: 32%;
            height: 155px;

            animation-duration: 4.7s;

        }

        .tree-four {

            right: 25%;
            height: 130px;

            transform: scale(0.85);

            animation-duration: 3.8s;

        }

        .tree-five {

            right: 12%;
            height: 175px;

            animation-duration: 4.5s;

        }

        .tree-six {

            right: 2%;
            height: 145px;

            transform: scale(0.9);

            animation-duration: 3.9s;

        }


        /* =====================================================
           GRASS
           ===================================================== */

        .safe-ground {

            position: absolute;

            left: 0;
            bottom: 0;

            width: 100%;

            height: 23%;

            background:
                linear-gradient(
                    180deg,
                    #579b4c,
                    #397d3b
                );

            border-radius:
                50% 50% 0 0;

        }


        /* =====================================================
           FLOATING LEAVES
           ===================================================== */

        .safe-leaf {

            position: absolute;

            width: 9px;
            height: 5px;

            border-radius:
                100% 0 100% 0;

            background: #4f9d45;

            animation:
                leafFloat var(--leaf-speed)
                ease-in-out infinite;

        }


        @keyframes leafFloat {

            0% {

                transform:
                    translate(
                        0,
                        0
                    )
                    rotate(0deg);

                opacity: 0;

            }

            25% {

                opacity: 1;

            }

            75% {

                opacity: 0.8;

            }

            100% {

                transform:
                    translate(
                        var(--leaf-x),
                        -120px
                    )
                    rotate(180deg);

                opacity: 0;

            }

        }


        /* =====================================================
           SAFE STATUS MESSAGE
           ===================================================== */

        .safe-status-card {

            position: relative;

            z-index: 10;

            width: min(
                650px,
                90%
            );

            margin: 10px auto 25px auto;

            padding: 18px 25px;

            text-align: center;

            background:
                rgba(
                    255,
                    255,
                    255,
                    0.88
                );

            border:
                2px solid
                rgba(
                    76,
                    160,
                    75,
                    0.5
                );

            border-radius: 16px;

            box-shadow:
                0 8px 25px
                rgba(
                    40,
                    100,
                    40,
                    0.15
                );

        }

        .safe-status-title {

            font-size: 25px;

            font-weight: 900;

            color: #287a32;

            margin-bottom: 5px;

        }

        .safe-status-subtitle {

            font-size: 15px;

            color: #4b684d;

        }

        </style>


        <!-- ===================================================
             ANIMATED FOREST
             =================================================== -->

        <div class="safe-forest-scene">

            <div class="safe-sun"></div>

            <div class="safe-cloud cloud-one"></div>
            <div class="safe-cloud cloud-two"></div>

            <div class="safe-hill hill-one"></div>
            <div class="safe-hill hill-two"></div>

            <div class="safe-tree tree-one"></div>
            <div class="safe-tree tree-two"></div>
            <div class="safe-tree tree-three"></div>
            <div class="safe-tree tree-four"></div>
            <div class="safe-tree tree-five"></div>
            <div class="safe-tree tree-six"></div>

            <div class="safe-ground"></div>

            <div
                class="safe-leaf"
                style="
                    left:20%;
                    bottom:20%;
                    --leaf-speed:5s;
                    --leaf-x:80px;
                "
            ></div>

            <div
                class="safe-leaf"
                style="
                    left:42%;
                    bottom:18%;
                    --leaf-speed:6s;
                    --leaf-x:-70px;
                "
            ></div>

            <div
                class="safe-leaf"
                style="
                    left:65%;
                    bottom:25%;
                    --leaf-speed:5.5s;
                    --leaf-x:90px;
                "
            ></div>

            <div
                class="safe-leaf"
                style="
                    left:80%;
                    bottom:22%;
                    --leaf-speed:7s;
                    --leaf-x:-80px;
                "
            ></div>

        </div>


        <!-- ===================================================
             SAFE MESSAGE
             =================================================== -->

        <div class="safe-status-card">

            <div class="safe-status-title">
                🌿 ENVIRONMENT SAFE
            </div>

            <div class="safe-status-subtitle">
                No wildfire activity detected. Monitoring environment is normal.
            </div>

        </div>
        """
    )

# ============================================================
# FIRE EMERGENCY MODE
# ============================================================

if st.session_state.alarm_active:

    st.html(
        """
        <style>

        /* =====================================================
           EMERGENCY BACKGROUND
           ===================================================== */

        .stApp {

            background:
                radial-gradient(
                    ellipse at 50% 100%,
                    #ff5a00 0%,
                    #a51a00 25%,
                    #420000 58%,
                    #080000 100%
                ) !important;

            overflow:
                hidden !important;

        }


        [data-testid="stAppViewContainer"] {

            background:
                transparent !important;

        }


        /* =====================================================
           HIDE NORMAL STREAMLIT HEADER
           ===================================================== */

        [data-testid="stHeader"] {

            visibility:
                hidden !important;

            height:
                0 !important;

        }


        /* =====================================================
           FULL SCREEN FIRE CONTAINER
           ===================================================== */

        .wildfire-screen {

            position:
                fixed;

            inset:
                0;

            width:
                100vw;

            height:
                100vh;

            overflow:
                hidden;

            pointer-events:
                none;

            z-index:
                9990;

            background:
                radial-gradient(
                    ellipse at center bottom,
                    rgba(255,100,0,0.45),
                    rgba(120,0,0,0.20) 50%,
                    rgba(0,0,0,0.10) 100%
                );

        }


        /* =====================================================
           FIRE ATMOSPHERIC GLOW
           ===================================================== */

        .fire-glow {

            position:
                absolute;

            left:
                -10vw;

            bottom:
                -30vh;

            width:
                120vw;

            height:
                100vh;

            border-radius:
                50%;

            background:
                radial-gradient(
                    ellipse at center bottom,

                    rgba(
                        255,
                        255,
                        180,
                        0.98
                    ) 0%,

                    rgba(
                        255,
                        200,
                        20,
                        0.85
                    ) 12%,

                    rgba(
                        255,
                        90,
                        0,
                        0.68
                    ) 32%,

                    rgba(
                        180,
                        20,
                        0,
                        0.40
                    ) 55%,

                    transparent 76%
                );

            filter:
                blur(40px);

            animation:
                fireGlow
                0.9s
                ease-in-out
                infinite
                alternate;

        }


        @keyframes fireGlow {

            0% {

                transform:
                    scale(0.94)
                    translateX(-2%);

                opacity:
                    0.70;

            }

            50% {

                transform:
                    scale(1.02)
                    translateX(2%);

                opacity:
                    0.88;

            }

            100% {

                transform:
                    scale(1.10)
                    translateX(-1%);

                opacity:
                    1;

            }

        }


        /* =====================================================
           FLAME FIELD
           ===================================================== */

        .flame-field {

            position:
                absolute;

            inset:
                0;

            overflow:
                hidden;

        }


        /* =====================================================
           MAIN FLAME
           ===================================================== */
.flame-svg {
    position: absolute;
    left: 0;
    bottom: 0;
    width: 100%;
    height: 100%;
    overflow: visible;
}

.main-flame {
    transform-origin: 50% 100%;
    transform-box: fill-box;
    animation: flameMove 2.8s ease-in-out infinite alternate;
}

.flame-left {
    animation-duration: 2.6s;
}

.flame-left-center {
    animation-duration: 3.2s;
}

.flame-center {
    animation-duration: 2.4s;
}

.flame-right-center {
    animation-duration: 3s;
}

.flame-right {
    animation-duration: 2.7s;
}

.inner-flame {
    transform-origin: 50% 100%;
    transform-box: fill-box;
    animation: innerFlameMove 1.8s ease-in-out infinite alternate;
}

@keyframes flameMove {

    0% {
        transform:
            translateX(-12px)
            scaleX(0.96)
            scaleY(1);
    }

    25% {
        transform:
            translateX(8px)
            scaleX(1.04)
            scaleY(1.03);
    }

    50% {
        transform:
            translateX(-5px)
            scaleX(0.98)
            scaleY(0.96);
    }

    75% {
        transform:
            translateX(14px)
            scaleX(1.03)
            scaleY(1.05);
    }

    100% {
        transform:
            translateX(-10px)
            scaleX(0.95)
            scaleY(1.01);
    }
}

@keyframes innerFlameMove {

    0% {
        transform:
            translateX(-8px)
            scaleX(0.92)
            scaleY(0.96);
    }

    50% {
        transform:
            translateX(10px)
            scaleX(1.06)
            scaleY(1.04);
    }

    100% {
        transform:
            translateX(-6px)
            scaleX(0.94)
            scaleY(1);
    }
}

        /* =====================================================
           FLAME MOVEMENT
           ===================================================== */

        @keyframes flameOne {

            0% {

                transform:
                    translateX(-2vw)
                    scaleX(0.90)
                    scaleY(0.92)
                    skewX(-3deg);

            }

            35% {

                transform:
                    translateX(2vw)
                    scaleX(1.04)
                    scaleY(1.05)
                    skewX(4deg);

            }

            70% {

                transform:
                    translateX(-1vw)
                    scaleX(0.94)
                    scaleY(0.98)
                    skewX(-5deg);

            }

            100% {

                transform:
                    translateX(3vw)
                    scaleX(1.07)
                    scaleY(1.10)
                    skewX(4deg);

            }

        }


        @keyframes flameTwo {

            0% {

                transform:
                    translateX(2vw)
                    scaleX(0.90)
                    scaleY(0.91)
                    skewX(3deg);

            }

            40% {

                transform:
                    translateX(-2vw)
                    scaleX(1.06)
                    scaleY(1.08)
                    skewX(-5deg);

            }

            100% {

                transform:
                    translateX(1vw)
                    scaleX(0.95)
                    scaleY(1.03)
                    skewX(5deg);

            }

        }


        @keyframes flameThree {

            0% {

                transform:
                    translateX(-1vw)
                    scaleX(0.90)
                    scaleY(0.93)
                    skewX(-4deg);

            }

            25% {

                transform:
                    translateX(2vw)
                    scaleX(1.05)
                    scaleY(1.10)
                    skewX(5deg);

            }

            60% {

                transform:
                    translateX(-2vw)
                    scaleX(0.95)
                    scaleY(0.97)
                    skewX(-6deg);

            }

            100% {

                transform:
                    translateX(1vw)
                    scaleX(1.08)
                    scaleY(1.13)
                    skewX(4deg);

            }

        }


        @keyframes flameFour {

            0% {

                transform:
                    translateX(1vw)
                    scaleX(0.93)
                    scaleY(0.92)
                    skewX(4deg);

            }

            45% {

                transform:
                    translateX(-2vw)
                    scaleX(1.06)
                    scaleY(1.08)
                    skewX(-4deg);

            }

            100% {

                transform:
                    translateX(2vw)
                    scaleX(0.97)
                    scaleY(1.04)
                    skewX(5deg);

            }

        }


        @keyframes flameFive {

            0% {

                transform:
                    translateX(-2vw)
                    scaleX(0.90)
                    scaleY(0.94)
                    skewX(-3deg);

            }

            35% {

                transform:
                    translateX(2vw)
                    scaleX(1.05)
                    scaleY(1.10)
                    skewX(5deg);

            }

            100% {

                transform:
                    translateX(-1vw)
                    scaleX(0.96)
                    scaleY(1.03)
                    skewX(-5deg);

            }

        }


        /* =====================================================
           INNER FLAME MOVEMENT
           ===================================================== */

        @keyframes innerDance {

            0% {

                transform:
                    translateX(-4%)
                    scaleX(0.88)
                    scaleY(0.94)
                    skewX(-3deg);

                opacity:
                    0.82;

            }

            50% {

                transform:
                    translateX(4%)
                    scaleX(1.05)
                    scaleY(1.05)
                    skewX(4deg);

                opacity:
                    1;

            }

            100% {

                transform:
                    translateX(-2%)
                    scaleX(0.93)
                    scaleY(1.10)
                    skewX(-5deg);

                opacity:
                    0.92;

            }

        }


        /* =====================================================
           HOT CORE MOVEMENT
           ===================================================== */

        @keyframes coreDance {

            0% {

                transform:
                    translateX(-5%)
                    scaleX(0.82)
                    scaleY(0.88)
                    rotate(-3deg);

                opacity:
                    0.75;

            }

            50% {

                transform:
                    translateX(5%)
                    scaleX(1.06)
                    scaleY(1.08)
                    rotate(4deg);

                opacity:
                    1;

            }

            100% {

                transform:
                    translateX(-2%)
                    scaleX(0.91)
                    scaleY(1.13)
                    rotate(-2deg);

                opacity:
                    0.88;

            }

        }


        /* =====================================================
           EXTRA FLAME TONGUES
           ===================================================== */

        .tongue {

            position:
                absolute;

            bottom:
                -5vh;

            width:
                13vw;

            min-width:
                110px;

            height:
                55vh;

            background:
                linear-gradient(
                    to top,

                    #ff2600 0%,

                    #ff6500 45%,

                    #ffb300 72%,

                    #fff18a 92%,

                    #fffbe0 100%
                );

            clip-path:
                polygon(

                    0% 100%,

                    12% 68%,
                    24% 78%,

                    31% 45%,
                    40% 61%,

                    47% 8%,
                    56% 52%,

                    65% 26%,
                    74% 66%,

                    84% 41%,
                    93% 73%,

                    100% 100%

                );

            filter:
                blur(1px)
                drop-shadow(
                    0 0 18px
                    #ff4500
                );

            opacity:
                0.88;

            transform-origin:
                bottom center;

        }


        .t1 {

            left:
                3%;

            animation:
                tongueDance
                0.67s
                infinite
                alternate;

        }


        .t2 {

            left:
                27%;

            animation:
                tongueDance
                0.91s
                0.18s
                infinite
                alternate-reverse;

        }


        .t3 {

            left:
                56%;

            animation:
                tongueDance
                0.73s
                0.31s
                infinite
                alternate;

        }


        .t4 {

            left:
                80%;

            animation:
                tongueDance
                0.86s
                0.12s
                infinite
                alternate-reverse;

        }


        @keyframes tongueDance {

            from {

                transform:
                    rotate(-8deg)
                    scaleX(0.78)
                    scaleY(0.88);

            }

            to {

                transform:
                    rotate(8deg)
                    scaleX(1.12)
                    scaleY(1.14);

            }

        }


        /* =====================================================
           SPARKS
           ===================================================== */

        .spark {

            position:
                absolute;

            bottom:
                15vh;

            width:
                6px;

            height:
                6px;

            border-radius:
                50%;

            background:
                #fff8b0;

            box-shadow:
                0 0 8px #ffffff,
                0 0 16px #ffd000,
                0 0 28px #ff5a00;

            animation:
                sparkRise
                var(--speed)
                linear
                infinite;

        }


        @keyframes sparkRise {

            0% {

                transform:
                    translate3d(
                        0,
                        0,
                        0
                    )
                    scale(1);

                opacity:
                    0;

            }

            12% {

                opacity:
                    1;

            }

            100% {

                transform:
                    translate3d(
                        var(--drift),
                        -82vh,
                        0
                    )
                    scale(0.1);

                opacity:
                    0;

            }

        }


        /* =====================================================
           EMERGENCY MESSAGE
           ABOVE FIRE
           ===================================================== */

        .emergency-message {

            position:
                fixed;

            top:
                5vh;

            left:
                50%;

            transform:
                translateX(-50%);

            width:
                96vw;

            text-align:
                center;

            z-index:
                10030;

            pointer-events:
                none;

            font-family:
                Arial,
                sans-serif;

            animation:
                warningFlash
                0.72s
                ease-in-out
                infinite
                alternate;

        }


        .emergency-title {

            margin:
                0;

            font-size:
                clamp(
                    38px,
                    6vw,
                    82px
                );

            line-height:
                1;

            font-weight:
                1000;

            letter-spacing:
                4px;

            color:
                #ffffff;

            text-shadow:

                0 0 5px
                #ffffff,

                0 0 15px
                #fff200,

                0 0 32px
                #ff8c00,

                0 0 65px
                #ff2600,

                0 0 110px
                #ff0000;

        }


        .emergency-subtitle {

            margin-top:
                18px;

            font-size:
                clamp(
                    15px,
                    2vw,
                    28px
                );

            font-weight:
                900;

            color:
                #fff9dc;

            text-shadow:

                0 0 8px
                #ffffff,

                0 0 20px
                #ff8c00,

                0 0 40px
                #ff2200;

        }


        @keyframes warningFlash {

            from {

                opacity:
                    0.82;

                transform:
                    translateX(-50%)
                    scale(0.99);

            }

            to {

                opacity:
                    1;

                transform:
                    translateX(-50%)
                    scale(1.015);

            }

        }


        /* =====================================================
   CENTERED STOP ALARM & RESET BUTTON
   ===================================================== */

.st-key-alarm_stop_container {
    position: fixed !important;

    /* CENTER OF SCREEN */
    top: 50% !important;
    left: 50% !important;

    transform: translate(-50%, -50%) !important;

    width: min(520px, 80vw) !important;

    z-index: 10050 !important;

    display: flex !important;
    justify-content: center !important;
    align-items: center !important;

    pointer-events: auto !important;
}


/* BUTTON ITSELF */

.st-key-alarm_stop_container .stButton {
    width: 100% !important;
}


.st-key-alarm_stop_container .stButton > button {

    width: 100% !important;

    height: 78px !important;

    border-radius: 18px !important;

    border: 4px solid #ffcc00 !important;

    background: #ffffff !important;

    color: #b30000 !important;

    font-size: 24px !important;

    font-weight: 1000 !important;

    box-shadow:
        0 0 15px #ffffff,
        0 0 35px #ff3d00,
        0 0 65px #ff9800 !important;

    animation:
        stopPulse
        0.7s
        infinite
        alternate;

    cursor: pointer !important;

    transition: all 0.2s ease !important;
}


/* HOVER EFFECT */

.st-key-alarm_stop_container .stButton > button:hover {

    transform: scale(1.05) !important;

    box-shadow:
        0 0 20px #ffffff,
        0 0 45px #ff3d00,
        0 0 80px #ff9800 !important;
}


/* BUTTON PRESS */

.st-key-alarm_stop_container .stButton > button:active {

    transform: scale(0.98) !important;
}


@keyframes stopPulse {

    from {
        transform: scale(1);
    }

    to {
        transform: scale(1.04);
    }

}

        @keyframes stopPulse {

            from {

                transform:
                    scale(1);

            }

            to {

                transform:
                    scale(1.04);

            }

        }

        </style>
        """
    )


    # ========================================================
    # FIRE HTML
    # ========================================================

    st.html(
        """
        <div class="wildfire-screen">

            <div class="fire-glow"></div>

            <div class="flame-field">

                <!-- FIVE LARGE FIRE FLAMES -->

                <div class="flame f1"></div>

                <div class="flame f2"></div>

                <div class="flame f3"></div>

                <div class="flame f4"></div>

                <div class="flame f5"></div>


                <!-- EXTRA MOVING FLAME TONGUES -->

                <div class="tongue t1"></div>

                <div class="tongue t2"></div>

                <div class="tongue t3"></div>

                <div class="tongue t4"></div>


                <!-- FIRE SPARKS -->

                <div
                    class="spark"
                    style="
                        left:8%;
                        --speed:2.1s;
                        --drift:-30px;
                    ">
                </div>


                <div
                    class="spark"
                    style="
                        left:18%;
                        --speed:1.7s;
                        --drift:45px;
                    ">
                </div>


                <div
                    class="spark"
                    style="
                        left:31%;
                        --speed:2.4s;
                        --drift:-55px;
                    ">
                </div>


                <div
                    class="spark"
                    style="
                        left:43%;
                        --speed:1.8s;
                        --drift:35px;
                    ">
                </div>


                <div
                    class="spark"
                    style="
                        left:56%;
                        --speed:2.2s;
                        --drift:-40px;
                    ">
                </div>


                <div
                    class="spark"
                    style="
                        left:68%;
                        --speed:1.9s;
                        --drift:50px;
                    ">
                </div>


                <div
                    class="spark"
                    style="
                        left:81%;
                        --speed:2.3s;
                        --drift:-45px;
                    ">
                </div>


                <div
                    class="spark"
                    style="
                        left:92%;
                        --speed:1.6s;
                        --drift:30px;
                    ">
                </div>

            </div>

        </div>


        <!-- ==================================================
             EMERGENCY TEXT
             ================================================== -->

        <div class="emergency-message">

            <div class="emergency-title">

                🚨 WILDFIRE DETECTED 🚨

            </div>


            <div class="emergency-subtitle">

                🔥 ACTIVE FLAMES DETECTED
                IN MONITORED ENVIRONMENT 🔥

            </div>


            <div class="emergency-subtitle">

                ⚠️ IMMEDIATE ATTENTION REQUIRED ⚠️

            </div>

        </div>

        """
    )


    # ========================================================
    # SIREN AUDIO
    # ========================================================

# ============================================================
# SIREN AUDIO CONTROLLER
# SIREN PLAYS ONLY WHEN alarm_active = True
# ============================================================

alarm_active_js = "true" if st.session_state.alarm_active else "false"

components.html(
    f"""
    <!DOCTYPE html>
    <html>
    <body style="
        margin:0;
        padding:0;
        overflow:hidden;
        background:transparent;
    ">

        <audio
            id="fireSiren"
            loop
            playsinline
            preload="auto">

            <source
                src="data:audio/wav;base64,{SIREN_BASE64}"
                type="audio/wav">

        </audio>

        <script>

            const siren = document.getElementById("fireSiren");
            const alarmActive = {alarm_active_js};

            function startSiren() {{

                if (!siren || !alarmActive) return;

                siren.volume = 1.0;

                const playPromise = siren.play();

                if (playPromise !== undefined) {{

                    playPromise.catch(function(error) {{

                        console.log(
                            "Autoplay blocked:",
                            error
                        );

                    }});

                }}
            }}


            function stopSiren() {{

                if (!siren) return;

                siren.pause();
                siren.currentTime = 0;
                siren.removeAttribute("src");

                const source = siren.querySelector("source");

                if (source) {{
                    source.removeAttribute("src");
                }}

                siren.load();
            }}


            // ==================================================
            // MAIN CONTROL
            // ==================================================

            if (alarmActive) {{

                startSiren();

                // Browser may block autoplay.
                // User interaction can start the siren.
                document.addEventListener(
                    "click",
                    startSiren,
                    {{ once: true }}
                );

            }} else {{

                // IMPORTANT:
                // Alarm has been reset.
                // Stop sound immediately.
                stopSiren();

            }}


            // ==================================================
            // CLEANUP
            // ==================================================

            window.addEventListener(
                "beforeunload",
                function() {{

                    stopSiren();

                }}
            );


            window.addEventListener(
                "pagehide",
                function() {{

                    stopSiren();

                }}
            );

        </script>

    </body>
    </html>
    """,
    height=1,
    scrolling=False,
)
    # ========================================================
    # STOP ALARM BUTTON
    # ========================================================


# ========================================================
# CENTERED STOP ALARM & RESET BUTTON
# SHOW ONLY WHEN FIRE IS DETECTED
# ========================================================

if st.session_state.alarm_active:

    with st.container(key="alarm_stop_container"):

        if st.button(
            "🛑 STOP ALARM & RESET",
            key="stop_alarm"
        ):

            st.session_state.alarm_active = False

            st.session_state.prediction_status = None

            st.session_state.raw_score = None

            st.session_state.batch_results = []

            st.session_state.total_result = None

            st.session_state.fire_count = 0

            st.session_state.no_fire_count = 0

            st.rerun()

# ============================================================
# MAIN PAGE TITLE
# ============================================================

st.html(
    """
    <style>

    .main-wildfire-title {
        width: 100%;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .main-wildfire-title h1 {
        font-size: clamp(42px, 5vw, 72px) !important;
        font-weight: 1000 !important;
        letter-spacing: 2px !important;
        color: #ff6500 !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.05 !important;

        text-shadow:
            0 0 6px rgba(255, 100, 0, 0.25),
            0 0 18px rgba(255, 80, 0, 0.18);
    }

    .main-wildfire-subtitle {
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        color: #555555;
        margin-top: 10px;
        margin-bottom: 30px;
    }

    </style>

    <div class="main-wildfire-title">
        <h1>🔥 WILDFIRE DETECTION SYSTEM 🔥</h1>
    </div>

    """
)


st.markdown("---")


    # ========================================================
    # MAIN COLUMNS
    # ========================================================

col1, col2 = st.columns(
    [1.15, 0.85],
    gap="large"
)


# ========================================================
# LEFT COLUMN
# ========================================================

with col1:

    st.subheader(
        "📷 Environment Stream Upload"
    )

    uploaded_files = st.file_uploader(
        "Choose environment image snapshots...",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True
    )


    # ============================================================
# MULTI-IMAGE ENVIRONMENT ANALYSIS
# ============================================================
# ============================================================
# MULTI-IMAGE ENVIRONMENT ANALYSIS
# ============================================================

if uploaded_files:

    st.markdown("### 🖼️ Uploaded Environment Frames")

    # --------------------------------------------------------
    # DISPLAY UPLOADED IMAGES
    # --------------------------------------------------------

    preview_cols = st.columns(
        min(len(uploaded_files), 4)
    )

    for index, uploaded_file in enumerate(uploaded_files):

        image = Image.open(uploaded_file)

        with preview_cols[index % len(preview_cols)]:

            st.image(
                image,
                caption=f"Frame {index + 1}: {uploaded_file.name}",
                use_container_width=True
            )

    # --------------------------------------------------------
    # ANALYZE ALL IMAGES
    # --------------------------------------------------------

    if st.button(
        "🚀 ANALYZE ALL FRAMES",
        key="analyze_all_frames"
    ):

        batch_results = []

        fire_count = 0
        no_fire_count = 0

        progress = st.progress(0)

        with st.spinner(
            "Analyzing all environment frames using ResNet50..."
        ):

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                # --------------------------------------------
                # LOAD IMAGE
                # --------------------------------------------

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                # --------------------------------------------
                # PREPROCESS
                # --------------------------------------------

                input_tensor = preprocess_image(
                    image
                )

                # --------------------------------------------
                # RESNET50 PREDICTION
                # --------------------------------------------

                with torch.no_grad():

                    logits = model(
                        input_tensor
                    )

                    probability = torch.sigmoid(
                        logits
                    ).item()

                # --------------------------------------------
                # IMAGE-LEVEL FIRE DETECTION
                #
                # YOUR MODEL LOGIC:
                #
                # probability < 0.5  -> FIRE
                # probability >= 0.5 -> NO FIRE
                # --------------------------------------------
                    # Individual image
                    if probability < 0.5:
                        prediction = "Fire"
                        fire_count += 1
                    else:
                        prediction = "No Fire"
                        no_fire_count += 1


                    # Entire upload batch
                    if fire_count >= 1:
                        total_result = "Fire"
                        st.session_state.alarm_active = True
                        st.session_state.prediction_status = "Fire"
                    else:
                        total_result = "No Fire"
                        st.session_state.alarm_active = False
                        st.session_state.prediction_status = "No Fire"

                # --------------------------------------------
                # STORE INDIVIDUAL RESULT
                # --------------------------------------------

                batch_results.append({

                    "Frame": index + 1,

                    "File": uploaded_file.name,

                    "Prediction": prediction,

                    "Probability": probability

                })

                # --------------------------------------------
                # UPDATE PROGRESS
                # --------------------------------------------

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

        # ====================================================
        # OVERALL FOREST FIRE LOGIC
        #
        # IMPORTANT:
        # EVEN ONE FIRE IMAGE = OVERALL FIRE
        # ====================================================

        total_images = len(uploaded_files)

        if fire_count >= 1:

            total_result = "Fire"

            st.session_state.alarm_active = True

            st.session_state.prediction_status = "Fire"

        else:

            total_result = "No Fire"

            st.session_state.alarm_active = False

            st.session_state.prediction_status = "No Fire"

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        st.session_state.batch_results = batch_results

        st.session_state.fire_count = fire_count

        st.session_state.no_fire_count = no_fire_count

        st.session_state.total_result = total_result

        # ----------------------------------------------------
        # RERUN UI
        # ----------------------------------------------------

        st.rerun()
        # =================================================
        # ANALYZE BUTTON
        # =================================================

        # if st.button(

        #     "🚀 ANALYZE FRAME",

        #     key="analyze_frame"

        # ):

        #     with st.spinner(

        #         "Processing frame via "
        #         "Edge-AI ResNet50 framework..."

        #     ):

        #         input_tensor =preprocess_image(
        #                 image
        #             )


        #         with torch.no_grad():

        #             logits = model(
        #                 input_tensor
        #             )


        #             probability = (
        #                 torch.sigmoid(
        #                     logits
        #                 ).item()
        #             )


        #         # Store probability

        #         st.session_state.raw_score = (
        #             probability
        #         )
        # ============================================================
# BATCH ANALYSIS RESULTS
# ============================================================

    if st.session_state.batch_results:

        st.markdown("---")

        st.markdown(
            "### 📊 Frame-by-Frame Detection Results"
        )

        for result in st.session_state.batch_results:

            if result["Prediction"] == "Fire":

                st.error(
                    f"🔥 Frame {result['Frame']} | "
                    f"{result['File']} | "
                    f"FIRE | "
                    f"Probability: {result['Probability']:.4f}"
                )

            else:

                st.success(
                    f"🌲 Frame {result['Frame']} | "
                    f"{result['File']} | "
                    f"NO FIRE | "
                    f"Probability: {result['Probability']:.4f}"
                )


               

# ============================================================
# OVERALL FOREST RESULT
# ============================================================

if st.session_state.total_result is not None:

    st.markdown("---")

    fire_count = st.session_state.fire_count

    no_fire_count = st.session_state.no_fire_count

    total_images = (
        fire_count + no_fire_count
    )

    # --------------------------------------------------------
    # FIRE DETECTED
    # EVEN ONE FIRE IMAGE IS ENOUGH
    # --------------------------------------------------------

    if st.session_state.total_result == "Fire":

        st.error(
            f"""
            🔥 **WILDFIRE DETECTED**

            **{fire_count} of {total_images}**
            analyzed image(s) indicate FIRE.

            ⚠️ At least one image contains fire.
            Immediate attention required.
            """
        )

    # --------------------------------------------------------
    # NO FIRE DETECTED
    # ONLY WHEN ALL IMAGES ARE NO FIRE
    # --------------------------------------------------------

    else:

        st.success(
            f"""
            🌿 **ENVIRONMENT SAFE**

            **0 of {total_images}**
            analyzed images indicate fire.

            ✅ All analyzed images are classified as NO FIRE.
            """
        )       

# ========================================================
# RIGHT COLUMN
# ========================================================

with col2:

    st.subheader(
        "📊 System Diagnostics"
    )


    # =====================================================
    # MODEL SCORE
    # =====================================================

    if st.session_state.raw_score is not None:

        st.info(

            f"""
            🔬 **Model Confidence Value**

            `{st.session_state.raw_score:.4f}`
            """

        )


    # =====================================================
    # NO FIRE
    # =====================================================

    if (
        st.session_state.prediction_status
        == "No Fire"
    ):

        st.success(

            "🌲 SECURE: Environment Normal. "
            "No fire cues detected."

        )


    # =====================================================
    # FIRE
    # =====================================================

    elif (
        st.session_state.prediction_status
        == "Fire"
    ):

        st.error(
            "🚨 WILDFIRE DETECTED"
        )


    # =====================================================
    # STANDBY
    # =====================================================

    else:

        st.info(

            "💡 Standby: Awaiting incoming "
            "RGB surveillance imagery."

        )


    st.markdown("---")


 