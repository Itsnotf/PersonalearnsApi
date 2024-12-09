from flask import Flask, request, jsonify
from jcopml.utils import load_model
import os
from flask_cors import CORS
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv

# Flask Configuration
app = Flask(__name__)
CORS(app)

# Load KNN Model
model_path = os.path.join("model", "knn_model.pkl")
knn = load_model(model_path)


# Google Gemini API Configuration
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
gemini_model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# Mapping hasil prediksi ke metode belajar
metode_belajar_mapping = {
    0: "Feynman",
    1: "Active Recall",
    2: "Circuit Learning",
    3: "Pomodoro",
    4: "SQ3R",
    5: "Mind Mapping",
    6: "Note Taking",
}

# Mapping input numerik ke deskripsi parameter
gaya_belajar_mapping = {0: "Auditory", 1: "Kinestetik", 2: "Visual"}
suasana_mapping = {0: "Individual", 1: "Sosial"}
durasi_mapping = {0: "Panjang (Lebih dari 60 menit)", 1: "Sedang (40 - 50 Menit)", 2: "Singkat (15 - 25 Menit)"}
interval_mapping = {0: "Ya", 1: "Tidak"}
tujuan_mapping = {
    0: "Jangka Panjang (Pemahaman Mendalam)",
    1: "Jangka Pendek (Ujian)",
    2: "Kebutuhan Khusus (Interview)",
    3: "Pengembangan Pribadi (Memperdalam Pengetahuan Keterampilan)"
}
kesulitan_mapping = {0: "Mudah", 1: "Sulit", 2: "Sedang"}
pemahaman_mapping = {0: "Sedang", 1: "Tinggi", 2: "Rendah"}


@app.route('/')
def index():
    return '<h1>Learning Technique Prediction API</h1>'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input JSON dari user
        input_data = request.get_json()

        # Pastikan semua kolom input tersedia
        required_columns = [
            "nama", "email", "GayaBelajar", "Suasana", "Durasi",
            "Interval", "Tujuan", "Kesulitan", "Pemahaman"
        ]
        if not all(col in input_data for col in required_columns):
            return jsonify({"error": "Missing required fields"}), 400

        # Konversi input ke DataFrame
        df = pd.DataFrame([input_data])

        # Prediksi menggunakan model KNN
        prediction = knn.predict(df.drop(columns=["nama", "email"]))
        metode_belajar = prediction[0]  # Hasil prediksi numerik
        metode_belajar_text = metode_belajar_mapping[metode_belajar]

        # Konversi parameter input numerik ke deskripsi
        gaya_belajar = gaya_belajar_mapping[int(input_data["GayaBelajar"])]
        suasana = suasana_mapping[int(input_data["Suasana"])]
        durasi = durasi_mapping[int(input_data["Durasi"])]
        interval = interval_mapping[int(input_data["Interval"])]
        tujuan = tujuan_mapping[int(input_data["Tujuan"])]
        kesulitan = kesulitan_mapping[int(input_data["Kesulitan"])]
        pemahaman = pemahaman_mapping[int(input_data["Pemahaman"])]

        # Bangun prompt untuk Google Gemini
        user_prompt = f"""
        Konteks: Saya memiliki metode belajar yang sudah ditentukan oleh model machine learning (KNN). 
        Tugas Anda adalah memberikan penjelasan yang detail dan jelas tentang metode tersebut serta alasannya 
        mengapa metode ini cocok untuk user berdasarkan parameter yang diberikan.

        Parameter User:
        - Nama: {input_data['nama']}
        - Email: {input_data['email']}
        - Gaya Belajar: {gaya_belajar}
        - Suasana: {suasana}
        - Durasi Belajar: {durasi}
        - Interval Belajar: {interval}
        - Tujuan Belajar: {tujuan}
        - Kesulitan yang Dihadapi: {kesulitan}
        - Tingkat Pemahaman: {pemahaman}

        Metode Belajar yang Direkomendasikan:
        "{metode_belajar_text}"

        Tugas Anda:
        1. Jelaskan metode belajar "{metode_belajar_text}" dengan bahasa yang mudah dipahami oleh user.
        2. Berikan alasan mengapa metode belajar ini cocok untuk user berdasarkan parameter yang telah diberikan.
        3. Berikan langkah-langkah implementasi metode belajar ini agar user bisa menerapkannya dalam kegiatan belajar mereka.
        """

        # Dapatkan penjelasan dari Google Gemini
        response = gemini_model.generate_content(user_prompt)

        # Kembalikan hasil ke user
        return jsonify({
            "MetodeBelajar": metode_belajar_text,
            "Alasan": response.text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
