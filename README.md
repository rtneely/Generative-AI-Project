# Generative-AI-Project

This project is a stock research and visualization agent built with Streamlit. Users can ask questions about a stock ticker or company name to retrieve real-time news information, view a historical stock chart, learn about company fundamentals, and compare companies. The app uses the Finnhub API to fetch financial data and was developed as part of the Generative AI course in the MSBA program at Cal Poly.

---

## 🛠 Features

- Real-time stock data retrieval
- Historical chart plotting
- Company comparison charts
- Up-to-date fundamentals information
- Streamlit UI for easy interaction
- Secure API key access via `.env` file (excluded from repo)

---

## 📂 Project Structure

```
Project/
├── app.py             # Main Streamlit app
├── requirements.txt   # Python dependencies
├── .gitignore         # Ensures .env is not tracked
└── README.md          # You're reading it!
```

---

## 💻 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/rtneely/Generative-AI-Project.git
cd Generative-AI-Project
```

### 2. Add Your API Key

Create a `.env` file in the root of the project and add your [Finnhub](https://finnhub.io) API key.
Also add AWS credentials such as your access key ID, secret access key, and region:

```
FINNHUB_API_KEY=your_actual_api_key_here
AWS_ACCESS_KEY_ID=your_actual_key_here
AWS_SECRET_ACCESS_KEY=your_actual_key_here
AWS_DEFAULT_REGION=your_region_here
```

> **Note:** This file is ignored by Git for security reasons.

### 3. Install Dependencies

Use `pip` to install the required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is too large, you can just install manually:

```bash
pip install streamlit requests pandas python-dotenv
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 👨‍🎓 Project Context

This app was developed as an individual submission for the GSB570 course on Generative AI in Business Analytics at Cal Poly. The goal was to build a functional AI-assisted or data-powered tool using real APIs and interface technologies.

Each student is required to host their project in their own GitHub repository.

---

## 🔐 Notes

- The `.env` file is intentionally excluded from this repo for security.
- You must sign up for a free Finnhub API key to run the app.
- Be sure to use `streamlit run app.py` to launch the interface in a browser.

---

## 👨‍💻 Author

Ryan Neely – Cal Poly MSBA  
GitHub: [@rtneely](https://github.com/rtneely)
