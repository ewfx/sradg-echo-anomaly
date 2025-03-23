import pandas as pd
from sklearn.ensemble import IsolationForest
import google.generativeai as genai
import os
import logging
import configparser
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# Load configuration from properties file
config = configparser.ConfigParser()
config_file = os.path.join(os.path.dirname(__file__), 'config.properties')
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Config file not found: {config_file}")
config.read(config_file)

# Extract config values
BANK_FILE = config['Paths']['bank_file']
INTERNAL_FILE = config['Paths']['internal_file']
LOG_DIR = config['Paths']['log_dir']
LOG_FILE = os.path.join(LOG_DIR, config['Paths']['log_file'])
OUTPUT_DIR = config['Paths']['output_dir']
OUTPUT_CSV = os.path.join(OUTPUT_DIR, config['Paths']['output_csv'])
GEMINI_API_KEY = config['Gemini']['api_key']
GEMINI_MODEL = config['Gemini']['model']
FEATURES = config['Detection'].get('features', 'Amount')
FEATURES = [f.strip() for f in FEATURES.split(',')] if FEATURES.strip() else ['Amount']
CONTAMINATION = config['Detection'].get('contamination', '')
if CONTAMINATION.strip().lower() == 'none' or not CONTAMINATION.strip():
    CONTAMINATION = None
else:
    try:
        CONTAMINATION = float(CONTAMINATION)
    except ValueError as e:
        raise ValueError(f"Invalid contamination value in config: {e}")

# Email config
SMTP_SERVER = config['Email']['smtp_server']
SMTP_PORT = int(config['Email']['smtp_port'])
SENDER_EMAIL = config['Email']['sender_email']
SENDER_PASSWORD = config['Email']['sender_password']
RECIPIENT_EMAIL = config['Email']['recipient_email']

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# Configure Gemini API
if not GEMINI_API_KEY:
    logger.error("Gemini API key not provided in config.properties.")
    raise ValueError("Gemini API key not provided in config.properties.")
genai.configure(api_key=GEMINI_API_KEY)

def send_email(subject, body, attachment_path=None):
    """Sends an email with optional attachment."""
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, 'rb') as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
                part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                msg.attach(part)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        logger.info("Email sent successfully: %s", subject)
    except Exception as e:
        logger.error("Failed to send email: %s", e)

def load_data(bank_file, internal_file):
    """Loads bank statement and internal transaction records with error handling."""
    try:
        bank_data = pd.read_csv(bank_file)
        internal_data = pd.read_csv(internal_file)
        required_cols = {'Transaction ID', 'Amount', 'Date'}
        if not (required_cols.issubset(bank_data.columns) and required_cols.issubset(internal_data.columns)):
            logger.error("Missing required columns in input files.")
            raise ValueError("Missing required columns.")
        logger.info("Data loaded successfully from %s and %s", bank_file, internal_file)
        return bank_data, internal_data
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except Exception as e:
        logger.error("Data loading error: %s", e)
        raise

def reconcile_transactions(bank_data, internal_data):
    """Finds mismatched transactions between bank and internal records."""
    try:
        merged = pd.merge(bank_data, internal_data, on=['Transaction ID', 'Amount', 'Date'],
                         how='outer', indicator=True)
        unmatched = merged[merged['_merge'] != 'both']
        logger.info("Reconciliation complete. Found %d unmatched transactions.", len(unmatched))
        logger.debug("Unmatched transactions:\n%s", unmatched.to_string())
        return unmatched
    except Exception as e:
        logger.error("Reconciliation error: %s", e)
        raise

def estimate_contamination(data, feature='Amount'):
    """Estimates contamination based on data variance."""
    std = data[feature].std()
    mean = data[feature].mean()
    outlier_threshold = mean + 3 * std
    contamination = len(data[data[feature] > outlier_threshold]) / len(data)
    estimated = max(min(contamination, 0.1), 0.01)
    logger.debug("Estimated contamination: %.3f", estimated)
    return estimated

def detect_fraud(data, features=FEATURES, contamination=CONTAMINATION):
    """Detects fraudulent transactions using Isolation Forest."""
    try:
        if contamination is None:
            contamination = estimate_contamination(data, features[0])
        if 'Date' in features:
            data['DateNumeric'] = pd.to_datetime(data['Date']).astype(int) / 10**9
            features = ['DateNumeric' if f == 'Date' else f for f in features]
        model = IsolationForest(contamination=contamination, random_state=42)
        data['Anomaly_Score'] = model.fit_predict(data[features])
        anomalies = data[data['Anomaly_Score'] == -1]
        logger.info("Detected %d anomalies using features: %s", len(anomalies), features)
        logger.debug("Anomalies:\n%s", anomalies.to_string())
        return anomalies
    except Exception as e:
        logger.error("Fraud detection error: %s", e)
        raise

def explain_anomalies(anomalies, bank_data):
    """Uses Gemini AI with context to explain anomalies."""
    try:
        explanations = []
        model = genai.GenerativeModel(GEMINI_MODEL)
        mean_amount = bank_data['Amount'].mean()
        std_amount = bank_data['Amount'].std()

        for idx, row in anomalies.iterrows():
            prompt = (
                f"Transaction ID {row['Transaction ID']} with amount {row['Amount']} on {row['Date']} "
                f"is flagged as suspicious. Details: {row.to_dict()}. "
                f"Context: Mean amount is {mean_amount:.2f}, std dev is {std_amount:.2f}. "
                f"Why might this be fraudulent?"
            )
            response = model.generate_content(prompt)
            explanations.append(response.text)
            logger.debug("Explanation for Transaction ID %s: %s", row['Transaction ID'], response.text)
        
        anomalies = anomalies.copy()
        anomalies.loc[:, 'Explanation'] = explanations
        logger.info("Generated explanations for %d anomalies", len(anomalies))
        logger.debug("Explained anomalies:\n%s", anomalies.to_string())
        return anomalies
    except Exception as e:
        logger.error("Gemini API error: %s", e)
        raise

if __name__ == "__main__":
    try:
        bank_data, internal_data = load_data(BANK_FILE, INTERNAL_FILE)
        unmatched_transactions = reconcile_transactions(bank_data, internal_data)
        
        anomalies = detect_fraud(bank_data)
        
        explained_anomalies = explain_anomalies(anomalies, bank_data)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist
        explained_anomalies.to_csv(OUTPUT_CSV, index=False)
        logger.info("Results saved to %s", OUTPUT_CSV)
        
        # Send success email with attachment
        send_email(
            subject="Fraud Detection Results",
            body=f"Anomaly detection completed successfully. Results saved to {OUTPUT_CSV}.",
            attachment_path=OUTPUT_CSV
        )
    except Exception as e:
        logger.error("Program failed: %s", e)
        # Send failure email
        send_email(
            subject="Fraud Detection Failed",
            body=f"Anomaly detection failed with error: {str(e)}"
        )