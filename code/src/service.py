import pandas as pd
from sklearn.ensemble import IsolationForest
import google.generativeai as genai  



genai.configure(api_key="xxxxxx")


def load_data(bank_file, internal_file):
    """Loads bank statement and internal transaction records."""
    bank_data = pd.read_csv(bank_file)
    internal_data = pd.read_csv(internal_file)
    return bank_data, internal_data


def reconcile_transactions(bank_data, internal_data):
    """Finds mismatched transactions between bank and internal records."""
    merged = pd.merge(bank_data, internal_data, on=['Transaction ID', 'Amount', 'Date'], how='outer', indicator=True)
    unmatched = merged[merged['_merge'] != 'both']
    return unmatched


def detect_fraud(data):
    """Detects fraudulent transactions using Isolation Forest."""
    model = IsolationForest(contamination=0.05, random_state=42)
    data['Anomaly_Score'] = model.fit_predict(data[['Amount']])
    anomalies = data[data['Anomaly_Score'] == -1]
    return anomalies


def explain_anomalies(anomalies):
    """Uses Gemini AI to explain why transactions might be fraudulent."""
    explanations = []
    model = genai.GenerativeModel("gemini-1.5-flash")  # Use Gemini model
   
    for idx, row in anomalies.iterrows():
        prompt = (
            f"Transaction ID {row['Transaction ID']} with amount {row['Amount']} on {row['Date']} looks suspicious. "
            f"Here are some details: {row.to_dict()}. Why might this transaction be fraudulent?"
        )
        response = model.generate_content(prompt)
        print(f"Prompt: {prompt}")  # Print the prompt for debugging
        print(f"Response: {response.text}")  # Print the full response for debugging
        explanations.append(response.text)

    # Ensure assignment is done safely using .loc
    anomalies = anomalies.copy()  # Prevent SettingWithCopyWarning
    anomalies.loc[:, 'Explanation'] = explanations
    return anomalies

if __name__ == "__main__":
    # Load transaction data
    bank_data, internal_data = load_data('bank_statement.csv', 'internal_records.csv')

    # Perform reconciliation
    unmatched_transactions = reconcile_transactions(bank_data, internal_data)
    print("Unmatched Transactions:")
    print(unmatched_transactions)

    # Detect fraud
    anomalies = detect_fraud(bank_data)
    print("Anomalous Transactions:")
    print(anomalies)

    # Generate AI-based explanations
    explained_anomalies = explain_anomalies(anomalies)
    print("Anomalies with Explanations:")
    print("#"*100)
    print(explained_anomalies)