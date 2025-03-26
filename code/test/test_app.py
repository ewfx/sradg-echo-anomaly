import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from fastapi.testclient import TestClient
import os
import torch
from src.service import BankDataService
from src.app import app  

# Sample data for testing
BANK_DATA = pd.DataFrame({
    'Transaction ID': ['T1', 'T2'],
    'Account Number': ['A1', 'A2'],
    'Balance Difference': [100.0, 200.0],
    'As of Date': ['2025-03-25', '2025-03-25']
})

INTERNAL_DATA = pd.DataFrame({
    'Transaction ID': ['T0'],
    'Account Number': ['A1'],
    'Balance Difference': [50.0],
    'As of Date': ['2025-03-24']
})

class TestBankDataService(unittest.TestCase):
    def setUp(self):
        # Mock config file content
        self.config_content = {
            'Paths': {
                'bank_file': 'mock_bank.csv',
                'internal_file': 'mock_internal.csv',
                'log_dir': 'logs',
                'log_file': 'test.log',
                'output_dir': 'output'
            },
            'Email': {
                'smtp_server': 'smtp.test.com',
                'smtp_port': '587',
                'sender_email': 'test@test.com',
                'sender_password': 'password',
                'recipient_email': 'recipient@test.com'
            }
        }
        # Mock file existence and permissions
        self.patcher_os = patch('os.path.exists', return_value=True)
        self.patcher_access = patch('os.access', return_value=True)
        self.patcher_os.makedirs = patch('os.makedirs')
        self.patcher_os.start()
        self.patcher_access.start()
        self.patcher_os.makedirs.start()

        # Mock logger
        self.logger = MagicMock()

        # Mock DistilBERT model and tokenizer
        self.tokenizer = MagicMock()
        self.model = MagicMock()
        self.device = 'cpu'
        self.model.eval.return_value = None
        self.tokenizer.return_tensors.return_value = {'input_ids': MagicMock(), 'attention_mask': MagicMock()}
        self.model.return_value.logits = torch.tensor([[0.0, 1.0]])  # Mock anomaly score

        # Initialize service with mocked dependencies
        with patch('src.service.load_config', return_value=self.config_content), \
             patch('src.service.setup_logging', return_value=self.logger), \
             patch('src.service.initialize_model', return_value=(self.tokenizer, self.model, self.device)):
            self.service = BankDataService('mock_config.properties')

        # FastAPI test client
        self.client = TestClient(app)

    def tearDown(self):
        self.patcher_os.stop()
        self.patcher_access.stop()
        self.patcher_os.makedirs.stop()

    def test_init_success(self):
        """Test successful initialization of BankDataService."""
        self.assertEqual(self.service.bank_file, 'mock_bank.csv')
        self.assertEqual(self.service.output_dir, 'output')
        self.assertIsNotNone(self.service.logger)
        self.assertIsNotNone(self.service.tokenizer)

    def test_init_file_not_found(self):
        """Test initialization failure when files are missing."""
        with patch('os.path.exists', side_effect=[False, True, True]), \
             self.assertRaises(FileNotFoundError):
            BankDataService('mock_config.properties')

    def test_smart_reconcile(self):
        """Test smart_reconcile method."""
        new_transactions, duplicates = self.service.smart_reconcile(BANK_DATA, INTERNAL_DATA)
        pd.testing.assert_frame_equal(new_transactions, BANK_DATA)
        self.assertTrue(duplicates.empty)
        self.logger.info.assert_called_with("Reconciliation: %d new, %d duplicates", 2, 0)

    @patch('src.service.get_anomaly_score', return_value=0.7)
    def test_detect_trend_anomalies(self, mock_get_anomaly_score):
        """Test anomaly detection logic."""
        anomalies = self.service.detect_trend_anomalies(BANK_DATA, INTERNAL_DATA)
        self.assertFalse(anomalies.empty)
        self.assertIn('Fraud_Score', anomalies.columns)
        self.assertIn('Explanation', anomalies.columns)
        self.assertTrue((anomalies['Fraud_Score'] >= 0.5).all())

    @patch('src.service.get_anomaly_score', return_value=0.7)
    @patch('src.service.train_prediction_model', return_value={'A1': MagicMock(predict=lambda x: [75.0])})
    @patch('src.service.predict_balance', return_value=75.0)
    def test_smart_correct_anomalies(self, mock_predict, mock_train, mock_score):
        """Test anomaly correction logic."""
        anomalies = pd.DataFrame({
            'Transaction ID': ['T1'],
            'Account Number': ['A1'],
            'Balance Difference': [100.0],
            'As of Date': ['2025-03-25'],
            'Fraud_Score': [0.7],
            'Explanation': ['Trend up detected']
        })
        corrected = self.service.smart_correct_anomalies(anomalies, BANK_DATA, INTERNAL_DATA)
        self.assertEqual(corrected.loc[corrected['Transaction ID'] == 'T1', 'Balance Difference'].iloc[0], 75.0)
        self.logger.info.assert_any_call("Corrected Balance Difference for %s: %s -> %s (predicted)", 
                                         'T1', 100.0, 75.0)

    @patch('src.service.load_data', return_value=(BANK_DATA, INTERNAL_DATA))
    @patch('pandas.DataFrame.to_csv')
    @patch('src.service.send_email')
    def test_process_bank_data(self, mock_send_email, mock_to_csv, mock_load_data):
        """Test full processing pipeline."""
        result = self.service.process_bank_data('test-request-id')
        self.assertIn('anomalies_file', result)
        self.assertIn('corrected_file', result)
        self.assertEqual(result['request_id'], 'test-request-id')
        mock_to_csv.assert_called()

    def test_api_endpoint_success(self):
        """Test FastAPI endpoint success case."""
        with patch.object(self.service, 'process_bank_data', return_value={
            'anomalies_file': 'anomalies_test.csv',
            'corrected_file': 'corrected_test.csv',
            'request_id': 'test-request-id'
        }):
            response = self.client.post("/process-bank-data/")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['request_id'], 'test-request-id')

    def test_api_endpoint_failure(self):
        """Test FastAPI endpoint failure case."""
        with patch.object(self.service, 'process_bank_data', side_effect=Exception("Test error")):
            response = self.client.post("/process-bank-data/")
            self.assertEqual(response.status_code, 500)
            self.assertIn("Test error", response.json()['detail'])

if __name__ == '__main__':
    unittest.main()