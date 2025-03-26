from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from uuid import uuid4
from service import BankDataService
import os
from utils import send_email


# FastAPI app initialization
app = FastAPI(title="Bank Anomaly Detection API")

# Instantiate the service
service = BankDataService(config_file=os.path.join(os.path.dirname(__file__), 'config.properties'))


@app.post("/process-bank-data/", response_class=JSONResponse)
def process_bank_data_api():
    """Process bank data using file paths from config and save results to output_dir."""
    try:
        request_id = str(uuid4())
        result = service.process_bank_data(request_id)
        return JSONResponse(content=result)
    except Exception as e:
        service.logger.error("Processing failed: %s", str(e))
        # Send email on API-level failure
        subject = "Bank Anomaly Detection API: Failure"
        body = f"API request failed with request ID {request_id}.\nError: {str(e)}"
        send_email(service.logger, service.smtp_config, subject, body)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)