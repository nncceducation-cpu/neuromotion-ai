import azure.functions as func
from api import app as fastapi_app

# Wrap the existing FastAPI app as an Azure Functions app
app = func.AsgiFunctionApp(app=fastapi_app, http_auth_level=func.AuthLevel.ANONYMOUS)
