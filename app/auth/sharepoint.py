import requests
from app.config import settings

class SharePointAuth:
    def __init__(self):
        self.client_id = settings.SHAREPOINT_CLIENT_ID
        self.client_secret = settings.SHAREPOINT_CLIENT_SECRET
        self.tenant_id = settings.SHAREPOINT_TENANT_ID
        self._token = None

    def get_token(self) -> str:
        if not self._token:
            url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
            data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "https://graph.microsoft.com/.default"
            }
            response = requests.post(url, data=data)
            response.raise_for_status()
            self._token = response.json()["access_token"]
        return self._token