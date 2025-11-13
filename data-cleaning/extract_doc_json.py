import json
import os
from apiclient import discovery
from httplib2 import Http
from oauth2client import client, file, tools

# Set doc ID as "YOUR_DOC_ID", as found at `https://docs.google.com/document/d/YOUR_DOC_ID/edit`
DOCUMENT_ID = "1sWQwqcSH23B30CKCVOaQ2kb4D4qES6YfPqmgJYR5rnY"

# Set the scopes and discovery info
SCOPES = "https://www.googleapis.com/auth/documents.readonly"
DISCOVERY_DOC = "https://docs.googleapis.com/$discovery/rest?version=v1"

# Initialize credentials and instantiate Docs API service
script_dir = os.path.dirname(os.path.abspath(__file__))
token_path = os.path.join(script_dir, "token.json")
store = file.Storage(token_path)
creds = store.get()
if not creds or creds.invalid:
  credentials_path = os.path.join(script_dir, "credentials.json")
  flow = client.flow_from_clientsecrets(credentials_path, SCOPES)
  creds = tools.run_flow(flow, store)
service = discovery.build(
    "docs",
    "v1",
    http=creds.authorize(Http()),
    discoveryServiceUrl=DISCOVERY_DOC,
)

# Do a document "get" request and print the results as formatted JSON
result = service.documents().get(documentId=DOCUMENT_ID, includeTabsContent=True).execute()

# Save JSON output to a file
save_dir = os.path.join(script_dir, "saved_data")
os.makedirs(save_dir, exist_ok=True)
data_path = os.path.join(save_dir, "raw_text_breast.json")
with open(data_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
print("Output saved to the json file.")