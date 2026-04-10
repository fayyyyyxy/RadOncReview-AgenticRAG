from apiclient import discovery
from httplib2 import Http
from oauth2client import client, file, tools
import os


def extract_json(DOCUMENT_ID):
    """ Extracts the Google Doc content as JSON. """
  
    # Set the scopes and discovery info
    SCOPES = "https://www.googleapis.com/auth/documents.readonly"
    DISCOVERY_DOC = "https://docs.googleapis.com/$discovery/rest?version=v1"
    
    # Initialize credentials and instantiate Docs API service
    token_path = os.path.join(os.path.dirname(__file__), "token.json")
    store = file.Storage(token_path)
    creds = store.get()
    if not creds or creds.invalid:
        credentials_path = os.path.join(os.path.dirname(__file__), "credentials.json")
        flow = client.flow_from_clientsecrets(credentials_path, SCOPES)
        creds = tools.run_flow(flow, store)
    service = discovery.build(
        "docs",
        "v1",
        http=creds.authorize(Http()),
        discoveryServiceUrl=DISCOVERY_DOC,
    )

    # Do a document "get" request, includeTabsContent=True to get hierarchical structure
    result = service.documents().get(documentId=DOCUMENT_ID, includeTabsContent=True).execute()
    return result