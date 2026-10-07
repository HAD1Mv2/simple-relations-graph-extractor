import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.relation_extract import RelationsExtractor
from app.utils import get_device

current_app_key = os.environ['APP_KEY']
openport = int(os.environ['PORT'])
ner_model_checkpoint = os.environ["NER_MODEL_PATH"]
rel_model_checkpoint = os.environ["REL_MODEL_PATH"]
# current_app_key = "12345XOXO"
# openport = 1025
# ner_model_checkpoint = "HAD1Mv2/distilbert-conllpp-eg"
# rel_model_checkpoint = "HAD1Mv2/distilbert-kbp37-eg"
device = get_device()

class ArticleInput(BaseModel):

    article: str

app = FastAPI()
rel_extractor = RelationsExtractor(ner_model_checkpoint, rel_model_checkpoint, device)

@app.get('/')
def engine_name(response: JSONResponse):
    
    name ="Relations Extraction"
    version = "September 2026"
    port = openport
    
    result = {
		"name":name,
		"version":version,
		"port":port
	}
    
    return result


@app.post('/get_relations', 
        status_code = 200, 
        openapi_extra={
            "parameters": [
                {
                    "name": "app-key",
                    "in": "header",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Custom application key"
                }
            ]
        })
def get_relation(input_:ArticleInput, request: Request, response: JSONResponse):
    """
    Get relations data from article
    """
    app_key = request.headers.get('app-key')
    if app_key != current_app_key:
        response.status_code = 403
        status=0
        return {'status':status}
        
    try:
        input_ = input_.model_dump()
        entities_data, relations_data = rel_extractor.extract_from_article(input_['article'])
        status = 1 # Success
    except Exception as e:
        entities_data, relations_data = [], []
        status = 0 #  Failed
        print(e)

    return {'status': status, 'entities_data':entities_data, 'relations_data':relations_data }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=openport)