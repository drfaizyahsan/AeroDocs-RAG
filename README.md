# AeroDocs-RAG

Retrieve instructions from aircraft maintenance manual for a given query


## start unvicorn 
`uvicorn api.main:app --reload`

## send a curl request
```
curl -X POST \                                            
http://localhost:8000/api/v1/query \
-H "Content-Type: application/json" \
-d '{
"question": "How do I inspect the bleed air valve?"
}'
```
