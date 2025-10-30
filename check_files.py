import google.generativeai as genai

genai.configure(api_key="AIzaSyDRsvt4NKLx7tjXzpxdsTvzjjPkpBZu8Jc")

models = genai.list_models()
for m in models:
    print(m.name)
