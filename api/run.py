import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:create_app", host="127.0.0.1", port=5001, reload=True)
