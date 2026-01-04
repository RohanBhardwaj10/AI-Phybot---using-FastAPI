from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from agent import run_agent

app = FastAPI(title="AI PhyBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static + template setup
templates = Jinja2Templates(directory="templates")



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the chat frontend."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(request: Request):
    """
    Receive message from frontend → get AI response → send back.
    """
    try:
        data = await request.json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return JSONResponse({"response": "Please enter a valid message."}, status_code=400)

        ai_response = run_agent(user_message)

        return JSONResponse({"response": ai_response})
    except Exception as e:
        print(f"❌ Error in /chat: {e}")
        return JSONResponse({"response": f"Server error: {str(e)}"}, status_code=500)

@app.post("/clear")
async def clear_memory():
    """
    Clear saved chat memory (reset chat_memory.json)
    """
    try:
        with open("chat_memory.json", "w", encoding="utf-8") as f:
            f.write('{"messages": []}')
        return JSONResponse({"message": "Chat memory cleared successfully!"})
    except Exception as e:
        return JSONResponse({"message": f"Failed to clear memory: {e}"}, status_code=500)


if __name__ == "__main__":
    print("🚀 Starting AI PhyBot FastAPI server at http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
