from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from engine import generate_optimized_schedule, load_all_department_requests
import numpy as np
from rl_engine import train_rl_agent

app = FastAPI(
    title="Indian Railways AI Block Optimizer API",
    description="Backend API providing optimized joint maintenance blocks and train conflict analytics.",
    version="1.0.0"
)

# Enable CORS so the React/Next.js frontend can connect smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "status": "online",
        "system": "Indian Railways Joint Block Optimization Engine",
        "docs_url": "/docs"
    }

@app.get("/api/requests/unoptimized")
def get_raw_requests():
    """Returns raw, separate requests from TMS, SMMS, and TDMS."""
    return {
        "status": "success",
        "unoptimized_requests": load_all_department_requests()
    }

@app.get("/api/requests/optimized")
def get_optimized_schedule():
    """Runs AI engine logic and returns merged joint blocks + efficiency metrics."""
    return generate_optimized_schedule()


@app.get("/api/requests/rl-optimized")
def get_rl_schedule():
    """Triggers the Reinforcement Learning agent and returns trained policies."""
    return train_rl_agent(episodes=500)