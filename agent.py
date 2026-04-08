# ================= IMPORTS =================
import os
import logging
import datetime
import google.cloud.logging
from google.cloud import datastore
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext


# ================= 1. LOGGING =================
try:
    cloud_logging_client = google.cloud.logging.Client()
    cloud_logging_client.setup_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)


# ================= 2. ENV =================
load_dotenv()
model_name = os.getenv("MODEL", "gemini-1.5-flash")

# 🔥 ADD YOUR DATABASE DETAILS HERE
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "concrete-list-491916-k3")
DATABASE_ID = os.getenv("GCP_DATABASE_ID", "genaiapacanu")


# ================= 3. DATABASE =================
try:
    db = datastore.Client(
        project=PROJECT_ID,
        database=DATABASE_ID
    )
    logging.info(f"✅ Connected to DB | Project: {PROJECT_ID}")
except Exception as e:
    logging.error(f"❌ DB Connection Failed: {e}")


# ================= DATABASE FUNCTIONS =================

def save_symptoms_to_db(symptoms: str):
    try:
        key = db.key("SymptomRecord")
        entity = datastore.Entity(key=key)
        entity.update({
            "symptoms": symptoms,
            "created_at": datetime.datetime.now()
        })
        db.put(entity)
    except Exception as e:
        logging.error(f"DB Error: {e}")


def save_full_report(symptoms, analysis, conditions, risk, response):
    try:
        key = db.key("MedicalReport")
        entity = datastore.Entity(key=key)
        entity.update({
            "symptoms": symptoms,
            "analysis": analysis,
            "conditions": conditions,
            "risk": risk,
            "final_response": response,
            "created_at": datetime.datetime.now()
        })
        db.put(entity)
    except Exception as e:
        logging.error(f"DB Error: {e}")


def list_reports():
    try:
        query = db.query(kind="MedicalReport")
        results = list(query.fetch(limit=5))

        if not results:
            return "No reports found."

        res = []
        for r in results:
            res.append({
                "symptoms": r.get("symptoms"),
                "risk": r.get("risk")
            })
        return res
    except Exception as e:
        return f"DB Error: {str(e)}"


# ================= STATE TOOL =================

def add_symptoms_to_state(tool_context: ToolContext, symptoms: str):
    tool_context.state["SYMPTOMS"] = symptoms
    save_symptoms_to_db(symptoms)
    return {"status": "ok"}


# ================= AGENTS =================

symptom_analysis_agent = Agent(
    name="symptom_analysis_agent",
    model=model_name,
    instruction="""
Analyze symptoms and extract key indicators.

SYMPTOMS:
{ SYMPTOMS }
""",
    output_key="symptom_analysis"
)

diagnosis_agent = Agent(
    name="diagnosis_agent",
    model=model_name,
    instruction="""
Suggest possible conditions.

SYMPTOM_ANALYSIS:
{ symptom_analysis }
""",
    output_key="possible_conditions"
)

risk_assessment_agent = Agent(
    name="risk_assessment_agent",
    model=model_name,
    instruction="""
Classify risk level (LOW, MEDIUM, HIGH, EMERGENCY)

POSSIBLE_CONDITIONS:
{ possible_conditions }
""",
    output_key="risk_level"
)

recommendation_agent = Agent(
    name="recommendation_agent",
    model=model_name,
    instruction="""
Provide recommendations.

POSSIBLE_CONDITIONS:
{ possible_conditions }

RISK_LEVEL:
{ risk_level }
""",
    output_key="final_response"
)


# ================= SAVE AGENT =================

def save_report_tool(tool_context: ToolContext):
    save_full_report(
        tool_context.state.get("SYMPTOMS"),
        tool_context.state.get("symptom_analysis"),
        tool_context.state.get("possible_conditions"),
        tool_context.state.get("risk_level"),
        tool_context.state.get("final_response"),
    )
    return {"status": "saved"}


save_agent = Agent(
    name="save_agent",
    model=model_name,
    instruction="Save report to database",
    tools=[save_report_tool]
)


# ================= WORKFLOW =================

symptoguard_workflow = SequentialAgent(
    name="symptoguard_workflow",
    sub_agents=[
        symptom_analysis_agent,
        diagnosis_agent,
        risk_assessment_agent,
        recommendation_agent,
        save_agent
    ]
)


# ================= ROOT AGENT =================

root_agent = Agent(
    name="symptoguard_agent",
    model=model_name,
    instruction="""
Welcome the user.
Ask for symptoms.
Store them and run analysis.
""",
    tools=[add_symptoms_to_state],
    sub_agents=[symptoguard_workflow]
)


# ================= API =================

app = FastAPI()

class UserRequest(BaseModel):
    symptoms: str


@app.post("/api/v1/symptoguard/chat")
async def chat(request: UserRequest):
    try:
        final_reply = ""

        async for event in root_agent.run_async({
            "SYMPTOMS": request.symptoms
        }):
            if hasattr(event, "text") and event.text:
                final_reply = event.text

        return {
            "status": "success",
            "response": final_reply
        }

    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/symptoguard/history")
def history():
    return {
        "status": "success",
        "data": list_reports()
    }


# ================= RUN =================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)