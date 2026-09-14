from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from datetime import timedelta, datetime
from typing import List, Optional, Dict, Any, Union
import uuid
from pydantic import BaseModel, ValidationError
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()
from whatsapp.router import router as whatsapp_router
from whatsapp.services.paystack_webhook import router as paystack_webhook_router
from whatsapp.services.onboarding import router as onboarding_router
from whatsapp.scheduler import scheduler, validate_templates
from agents import evaluate_theory_answer, generate_learning_progress, evaluate_waec_theory
from swot_analysis import generate_swot_analysis
from save_swot_to_db import save_swot_to_database, get_swot_database_payload, get_db_session
from sqlalchemy import text
from models import (
    TheoryEvaluationRequest, 
    EssayEvaluationRequest, ComprehensionEvaluationRequest, 
    EssayEvaluationResponse, 
    ComprehensionEvaluationResponse, TheoryEvaluationResponse,
    MathematicsEvaluationRequest, MathematicsEvaluationResponse,
    LearningProgressRequest, LearningProgressResponse,
    SWOTAnalysisRequest, SWOTAnalysisResponse,
    AdaptationResponse,
    WAECTheoryEvalRequest, WAECTheoryEvalResponse
)
from learning_pathway_agent import adapt_pathway, generate_pathways_for_student
from save_pathway_to_db import fetch_pathway
from mongo_token_db import get_student_total_tokens
from agents import create_evaluation_agent_for_schema
from langchain_core.messages import HumanMessage
import asyncio
import json
import logging
import traceback
import aiohttp
from faker import Faker
from livekit import api
import os


logging.basicConfig(level=logging.INFO)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

ALERT_RECIPIENTS = [e.strip() for e in os.getenv("INTERNAL_MAIL_RECIPIENTS", "").split(",") if e.strip()]

async def send_error_email(subject: str, error: Exception, context: str = ""):
    """Send an error alert email via the internal mail API. Skipped if INTERNAL_MAIL_URL is not set."""
    mail_url = os.getenv("INTERNAL_MAIL_URL")
    mail_from = os.getenv("INTERNAL_MAIL_FROM", "alerts@publicaai.com")
    if not mail_url:
        logger.warning("[EMAIL-ALERT] INTERNAL_MAIL_URL not set — skipping error email")
        return
    tb = traceback.format_exc()
    html = f"""
    <h2 style="color:#cc0000;">Prep AI Error Alert</h2>
    <p><strong>Component:</strong> {context}</p>
    <p><strong>Error:</strong> {type(error).__name__}: {error}</p>
    <pre style="background:#f4f4f4;padding:12px;border-radius:6px;font-size:12px;">{tb}</pre>
    <p style="color:#888;font-size:11px;">Sent automatically by Prep AI backend</p>
    """
    payload = {"from": mail_from, "to": ALERT_RECIPIENTS, "subject": subject, "html": html}
    mail_api_key = os.getenv("INTERNAL_MAIL_API_KEY", "")
    headers = {"ApiKey": mail_api_key} if mail_api_key else {}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(mail_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status in (200, 201):
                    logger.info(f"[EMAIL-ALERT] Error email sent: {subject}")
                else:
                    body = await resp.text()
                    logger.warning(f"[EMAIL-ALERT] Mail API returned {resp.status}: {body}")
    except Exception as mail_err:
        logger.error(f"[EMAIL-ALERT] Failed to send error email: {mail_err}")


_scheduler_lock_fd = None


def _claim_scheduler_lock() -> bool:
    """
    Advisory file lock — ensures only ONE gunicorn worker starts the scheduler.
    Returns True if this process acquired the lock (should start scheduler).
    Falls back to True on Windows (no fcntl) so dev still works.
    """
    global _scheduler_lock_fd
    try:
        import fcntl
        _scheduler_lock_fd = open("/tmp/prepai_scheduler.lock", "w")
        fcntl.flock(_scheduler_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except ImportError:
        return True   # Windows / no fcntl — always start (single process dev)
    except OSError:
        return False  # Lock already held by another gunicorn worker


_V2_ENABLED = _os.getenv("PREPAI_WA_V2_ENABLED", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _owns_scheduler = _claim_scheduler_lock()
    if _owns_scheduler:
        logger.info(f"[lifespan] PID {os.getpid()} starting scheduler")
        scheduler.start()
        await validate_templates()
        if _V2_ENABLED:
            from whatsapp.v2.scheduler import start_scheduler as _start_v2

            _start_v2()
            logger.info("[lifespan] WhatsApp v2 scheduler started")
    else:
        logger.info(f"[lifespan] PID {os.getpid()} — scheduler already running in another worker")
    yield
    if _owns_scheduler:
        if _V2_ENABLED:
            from whatsapp.v2.scheduler import stop_scheduler as _stop_v2

            _stop_v2()
        scheduler.shutdown()


app = FastAPI(
    title="PREP AI AGENT",
    description="An API with specialized agents for generating and evaluating educational questions.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(whatsapp_router)
app.include_router(paystack_webhook_router)
app.include_router(onboarding_router)

# WhatsApp v2 (Bola) — flag-gated; live system unchanged when flag is off (HR-2)
import os as _os

if _os.getenv("PREPAI_WA_V2_ENABLED", "false").lower() == "true":
    from whatsapp.v2.inbound.webhook import router as _v2_webhook_router
    from whatsapp.v2.inbound.payment_confirmed import router as _v2_payment_confirmed_router

    app.include_router(_v2_webhook_router)
    app.include_router(_v2_payment_confirmed_router)
    _app_logger = logging.getLogger("app")
    _app_logger.info("WhatsApp v2 (Bola) routers mounted")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("=" * 80)
    logger.error("❌ VALIDATION ERROR")
    logger.error(f"URL: {request.url}")
    logger.error(f"Method: {request.method}")
    
    # Try to get raw body
    try:
        body = await request.body()
        logger.error(f"Raw request body: {body.decode('utf-8')}")
    except Exception as e:
        logger.error(f"Could not read request body: {e}")
    
    logger.error(f"Validation errors: {json.dumps(exc.errors(), indent=2)}")
    logger.error("=" * 80)
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": str(exc.body) if hasattr(exc, 'body') else None
        }
    )

@app.post("/api/evaluate-waec", response_model=Union[TheoryEvaluationResponse, dict])
async def evaluate_waec(request: TheoryEvaluationRequest):
    """
    Evaluates WAEC theory answers for multiple subjects.
    
    Supported subjects:
    - Government
    - Literature-in-English
    - Biology
    - IRS (Islamic Religious Studies)
    - Accounting
    - Chemistry
    - Physics
    - History
    - Economics
    - Christian-Religious-Studies
    - Civic-Education
    - Geography
    """
    allowed_subjects = [
        "government", "literature-in-english", "biology", "irs", "accounting",
        "chemistry", "physics", "history", "economics", "christian-religious-studies",
        "civic-education"
    ]
    if request.subject.lower() not in allowed_subjects:
        raise HTTPException(
            status_code=400, 
            detail=f"Subject must be one of: {', '.join(allowed_subjects)}"
        )
    return await evaluate_theory_answer(request)


@app.post("/api/evaluate-essay", response_model=EssayEvaluationResponse)
async def evaluate_essay(request: EssayEvaluationRequest):
    try:
        if not request.user_answer.strip() or len(request.user_answer.strip()) < 5:
            raise HTTPException(status_code=400, detail="The user answer is too short to be evaluated.")

        user_message = (
            "Please evaluate the following student's essay based on WAEC standards.\n\n"
            f"**Essay Question:**\n{request.question}\n\n"
            f"**Model Answer (for reference):**\n{request.answer}\n\n"
            f"**Student's Answer to Evaluate:**\n{request.user_answer}\n\n"
        )

        agent = create_evaluation_agent_for_schema(EssayEvaluationResponse)

        logger.info("Invoking evaluation agent for an essay.")
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": str(uuid4())}}
        )

        # Extract structured response as per ProviderStrategy
        structured_result = result["structured_response"]
        validated_response = EssayEvaluationResponse.model_validate(structured_result)
        logger.info("Successfully generated essay evaluation")
        return validated_response

    except Exception as e:
        logger.error(f"An error occurred during essay evaluation: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        asyncio.create_task(send_error_email("[Prep AI] Essay Evaluation Failed", e, "Essay evaluation endpoint"))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate-comprehension", response_model=ComprehensionEvaluationResponse)
async def evaluate_comprehension(request: ComprehensionEvaluationRequest):
    """Evaluates a student's comprehension answers using the evaluation agent."""
    try:
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided for evaluation.")

        questions_text = []
        has_empty_answers = False
        for i, q in enumerate(request.questions, 1):
            user_answer_stripped = (q.user_answer or "").strip()
            if not user_answer_stripped:
                student_answer_line = "Student's Answer: [BLANK — the student did not provide any answer]"
                has_empty_answers = True
            else:
                student_answer_line = f"Student's Answer: {q.user_answer}"
            questions_text.append(
                f"Question {i}: {q.question}\n"
                f"Reference Answer (examiner's model answer — for your reference ONLY, do NOT score this): {q.correct_answer}\n"
                f"{student_answer_line}\n"
            )

        empty_answer_note = (
            "\nCRITICAL RULE: One or more questions show [BLANK] for the Student's Answer. "
            "Any question where the Student's Answer is [BLANK] or empty MUST receive a score of 0. "
            "Do NOT use the Reference Answer as a substitute for a missing student response.\n"
        ) if has_empty_answers else ""

        user_message = (
            "Please evaluate this student's answers to comprehension questions based on WAEC standards.\n\n"
            f"**Passage:**\n{request.passage}\n\n"
            f"**Questions and Answers:**\n" + "\n".join(questions_text) + "\n"
            f"{empty_answer_note}"
            "Important: Evaluate EACH question individually with its own score (0-5) and feedback. "
            "The Reference Answer is provided so you can compare the student's response against the correct answer — evaluate ONLY the Student's Answer. "
            "The response should include individual evaluations for each question."
        )

        agent = create_evaluation_agent_for_schema(ComprehensionEvaluationResponse)

        logger.info("Invoking evaluation agent for comprehension.")
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": str(uuid4())}}
        )

        # Extract structured response as per ProviderStrategy
        structured_result = result["structured_response"]
        validated_response = ComprehensionEvaluationResponse.model_validate(structured_result)
        logger.info("Successfully generated comprehension evaluation")
        return validated_response

    except Exception as e:
        logger.error(f"An error occurred during comprehension evaluation: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        asyncio.create_task(send_error_email("[Prep AI] Comprehension Evaluation Failed", e, "Comprehension evaluation endpoint"))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate-mathematics", response_model=MathematicsEvaluationResponse)
async def evaluate_mathematics(request: MathematicsEvaluationRequest):
    """Evaluates a student's mathematics answers using the evaluation agent."""
    try:
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided for evaluation.")

        questions_text = []
        for i, q in enumerate(request.questions, 1):
            questions_text.append(
                f"Question {i}: {q.question}\n"
                f"Correct Answer: {q.correct_answer}\n"
                f"Student's Answer: {q.user_answer}\n"
            )

        # Build user message with optional instruction
        instruction_text = ""
        if request.instruction and request.instruction.strip():
            instruction_text = f"**Main Topic/Instruction:**\n{request.instruction}\n\n"
        
        user_message = (
            "Please evaluate this student's answers to mathematics questions based on WAEC standards.\n\n"
            f"{instruction_text}"
            f"**Questions and Answers:**\n" + "\n".join(questions_text) + "\n\n"
            "Important: Evaluate EACH question individually with its own score (0-10) and feedback. "
            "Mathematics requires precision. Award marks for correct method, working, and final answer. "
            "Be specific about where errors occurred (formula, substitution, calculation, units)."
        )

        agent = create_evaluation_agent_for_schema(MathematicsEvaluationResponse)

        logger.info("Invoking mathematics evaluation agent.")
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": str(uuid4())}}
        )

        # Extract structured response as per ProviderStrategy
        structured_result = result["structured_response"]
        validated_response = MathematicsEvaluationResponse.model_validate(structured_result)
        logger.info("Successfully generated mathematics evaluation")
        return validated_response

    except Exception as e:
        logger.error(f"An error occurred during mathematics evaluation: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        asyncio.create_task(send_error_email("[Prep AI] Mathematics Evaluation Failed", e, "Mathematics evaluation endpoint"))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate-waec-theory", response_model=WAECTheoryEvalResponse)
async def evaluate_waec_theory_endpoint(request: WAECTheoryEvalRequest):
    """
    Evaluates WAEC theory answers for all supported subjects using subject-specific marking schemes.

    **Supported subjects:**
    - biology, chemistry, physics, mathematics (or maths)
    - accounting, economics (or econs), government
    - crs, irs, literature (or literature-in-english)
    - english — requires `english_type`: "essay", "comprehension", or "letter"

    **For English subjects**, pass `english_type` to select the correct marking scheme.
    For comprehension, also pass `passage`.
    """
    try:
        return await evaluate_waec_theory(request)
    except Exception as e:
        logger.error(f"Error in /api/evaluate-waec-theory: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        asyncio.create_task(send_error_email("[Prep AI] WAEC Theory Evaluation Failed", e, "evaluate-waec-theory endpoint"))
        raise HTTPException(status_code=500, detail=str(e))


class TokenRequest(BaseModel):
    studentId: str
    name: str
    subject: Optional[Any] = None  # Allow any to prevent coercion before we check type
    subjectId: Optional[str] = None # Support direct subjectId if sent separately
    topic: Optional[str] = None
    token: Optional[str] = None # Received from frontend body

@app.post("/api/livekit-token")
async def get_livekit_token(request: Request, body: TokenRequest):
    try:
        logger.info("=" * 80)
        logger.info("🎫 LIVEKIT TOKEN REQUEST")
        
        # 1. Extract auth token from Header or Body
        # Frontend sends: "Authorization": `Bearer ${authToken}`
        auth_header = request.headers.get("Authorization")
        auth_token = None
        if auth_header and auth_header.startswith("Bearer "):
            auth_token = auth_header.split(" ")[1]
        
        # Fallback to token in body if not in header
        if not auth_token and body.token:
            auth_token = body.token

        # Handle subject potentially being a dictionary or having a separate ID
        subject_name = None
        subject_id = body.subjectId # Try direct field first
        
        if body.subject:
            if isinstance(body.subject, dict):
                subject_name = body.subject.get("name")
                if not subject_id:
                    subject_id = body.subject.get("id")
            else:
                subject_name = str(body.subject)

        logger.info(f"Student Name: {body.name}")
        logger.info(f"Student ID: {body.studentId}")
        logger.info(f"Subject Name: {subject_name or 'None'}")
        logger.info(f"Subject ID: {subject_id or 'None'}")
        logger.info(f"Topic: {body.topic or 'Not specified'}")
        logger.info(f"Auth Token Provided: {'Yes' if auth_token else 'No'}")
        
        room_name = f"prepAI-{uuid.uuid4().hex[:6]}"
        
        token = generate_livekit_token(
            room=room_name, 
            student_id=body.studentId, 
            name=body.name, 
            subject=subject_name, 
            topic=body.topic,
            subject_id=subject_id,
            auth_token=auth_token
        )
        
        logger.info(f"✅ Token generated for room: {room_name}")
        logger.info("=" * 80)
        
        return {"token": token, "room": room_name}
    except Exception as e:
        logger.error(f"❌ Token generation failed: {e}")
        return {"error": str(e), "success": False}

def generate_livekit_token(
    room: str, 
    student_id: str, 
    name: str, 
    subject: Optional[str] = None, 
    topic: Optional[str] = None,
    subject_id: Optional[str] = None,
    auth_token: Optional[str] = None
) -> str:
    import json
    
    # Create participant metadata with subject and topic
    participant_metadata = {}
    if subject:
        participant_metadata["subject"] = subject
    if topic:
        participant_metadata["topic"] = topic
    if subject_id:
        participant_metadata["subjectId"] = subject_id
    if auth_token:
        participant_metadata["authToken"] = auth_token
        
    logger.info(f"Participant Metadata: {participant_metadata}")
    
    token = (
        api.AccessToken(
            os.getenv('LIVEKIT_API_KEY'),
            os.getenv('LIVEKIT_API_SECRET')
        )
        .with_identity(student_id)
        .with_name(name)
        .with_metadata(json.dumps(participant_metadata) if participant_metadata else "")
        .with_ttl(timedelta(minutes=60)) # Increased to 60 mins
        .with_grants(api.VideoGrants(
            room_join=True,
            room=room,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True
        ))
    )
    return token.to_jwt()

@app.post("/api/learning-progress", response_model=LearningProgressResponse)
async def generate_student_learning_progress(request: LearningProgressRequest):
    """
    Generates a comprehensive learning progress report for a student based on their assessment data.
    
    This endpoint analyzes the student's performance, providing personalized insights, 
    strengths, weaknesses, and actionable recommendations.

    """
    # Log incoming request data
    logger.info("=" * 80)
    logger.info("📥 LEARNING PROGRESS REQUEST RECEIVED")
    logger.info(f"Student ID: {request.studentId}")
    logger.info(f"Subject ID: {request.subjectId}")
    logger.info(f"Number of Assessments: {len(request.attemptedData)}")
    logger.info(f"Exam Type: {request.exam_type}")
    logger.info(f"Assessment topics: {[a.topic_name for a in request.attemptedData]}")
    logger.info(f"Raw request data: {json.dumps(request.model_dump(), indent=2)}")
    logger.info("=" * 80)
    
    return await generate_learning_progress(request)

@app.post("/api/swot-analysis")
async def generate_swot_report(
    request: SWOTAnalysisRequest,
    save_to_db: bool = True,  # Changed to True - auto-save by default
    subject_id: Optional[str] = None,
    assessment_id: Optional[str] = None,
    analysis_type: str = "initial"
):
    """
    Generates a comprehensive SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis 
    for a student based on their initial assessment performance.
    
    Query Parameters:
    - save_to_db: If true, saves directly to MySQL database (default: true)
    - subject_id: Optional subject UUID (for subject-specific SWOT, otherwise NULL for overall)
    - assessment_id: Optional assessment UUID (to link SWOT to specific assessment)
    - analysis_type: Type of analysis - 'initial', 'weekly', or 'monthly' (default: 'initial')
    
    Example: POST /api/swot-analysis?subject_id=xxx&assessment_id=yyy
    """
    # Body analysis_type takes priority over query param
    if request.analysis_type:
        analysis_type = request.analysis_type

    # Log incoming request
    logger.info("📊 SWOT ANALYSIS REQUEST RECEIVED")
    try:
        logger.info(f"Full Request Data: {json.dumps(request.model_dump(), indent=2)}")
    except Exception as log_err:
        logger.error(f"Failed to log request data: {log_err}")

    # Save raw frontend payload to MongoDB immediately — regardless of what happens next
    save_swot_payload_to_mongo(request.studentId, request)

    # Generate SWOT analysis
    try:
        swot_result = await generate_swot_analysis(request, max_retries=5)
    except Exception as swot_err:
        asyncio.create_task(send_error_email("[Prep AI] SWOT Analysis Generation Failed", swot_err, f"SWOT Analysis — studentId: {request.studentId}"))
        raise
    
    # Prepare response
    response_data = {
        "swot_analysis": swot_result.model_dump(),
        "database_payload": None,
        "saved_to_database": False
    }
    
    # Get database-formatted payload
    db_payload = get_swot_database_payload(
        swot_response=swot_result,
        student_id=request.studentId,
        subject_id=subject_id,
        analysis_type=analysis_type
    )
    
    response_data["database_payload"] = db_payload
    
    # Optionally save to MySQL database
    if save_to_db:
        try:
            # Prepare subject and topic performance data - handle both camelCase and snake_case from request
            # SWOTAnalysisRequest model uses aliases, but we need to ensure we get the data whichever way it arrived
            req_dict = request.model_dump()
            logger.info(f"🔍 req_dict keys: {list(req_dict.keys())}")
            logger.info(f"🔍 subjectPerformance raw: {req_dict.get('subjectPerformance')}")
            logger.info(f"🔍 subject_performance raw: {req_dict.get('subject_performance')}")
            subject_perf = req_dict.get('subjectPerformance') or req_dict.get('subject_performance') or []
            topic_perf = req_dict.get('topicPerformance') or req_dict.get('topic_performance') or []
            logger.info(f"🔍 subject_perf count: {len(subject_perf)}, topic_perf count: {len(topic_perf)}")
            
            # Get student profile and time management data
            student_prof = req_dict.get('student_profile')
            time_mgmt = req_dict.get('time_management') or req_dict.get('timeManagement')
            
            db_response = save_swot_to_database(
                swot_response=swot_result,
                student_id=request.studentId,
                subject_performance_list=subject_perf,
                topic_performance_list=topic_perf,
                student_profile=student_prof,
                time_management=time_mgmt,
                subject_id=subject_id,
                assessment_id=assessment_id,
                analysis_type=analysis_type
            )
            response_data["saved_to_database"] = True
            response_data["database_response"] = db_response
            logger.info("✅ SWOT analysis saved to MySQL database (both tables)")

            # INTEGRATION: Automatically generate learning pathways after SWOT
            try:
                logger.info(f"🚀 Triggering automatic pathway generation for student: {request.studentId}")
                
                # The generator handles None for exam_date using its own defaults
                pathway_results = await generate_pathways_for_student(
                    student_id=request.studentId,
                    swot_analysis_id=db_response.get("inserted_id"),
                    exam_date=request.exam_date
                )
                response_data["pathway_integration"] = pathway_results
                logger.info(f"✅ Automatically generated {pathway_results.get('pathways_generated', 0)} pathways")
            except Exception as pathway_error:
                logger.error(f"⚠️ SWOT saved, but pathway generation failed: {pathway_error}")
                response_data["pathway_integration_error"] = str(pathway_error)
                asyncio.create_task(send_error_email("[Prep AI] Learning Pathway Generation Failed", pathway_error, f"Pathway generation — studentId: {request.studentId}"))

        except Exception as e:
            logger.error(f"❌ Failed to save SWOT to database: {e}")
            response_data["database_error"] = str(e)
            asyncio.create_task(send_error_email("[Prep AI] SWOT Database Save Failed", e, f"SWOT DB save — studentId: {request.studentId}"))

    # Fire-and-forget: verify SWOT was persisted after 15 minutes
    asyncio.create_task(_verify_swot_saved(request.studentId, delay_seconds=900))
    logger.info(f"[SWOT-VERIFY] Background verification scheduled for student {request.studentId} in 15 min")

    return response_data

@app.post("/api/get-tutor-prompt")
async def get_tutor_prompt(subject: str, topic: Optional[str] = None):
    """
    Get AI tutor prompt based on subject and topic
    
    Args:
        subject: Subject name (e.g., "english", "mathematics", "biology")
        topic: Optional topic name (e.g., "comprehension", "algebra")
    
    Returns:
        prompt: The appropriate tutor prompt text
        prompt_type: "core" or "topic"
        subject: Subject name
        topic: Topic name (if provided)
    
    Example:
        POST /api/get-tutor-prompt?subject=english&topic=comprehension
        POST /api/get-tutor-prompt?subject=mathematics  (returns core prompt)
    """
    import os
    
    # Normalize subject and topic names
    subject_normalized = subject.lower().replace(" ", "_")
    topic_normalized = topic.lower().replace(" ", "_") if topic else None
    
    prompts_dir = "prompts"
    subject_dir = os.path.join(prompts_dir, subject_normalized)
    
    # Check if subject directory exists
    if not os.path.exists(subject_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Subject '{subject}' not found. Available subjects: english, mathematics, biology"
        )
    
    prompt_text = None
    prompt_type = None
    prompt_file = None
    
    # Try to load topic-specific prompt first
    if topic_normalized:
        topic_file = os.path.join(subject_dir, f"{topic_normalized}.txt")
        if os.path.exists(topic_file):
            with open(topic_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            prompt_type = "topic"
            prompt_file = f"{topic_normalized}.txt"
            logger.info(f"✅ Loaded topic prompt: {subject}/{topic_normalized}")
        else:
            logger.warning(f"⚠️  Topic prompt not found: {subject}/{topic_normalized}, falling back to core")
    
    # Fall back to core prompt if topic not found or not provided
    if not prompt_text:
        core_file = os.path.join(subject_dir, f"core_{subject_normalized}_tutor.txt")
        if os.path.exists(core_file):
            with open(core_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            prompt_type = "core"
            prompt_file = f"core_{subject_normalized}_tutor.txt"
            logger.info(f"✅ Loaded core prompt: {subject}/core")
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Core prompt not found for subject '{subject}'"
            )
    
    return {
        "success": True,
        "prompt": prompt_text,
        "prompt_type": prompt_type,
        "prompt_file": prompt_file,
        "subject": subject,
        "topic": topic or "core"
    }


@app.get("/api/available-subjects")
async def get_available_subjects():
    """
    Get list of available subjects and their topics
    
    Returns:
        subjects: Dictionary of subjects with available topics
    """
    import os
    
    prompts_dir = "prompts"
    subjects_data = {}
    
    if not os.path.exists(prompts_dir):
        return {"subjects": {}}
    
    for subject in os.listdir(prompts_dir):
        subject_path = os.path.join(prompts_dir, subject)
        if os.path.isdir(subject_path):
            topics = []
            has_core = False
            
            for file in os.listdir(subject_path):
                if file.endswith('.txt'):
                    if file.startswith('core_'):
                        has_core = True
                    else:
                        # Extract topic name from filename
                        topic_name = file.replace('.txt', '').replace('_', ' ').title()
                        topics.append({
                            "name": topic_name,
                            "file": file.replace('.txt', '')
                        })
            
            subjects_data[subject] = {
                "has_core": has_core,
                "topics": topics
            }
    
    return {"subjects": subjects_data}


class AdaptationRequest(BaseModel):
    pathwayId: str
    studentId: str
    current_milestone: int
    milestone_score: float
    pass_mark: float
    trigger: str = "weekly_review"
    recent_performance: Optional[Dict[str, Any]] = None
    exam_date: Optional[str] = None

@app.post("/api/adapt-pathway", response_model=AdaptationResponse)
async def adapt_student_pathway(request: AdaptationRequest):
    """
    Adapts an existing learning pathway based on recent student performance.
    """
    logger.info(f"🔄 ADAPTATION REQUEST RECEIVED for pathway: {request.pathwayId}")
    
    # Construct performance data override from request
    performance_override = None
    if request.recent_performance:
        performance_override = request.recent_performance
        # Add basic metrics if missing from payload but present in root
        if "score" not in performance_override:
            performance_override["score"] = request.milestone_score
            
    return await adapt_pathway(
        pathway_id=request.pathwayId, 
        trigger=request.trigger,
        performance_data_override=performance_override
    )

@app.get("/api/pathway/{pathway_id}")
async def get_pathway_details(pathway_id: str):
    """
    Retrieves the full details of a learning pathway, including milestones and overall strategy.
    
    Args:
        pathway_id: UUID of the pathway
    """
    logger.info(f"🔍 FETCHING PATHWAY DETAILS: {pathway_id}")
    return await fetch_pathway(pathway_id)

@app.get("/api/pathway/{pathway_id}/milestones")
async def get_pathway_milestones(pathway_id: str):
    """
    Retrieves the individual milestones for a specific learning pathway.
    
    Args:
        pathway_id: UUID of the pathway
    """
    logger.info(f"� FETCHING MILESTONES for pathway: {pathway_id}")
    # Milestones are part of the pathway_data JSONB in learning_pathways table
    pathway = await fetch_pathway(pathway_id)
    return {
        "pathway_id": pathway_id,
        "subject_name": pathway.get("subject_name"),
        "milestones": pathway.get("pathway_data", {}).get("milestones", [])
    }

@app.post("/api/generate-pathways")
async def trigger_pathway_generation(student_id: str, swot_analysis_id: str, exam_date: Optional[str] = None):
    """
    Triggers the generation of initial learning pathways for all subjects after SWOT analysis.
    
    Args:
        student_id: UUID of the student
        swot_analysis_id: UUID of the SWOT analysis to use
        exam_date: Optional exam date (ISO format)
    """
    logger.info(f"🚀 PATHWAY GENERATION REQUEST RECEIVED for student: {student_id}")
    return await generate_pathways_for_student(student_id, swot_analysis_id, exam_date)

class TokenUsageQuery(BaseModel):
    studentId: str
    subjectId: Optional[str] = None
    tokensUsed: Optional[int] = None  # Included for compatibility with frontend payload
    requestId: Optional[str] = None  # Included for compatibility with frontend payload

@app.post("/api/get-token-usage")
async def get_token_usage(request: TokenUsageQuery):
    """
    Retrieves the total token usage for a student, optionally filtered by subject.
    """
    # Clean up subject_id: treat "string" (Swagger default), "null", or empty strings as None
    subject_id = request.subjectId
    if subject_id in [None, "string", "null", ""]:
        subject_id = None
        
    logger.info(f"📊 TOKEN USAGE QUERY: Student={request.studentId}, Subject={subject_id or 'ALL'}")
    
    total_tokens = get_student_total_tokens(
        student_id=request.studentId,
        subject_id=subject_id
    )
    
    return {
        "success": True,
        "student_id": request.studentId,
        "subject_id": subject_id,
        "total_tokens": total_tokens
    }

# --- MongoDB connection for feedback ---
def _get_feedback_collection():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        return None
    try:
        client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        client.admin.command('ping')
        return client["Voice_agent_db"]["conversation_feedback"]
    except Exception as e:
        logger.error(f"MongoDB feedback connection failed: {e}")
        return None

_feedback_col = _get_feedback_collection()


# --- MongoDB collection for SWOT raw payloads ---
def _get_swot_payload_collection():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        return None
    try:
        client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        client.admin.command('ping')
        return client["Voice_agent_db"]["swot_raw_payloads"]
    except Exception as e:
        logger.error(f"MongoDB swot_payload connection failed: {e}")
        return None

_swot_payload_col = _get_swot_payload_collection()


def save_swot_payload_to_mongo(student_id: str, request: SWOTAnalysisRequest):
    """Save the raw SWOT request payload to MongoDB for future regeneration."""
    if _swot_payload_col is None:
        logger.warning("⚠️ MongoDB unavailable — SWOT payload not saved")
        return
    try:
        doc = {
            "studentId": student_id,
            "payload": request.model_dump(),
            "savedAt": datetime.utcnow(),
        }
        _swot_payload_col.replace_one(
            {"studentId": student_id},
            doc,
            upsert=True
        )
        logger.info(f"✅ SWOT raw payload saved to MongoDB for student: {student_id}")
    except Exception as e:
        logger.error(f"❌ Failed to save SWOT payload to MongoDB: {e}")


async def _verify_swot_saved(student_id: str, delay_seconds: int = 900):
    """
    Background task: waits delay_seconds then checks if the SWOT analysis
    was actually persisted to MySQL. If not, re-runs generation automatically.
    Only runs for the specific student whose SWOT was just generated.
    """
    await asyncio.sleep(delay_seconds)
    logger.info(f"[SWOT-VERIFY] Checking DB for student {student_id} after {delay_seconds}s...")

    session = None
    try:
        session = get_db_session()
        row = session.execute(
            text("SELECT id FROM swot_analysis WHERE studentId = :sid LIMIT 1"),
            {"sid": student_id}
        ).fetchone()
    except Exception as e:
        logger.error(f"[SWOT-VERIFY] DB check failed for {student_id}: {e}")
        return
    finally:
        if session:
            session.close()

    if row:
        logger.info(f"[SWOT-VERIFY] SWOT confirmed in DB for {student_id} — no action needed.")
        return

    logger.warning(f"[SWOT-VERIFY] SWOT NOT found in DB for {student_id} — triggering regeneration...")
    try:
        await regenerate_swot_and_pathways(student_id)
        logger.info(f"[SWOT-VERIFY] Regeneration complete for {student_id}")
    except Exception as e:
        logger.error(f"[SWOT-VERIFY] Regeneration failed for {student_id}: {e}")


# --- Feedback endpoint model ---
class ConversationFeedbackRequest(BaseModel):
    studentId: str
    feedback: str           # "thumbs_up" | "thumbs_down"
    message: str            # the tutor response the user is rating
    sessionId: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None


@app.post("/api/voice/save-feedback")
async def save_conversation_feedback(request: ConversationFeedbackRequest):
    """
    Save feedback (thumbs up/down) on a tutor response.

    Frontend calls this when the user clicks the thumbs-up or thumbs-down
    button on a specific tutor message.

    Body:
        studentId  – student's unique ID  (required)
        feedback   – "thumbs_up" or "thumbs_down"  (required)
        message    – the tutor's response text being rated  (required)
        sessionId  – optional LiveKit session ID
        subject    – optional subject name
        topic      – optional topic name
    """
    if request.feedback not in ("thumbs_up", "thumbs_down"):
        raise HTTPException(status_code=400, detail="feedback must be 'thumbs_up' or 'thumbs_down'")

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    if _feedback_col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        doc = {
            "studentId": request.studentId,
            "sessionId": request.sessionId,
            "feedback": request.feedback,
            "message": request.message,
            "subject": request.subject,
            "topic": request.topic,
            "createdAt": datetime.utcnow(),
        }
        result = _feedback_col.insert_one(doc)
        logger.info(
            f"✅ Feedback saved: {request.feedback} | student={request.studentId} "
            f"| message='{request.message[:60]}...' | id={result.inserted_id}"
        )
        return {
            "success": True,
            "id": str(result.inserted_id),
            "feedback": request.feedback,
        }
    except Exception as e:
        logger.error(f"❌ Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/regenerate-swot/{student_id}")
async def regenerate_swot_and_pathways(student_id: str, exam_date: Optional[str] = None):
    """
    Re-runs SWOT analysis and regenerates all learning pathways for a student
    using their last saved SWOT request payload stored in MongoDB.

    Args:
        student_id: The student's UUID
        exam_date: Optional override for exam date (ISO format). Uses saved value if not provided.
    """
    logger.info(f"🔄 REGENERATE SWOT requested for student: {student_id}")

    if _swot_payload_col is None:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")

    # Fetch saved payload
    doc = _swot_payload_col.find_one({"studentId": student_id})
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"No saved SWOT payload found for student {student_id}. Student must complete initial assessment first."
        )

    try:
        saved_payload = doc["payload"]

        if exam_date:
            saved_payload["exam_date"] = exam_date

        request = SWOTAnalysisRequest(**saved_payload)
        logger.info(f"📊 Re-running SWOT analysis for student: {student_id}")
        swot_result = await generate_swot_analysis(request, max_retries=5)
        req_dict = request.model_dump()
        db_response = save_swot_to_database(
            swot_response=swot_result,
            student_id=student_id,
            subject_performance_list=req_dict.get('subjectPerformance') or req_dict.get('subject_performance') or [],
            topic_performance_list=req_dict.get('topicPerformance') or req_dict.get('topic_performance') or [],
            student_profile=req_dict.get('student_profile'),
            time_management=req_dict.get('time_management') or req_dict.get('timeManagement'),
            analysis_type="regenerated"
        )
        logger.info(f"✅ SWOT re-saved to MySQL. ID: {db_response.get('inserted_id')}")

        # Regenerate all learning pathways
        pathway_results = await generate_pathways_for_student(
            student_id=student_id,
            swot_analysis_id=db_response.get("inserted_id"),
            exam_date=request.exam_date
        )
        logger.info(f"✅ Regenerated {pathway_results.get('pathways_generated', 0)} pathways")

        return {
            "success": True,
            "student_id": student_id,
            "swot_analysis_id": db_response.get("inserted_id"),
            "pathways_generated": pathway_results.get("pathways_generated", 0),
            "pathway_results": pathway_results,
        }

    except Exception as e:
        logger.error(f"❌ Regeneration failed for student {student_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": uuid4().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    