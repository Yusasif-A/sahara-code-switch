import logging
import json
import asyncio
import time
import re
import sys
from dotenv import load_dotenv
import os
import aiohttp
import uuid

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    metrics,
    get_job_context
)
from livekit.agents.llm import ChatContext, ChatMessage, ImageContent
from livekit.agents.voice import MetricsCollectedEvent, UserInputTranscribedEvent
from livekit.plugins import openai, silero, noise_cancellation
from livekit import api
from livekit.plugins import openai
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
from openai import AsyncOpenAI 
from mongo_token_db import store_token_usage 

def _get_log_stream():
    """Write to /proc/1/fd/1 (Docker PID 1 stdout) so child processes show in docker logs."""
    try:
        return open("/proc/1/fd/1", "w", buffering=1)
    except Exception:
        return sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(stream=_get_log_stream())],
    force=True
)
logger = logging.getLogger("waec_exam_agent")
logger.setLevel(logging.INFO)
logging.getLogger("pymongo").setLevel(logging.WARNING)


load_dotenv()

def get_mongodb_client():
    connection_string = os.getenv("MONGO_URI")
    if not connection_string:
        logger.warning("MONGO_URI is not set - memory features will be disabled")
        return None
    
    try:
        client = MongoClient(connection_string, server_api=ServerApi('1'))
        client.admin.command('ping')
        logger.info("Connected to MongoDB Atlas!")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None

# Initialize MongoDB client
mongo_client = get_mongodb_client()
if mongo_client is not None:
    db = mongo_client.get_database('Voice_agent_db')
    conversations_collection = db.get_collection("user_conversations")
else:
    conversations_collection = None

# Initialize OpenAI client for summarization
openai_async_client = AsyncOpenAI(base_url=os.getenv("PUBLICAAI_BASE_URL"), api_key=os.getenv("API_KEY"))


def _math_to_speech(text: str) -> str:
    """Convert LaTeX/math symbols to spoken English so TTS can pronounce them."""

    # --- Common physics/chemistry unit abbreviations (must run BEFORE generic slash handling) ---
    unit_map = [
        # Compound units with slashes
        (r'\bm/s\^2\b',        'metres per second squared'),
        (r'\bm/s²',            'metres per second squared'),
        (r'\bm\s*/\s*s\b',     'metres per second'),
        (r'\bkm/h\b',          'kilometres per hour'),
        (r'\bkm/hr\b',         'kilometres per hour'),
        (r'\bcm/s\b',          'centimetres per second'),
        (r'\bmm/s\b',          'millimetres per second'),
        (r'\bkg/m\^3\b',       'kilograms per cubic metre'),
        (r'\bkg/m³',           'kilograms per cubic metre'),
        (r'\bg/cm\^3\b',       'grams per cubic centimetre'),
        (r'\bg/cm³',           'grams per cubic centimetre'),
        (r'\bN/m\^2\b',        'Newtons per square metre'),
        (r'\bN/m²',            'Newtons per square metre'),
        (r'\bJ/kg\b',          'joules per kilogram'),
        (r'\bW/m\^2\b',        'watts per square metre'),
        (r'\bA/m\b',           'amperes per metre'),
        (r'\bV/m\b',           'volts per metre'),
        (r'\bmol/L\b',         'moles per litre'),
        (r'\bmol/dm\^3\b',     'moles per cubic decimetre'),
        (r'\bg/mol\b',         'grams per mole'),
        # Single SI units
        (r'\b(\d[\d.,]*)\s*m/s\b',   r'\1 metres per second'),
        (r'\b(\d[\d.,]*)\s*km/h\b',  r'\1 kilometres per hour'),
        # Chemical formulas (subscript numbers, e.g. H2O, CO2, H2SO4)
        (r'\bH2O\b',   'H 2 O'),
        (r'\bCO2\b',   'C O 2'),
        (r'\bO2\b',    'O 2'),
        (r'\bN2\b',    'N 2'),
        (r'\bH2\b',    'H 2'),
        (r'\bCl2\b',   'C L 2'),
        (r'\bH2SO4\b', 'H 2 S O 4'),
        (r'\bHCl\b',   'H C L'),
        (r'\bNaCl\b',  'sodium chloride'),
        (r'\bCaCO3\b', 'calcium carbonate'),
        (r'\bNaOH\b',  'sodium hydroxide'),
        (r'\bCH4\b',   'C H 4'),
        (r'\bNH3\b',   'N H 3'),
        (r'\bCO\b',    'C O'),
        # Unicode subscripts in chemical formulas
        (r'₂', ' 2'), (r'₃', ' 3'), (r'₄', ' 4'), (r'₅', ' 5'),
        (r'₆', ' 6'), (r'₇', ' 7'), (r'₈', ' 8'), (r'₉', ' 9'),
    ]
    for pat, repl in unit_map:
        text = re.sub(pat, repl, text)

    # --- LaTeX fractions: \frac{a}{b} -> "a divided by b" ---
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1 divided by \2', text)
    # --- Square roots: \sqrt{x}, \sqrt(x), sqrt(x), √x -> "square root of x" ---
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'square root of \1', text)
    text = re.sub(r'\\sqrt\(([^)]+)\)', r'square root of \1', text)
    text = re.sub(r'\bsqrt\(([^)]+)\)', r'square root of \1', text)
    text = re.sub(r'√\(([^)]+)\)', r'square root of \1', text)
    text = re.sub(r'√([a-zA-Z0-9]+)', r'square root of \1', text)
    # --- Powers/exponents ---
    text = re.sub(r'\^2\b', ' squared', text)
    text = re.sub(r'\^\{2\}', ' squared', text)
    text = re.sub(r'\^3\b', ' cubed', text)
    text = re.sub(r'\^\{3\}', ' cubed', text)
    text = re.sub(r'\^\{([^}]+)\}', r' to the power of \1', text)
    text = re.sub(r'\^([a-zA-Z0-9]+)', r' to the power of \1', text)
    # --- Superscript unicode ---
    text = text.replace('²', ' squared').replace('³', ' cubed')
    # --- Greek letters ---
    greek = {
        r'\\pi': 'pi', r'\\theta': 'theta', r'\\alpha': 'alpha', r'\\beta': 'beta',
        r'\\gamma': 'gamma', r'\\delta': 'delta', r'\\lambda': 'lambda',
        r'\\mu': 'mu', r'\\sigma': 'sigma', r'\\omega': 'omega',
    }
    for pat, word in greek.items():
        text = re.sub(pat, word, text)
    # --- Operators ---
    text = re.sub(r'\\times', ' times ', text)
    text = re.sub(r'\\div', ' divided by ', text)
    text = re.sub(r'\\pm', ' plus or minus ', text)
    text = re.sub(r'\\geq', ' greater than or equal to ', text)
    text = re.sub(r'\\leq', ' less than or equal to ', text)
    text = re.sub(r'\\neq', ' not equal to ', text)
    text = re.sub(r'\\approx', ' approximately equals ', text)
    text = re.sub(r'\\infty', ' infinity ', text)
    text = re.sub(r'\\sum', ' sum ', text)
    text = re.sub(r'\\int', ' integral ', text)
    # --- Inline math: $...$ -> just the inner content (already converted above) ---
    text = re.sub(r'\$\$([^$]+)\$\$', r'\1', text)
    text = re.sub(r'\$([^$\n]+)\$', r'\1', text)
    # --- Remove remaining LaTeX braces ---
    text = re.sub(r'[{}]', '', text)
    # --- Remove remaining backslash commands ---
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    return text


def sanitize_for_tts(text: str) -> str:
    """Convert math to spoken form and strip markdown so TTS pronounces everything correctly."""
    # Convert math/LaTeX to spoken English FIRST (before stripping)
    text = _math_to_speech(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove bold/italic markers: **, __, *, _
    text = re.sub(r'\*{1,3}|_{1,3}', '', text)
    # Remove markdown headers (### Title -> Title)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Remove inline code and code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]*`', '', text)
    # Remove markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove bullet/numbered list markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Collapse excessive whitespace/newlines
    text = re.sub(r'\n{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class WAECExamAgent(Agent):
    def __init__(self, user_id: str = None, student_name: str = None, subject: str = None, chat_ctx: ChatContext = None, instructions_text: str = None) -> None:
        # ALWAYS use instructions from file, replace {student_name} placeholder
        if instructions_text:
            base_instructions = instructions_text.replace("{student_name}", student_name or "Student")
        else:
            # Fallback if no file loaded
            base_instructions = f"""You are an exam tutor for WAEC and JAMB exams. The student's name is {student_name}.

MEMORY:  Track student progress, weak areas, scores, strengths across subjects. Reference previous sessions and answer questions about their learning history.
RULES:

OPENING - Start immediately with varied phrases (rotate, never repeat same one twice):

IF CORRECT (include {student_name} in some):
"Nailed it, {student_name}!", "Brilliant!", "Spot on!", "You're crushing it, {student_name}!", "Perfect!", "You're on fire!", "Outstanding, {student_name}!", "Boom! Correct!", "Superb!", "Yes! That's it!", "Right on, {student_name}!", "You smashed that!", "Flawless!", "Excellent, {student_name}!", "Sharp thinking!", "Impressive!", "Killing it, {student_name}!", "You've got this!", "Couldn't be better, {student_name}!"

IF WRONG:
"Not quite right, but great effort! {student_name}", "So close!", "Almost had it!", "Keep pushing, {student_name}! Let me help.", "Good thinking, just needs a small fix.", "That's a common mistake!", "No worries, let's fix this.", "Close call! Let's work through it.", "Hey, mistakes help us learn!", "You're improving!", "This one's tricky!", "Let's break it down.", "Not quite right, but good effort! {student_name}"

IMPORTANT: Do NOT always use "Brilliant!" or "Not quite right!" - use the varied phrases from section 1 above.

MCQs (All Subjects):

IF CORRECT:  mention their name, give 1-2 sentence explanation why it's right. End with varied check-in.
Example correct: "Spot on, {student_name}! You correctly identified that mitosis produces two identical cells. Any questions?"

IF WRONG: mention their name, explain the mistake briefly. State correct answer and reason. End with varied check-in.

Example WRONG: "Keep pushing, {student_name}! You correctly identified that mitosis produces two identical cells. Any questions?"

Keep it 2-3 sentences max
Only mark correct if answer EXACTLY matches
For Math/Physics/Chemistry: mention key step if wrong

Theory/Short Answers:

Point out what's correct and what's missing
Provide complete answer if significantly wrong
For calculations: check formula, substitution, units
End with: "Want me to break that down?"

Essays/ Theory (English, Literature, Government, History, etc.):
Grade on: Content, Organization, Expression, Mechanical Accuracy
Performance levels: Poor, Needs Improvement, Fair, Good, Very Good, Excellent

Response fromat :
Start with appreciation  and mention name
Point out what's correct and what's missing
Provide complete answer if significantly wrong
For calculations: check formula, substitution, units
End with varied check-in like: "Want me to break that down?", "Need more details?", "Should I explain further?"

""Thanks for your essay, {student_name}! Overall, it's [level]. You did well in [1-2 strengths], but could improve [1-2 areas briefly]. Want tips on improving it?"
Comprehension:
Scored out of 5/10.
"Great effort! That was [level]. You got [what's correct]... to improve, try to [1-2 brief tips]. Want me to explain more?"


#Subject-Specific Guidelines:

-Mathematics/Physics/Chemistry:
Check for: correct formula, proper substitution, correct units, logical steps
"Good start! Your formula is correct, but watch the unit conversion at the end."

-Biology/Agricultural Science:
Check for: key terms, accurate descriptions, proper labeling

-Economics/Commerce/Accounting:
Check for: relevant concepts, examples, proper terminology

ALWAYS END with friendly check-in (vary these):
"Any questions?", "Does that help?", "Want to go over that again?", "Need more explanation?", "Ready to move on?", "Does that clear it up?"
IF NO QUESTIONS, respond warmly (vary these):
"Glad that helped!", "Awesome! Keep it up!", "Happy to help!", "Perfect! You're doing great!", "Excellent! On to the next!", "Great! Let's keep going!"
Be warm, supportive, and conversational. Answer other learning questions too. Use conversation history for personalized support. Make connections across subjects when relevant.

# General Guidelines:
- Use conversation history for continuity and personalized support
- Be culturally aware—use Nigerian context and examples where appropriate
- Make connections across subjects when relevant (e.g., "This is like the concept we covered in Physics!")
-  Reference specific options (A, B, etc.) only if the student asks for clarification.
- You may also answer other learning-related questions outside of exam practice.

REMEMBER: You're building confidence, celebrating growth, and preparing {student_name} for exam success

VOICE FORMAT (CRITICAL): This is a voice agent. NEVER use markdown, LaTeX, bullet points, bold (**), headers (###), or math symbols. Speak in plain conversational English only.
UNITS & FORMULAS (CRITICAL): NEVER write abbreviations or symbolic formulas. Always spell out fully: say "metres per second" not "m/s", say "kilometres per hour" not "km/h", say "Newton" not "N", say "H 2 O" not "H₂O", say "the formula is velocity equals distance divided by time" not "v = d/t".
"""
        
        # Pass the chat_ctx to the parent Agent class
        super().__init__(instructions=base_instructions, chat_ctx=chat_ctx or ChatContext.empty())
        self._session = None
        self.user_id = user_id or "default_student"
        self.student_name = student_name or "Student"
        self.subject = subject or "general"
        logger.info(f"Initialized WAEC agent for user: {self.user_id}, name: {self.student_name}, subject: {self.subject}")

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, value):
        self._session = value

    async def tts_node(self, input, model_settings):
        """Override tts_node to strip markdown/HTML before synthesis."""
        async def clean_input():
            async for chunk in input:
                leading = ' ' if chunk and chunk[0].isspace() else ''
                trailing = ' ' if chunk and chunk[-1].isspace() else ''
                cleaned = sanitize_for_tts(chunk)
                yield leading + cleaned + trailing

        async for frame in super().tts_node(clean_input(), model_settings):
            yield frame



    async def process_vision_message(self, message_data: dict):
        """Process incoming message with potential image and add to chat context"""
        try:
            logger.info(f"📨 Processing vision message: {message_data.get('question_type', 'Unknown')}")
            
            # Extract question details
            question_type = message_data.get('question_type', 'MCQ')
            question_text = message_data.get('question', '')
            user_answer = message_data.get('user_answer', '')
            correct_answer = message_data.get('correct_answer', '')
            image_url = message_data.get('image')

            # Detect blank / non-substantive student answers
            user_answer_stripped = (user_answer or '').strip()
            user_answer_lowered = user_answer_stripped.lower()
            is_blank = (
                not user_answer_stripped or
                user_answer_lowered in (
                    "i don't know", "i dont know",
                    "i don't understand", "i dont understand",
                    "idk", "no idea"
                )
            )
            user_answer_display = '[BLANK — student provided no answer]' if is_blank else user_answer

            # Build the text content
            if question_type == 'Comprehension':
                passage = message_data.get('passage', '')
                content_text = (
                    f"Question Type: {question_type}\n\n"
                    f"Passage: {passage}\n\n"
                    f"Question: {question_text}\n\n"
                    f"Student's Answer: {user_answer_display}\n"
                    f"Reference Answer (examiner's model answer — for your reference ONLY, do NOT treat this as the student's answer): {correct_answer}."
                )

            elif question_type == 'Essay':
                model_answer = message_data.get('model_answer', '')
                score = message_data.get('score', 0)
                feedback = message_data.get('feedback', '')
                content_text = (
                    f"Question Type: {question_type}\n\n"
                    f"Question: {question_text}\n\n"
                    f"Student's Answer: {user_answer_display}\n"
                    f"Reference Answer (examiner's model answer — for your reference ONLY, do NOT treat this as the student's answer): {model_answer}\n"
                    f"Score: {score}"
                )

            else:  # MCQ or other types
                options = message_data.get('options', {})
                options_text = ""
                if options:
                    if isinstance(options, dict):
                        options_text = "\\n".join([f"{key}: {value}" for key, value in options.items()])
                    elif isinstance(options, list):
                        options_text = "\\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])

                formatted_options = f"Options:\n{options_text}\n" if options_text else ""
                is_correct = (not is_blank) and (str(user_answer).strip().upper() == str(correct_answer).strip().upper())
                content_text = (
                    f"Question Type: {question_type}\n\n"
                    f"Question: {question_text}\n\n"
                    f"{formatted_options}"
                    f"Student's Answer: {user_answer_display}\n"
                    f"Is Correct: {'YES' if is_correct else 'NO'}\n"
                    f"Reference Answer (examiner's model answer — for your reference ONLY, do NOT treat this as the student's answer): {correct_answer}"
                )

            if is_blank:
                content_text += (
                    "\n\nNOTE: The student has not provided an answer (shown as [BLANK] above). "
                    "Do NOT treat the Reference Answer as the student's response. "
                    "Acknowledge that the student has not answered yet and guide them with a helpful hint or question."
                )
            
            # Create a copy of the context to modify it (Read-Only fix)
            new_ctx = self.chat_ctx.copy()

            # If there's an image, add it to context using the URL directly
            if image_url:
                logger.info(f"🖼️ Question has image URL: {image_url}")
                # Using the URL directly is more reliable and cleaner for the LLM
                chat_image = ImageContent(image=image_url)
                
                new_ctx.add_message(
                    role="user",
                    content=[content_text, chat_image]
                )
                logger.info(f"✅ Added message with image URL to chat context")
            else:
                logger.info("📝 No image in question, adding text only")
                new_ctx.add_message(role="user", content=content_text)
            
            # Update the agent's context using the documented method
            await self.update_chat_ctx(new_ctx)
            
            # Trigger assistant response
            logger.info("✅ Vision message processed. Triggering response...")
            if self.session:
                 try:
                     # LiveKit Agents 0.8+ (New API)
                     if hasattr(self.session, 'response') and hasattr(self.session.response, 'create'):
                         logger.info("Triggering response via session.response.create()")
                         await self.session.response.create()
                     # Legacy API
                     elif hasattr(self.session, 'generate_reply'):
                         logger.info("Triggering response via session.generate_reply()")
                         await self.session.generate_reply()
                     else:
                         logger.warning("Could not trigger LLM response - unknown session method")
                 except Exception as e:
                     logger.error(f"Error triggering response: {e}", exc_info=True) 
        
        except Exception as e:
            logger.error(f"❌ Error processing vision message: {str(e)}", exc_info=True)
            
    async def summarize_conversation(self, chat_ctx: ChatContext) -> str:
        """Generate a summary of the conversation using OpenAI"""
        try:
            # Extract conversation content
            conversation_lines = []
            for item in chat_ctx.items:
                if isinstance(item, ChatMessage):
                    if item.role == "user":
                        content = item.text_content or ''
                        if content.strip():
                            conversation_lines.append(f"Student: {content}")
                    elif item.role == "assistant":
                        content = item.text_content or ''
                        if content.strip():
                            conversation_lines.append(f"Tutor: {content}")
            
            if not conversation_lines:
                return ""
            
            conversation_text = "\n".join(conversation_lines)
            
            # Create summary using OpenAI
            response = await openai_async_client.chat.completions.create(
                model="google/gemma-4-E4B-it",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful assistant that summarizes educational conversations between an Exam tutor and students. 

Create a concise summary that captures:
1. Topics discussed and questions asked
2. very important : Student's performance and understanding level (The student's answer)
3. Areas where student struggled or excelled
5. Student's progress and learning patterns

Keep the summary focused on educational progress and learning outcomes. 
important : make it very simple and concise but informative."""
                    },
                    {
                        "role": "user", 
                        "content": f"Please summarize this tutoring conversation:\n\n{conversation_text}"
                    }
                ],
                max_completion_tokens=500
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated conversation summary for user {self.user_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            return f"Conversation summary unavailable due to error: {str(e)}"

    async def load_chat_history(self) -> ChatContext:
        """Load previous conversation summary and convert to context"""
        if conversations_collection is None:
            logger.warning("MongoDB not available - cannot load chat history")
            return ChatContext.empty()
            
        try:
            logger.info(f"Loading conversation summary for student: {self.user_id}")
            
            # Get the conversation document for this user+subject
            conversation = conversations_collection.find_one({"user_id": self.user_id, "subject": self.subject})

            if conversation and "conversation_summary" in conversation:
                logger.info(f"Found previous conversation summary for user {self.user_id}")
                
                # Create ChatContext with the summary as system context
                chat_ctx = ChatContext.empty()
                
                # Add the summary as a system message to provide context
                summary = conversation['conversation_summary']
                if summary.strip():
                    # Add system context about previous learning
                    context_message = f"Previous learning session summary: {summary}\n\nUse this context to provide personalized feedback and track the student's progress."
                    chat_ctx.add_message(role="system", content=context_message)
                
                logger.info(f"Loaded conversation summary for user {self.user_id}")
                return chat_ctx
            else:
                logger.info(f"No previous conversation found for student {self.user_id}")
                return ChatContext.empty()
                
        except Exception as e:
            logger.error(f"Error loading chat history for student {self.user_id}: {str(e)}")
            return ChatContext.empty()

    async def save_chat_history(self, chat_ctx: ChatContext):
        """Generate summary of current conversation and save to MongoDB"""
        if conversations_collection is None:
            logger.warning("MongoDB not available - cannot save chat history")
            return
            
        try:
            # Check if there's meaningful conversation content
            message_count = len([item for item in chat_ctx.items if isinstance(item, ChatMessage) and item.text_content and item.text_content.strip()])
            
            if message_count < 2:  # Need at least some back-and-forth
                logger.info("Not enough conversation content to summarize")
                return
            
            # Generate summary of the current conversation
            current_summary = await self.summarize_conversation(chat_ctx)
            
            if not current_summary.strip():
                logger.warning("Generated summary is empty")
                return
            
            # Get existing summary for this user+subject
            existing_doc = conversations_collection.find_one({"user_id": self.user_id, "subject": self.subject})
            
            if existing_doc and "conversation_summary" in existing_doc and existing_doc["conversation_summary"].strip():
                # Combine existing summary with new summary
                existing_summary = existing_doc["conversation_summary"]
                
                # Create a combined summary using LLM
                combined_summary = await self.combine_summaries(existing_summary, current_summary)
            else:
                # No existing summary, use current summary
                combined_summary = current_summary
            
            # Update the conversation document for this user+subject
            conversations_collection.update_one(
                {"user_id": self.user_id, "subject": self.subject},
                {
                    "$set": {
                        "user_id": self.user_id,
                        "subject": self.subject,
                        "conversation_summary": combined_summary,
                        "updated_at": datetime.utcnow(),
                        "session_count": existing_doc.get("session_count", 0) + 1 if existing_doc else 1
                    },
                    "$setOnInsert": {
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            
            logger.info(f"Saved conversation summary for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error saving chat history for user {self.user_id}: {str(e)}")

    async def combine_summaries(self, existing_summary: str, new_summary: str) -> str:
        """Combine existing and new summaries into a comprehensive one"""
        try:
            response = await openai_async_client.chat.completions.create(
                model="google/gemma-4-E4B-it",
                messages=[
                    {
                        "role": "system",
                        "content": """You are combining two educational conversation summaries for a WAEC exam student. 

Create a comprehensive summary that:
1. Maintains continuity of the student's learning journey 
2. Tracks progress over time (right or wrong answers by the student)
3. Identifies recurring patterns and improvements
4. Keeps important historical context
5. make it simple and very concise but informative

Focus on the student's overall learning progress, strengths, weaknesses, and development patterns."""
                    },
                    {
                        "role": "user",
                        "content": f"""Please combine these two conversation summaries into one comprehensive summary:

PREVIOUS SESSIONS:
{existing_summary}

LATEST SESSION:
{new_summary}

Create a unified summary that shows the student's learning journey and progress."""
                    }
                ],
                max_completion_tokens=600
            )
            
            combined = response.choices[0].message.content.strip()
            logger.info("Successfully combined conversation summaries")
            return combined
            
        except Exception as e:
            logger.error(f"Error combining summaries: {str(e)}")
            # Fallback: simple concatenation
            return f"{existing_summary}\n\nLatest session: {new_summary}"
  
def prewarm(proc: JobProcess):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
        force=True
    )
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=0.5,   # wait 500ms of silence before cutting off (reduces false ends)
        activation_threshold=0.6,   # higher = less sensitive to background noise (default 0.5)
    )
  
async def load_prompt_from_file(subject: str, topic: str = None) -> str:
    """Load core subject prompt + topic prompt (combined). Core is always loaded first."""
    import os

    subject_normalized = subject.lower().replace(" ", "_").replace("-", "_")
    topic_normalized = topic.lower().replace(" ", "_").replace("-", "_") if topic else None

    prompts_dir = "prompts"
    subject_dir = os.path.join(prompts_dir, subject_normalized)

    logger.info(f"🔍 Looking for prompt - Subject: '{subject}' -> '{subject_normalized}'")
    if topic:
        logger.info(f"🔍 Topic: '{topic}' -> '{topic_normalized}'")

    if not os.path.exists(subject_dir):
        logger.warning(f"Subject prompt directory not found: {subject_dir}")
        return None

    # Always load the core subject prompt first
    core_file = os.path.join(subject_dir, f"core_{subject_normalized}_tutor.txt")
    core_prompt = None
    if os.path.exists(core_file):
        with open(core_file, 'r', encoding='utf-8') as f:
            core_prompt = f.read()
        logger.info(f"✅ Loaded core prompt: {core_file} ({len(core_prompt)} chars)")
    else:
        logger.warning(f"⚠️  Core prompt not found: {core_file}")

    # If topic specified, load topic prompt and combine with core
    if topic_normalized:
        topic_file = os.path.join(subject_dir, f"{topic_normalized}.txt")
        logger.info(f"🔍 Checking topic file: {topic_file}")
        if os.path.exists(topic_file):
            with open(topic_file, 'r', encoding='utf-8') as f:
                topic_prompt = f.read()
            logger.info(f"✅ Loaded topic prompt: {topic_file} ({len(topic_prompt)} chars)")
            if core_prompt:
                combined = f"{core_prompt}\n\n# TOPIC-SPECIFIC INSTRUCTIONS ({topic}):\n{topic_prompt}"
                logger.info(f"✅ COMBINED: core + topic prompt ({len(combined)} chars total)")
                return combined
            else:
                logger.info(f"✅ USING TOPIC ONLY (no core found): {topic_file}")
                return topic_prompt
        else:
            logger.warning(f"  Topic file not found: {topic_file}, using core only")

    if core_prompt:
        logger.info(f"✅ USING CORE PROMPT ONLY: {core_file}")
        return core_prompt

    logger.error(f"❌ FAILED: No prompt file found for subject '{subject}'")
    return None


async def entrypoint(ctx: JobContext):
    # Force all output to /proc/1/fd/1 (Docker stdout) — works even if LiveKit redirects stdout
    try:
        _docker_out = open("/proc/1/fd/1", "w", buffering=1)
        print(f"[ENTRYPOINT] session started room={ctx.room.name}", file=_docker_out, flush=True)
    except Exception:
        _docker_out = sys.stdout

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler(stream=_docker_out)
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger.handlers = [_handler]
    logging.getLogger("waec_exam_agent").setLevel(logging.INFO)
    logging.getLogger("livekit.agents").setLevel(logging.INFO)

    # Set log context fields early
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }  
    
    # IMPORTANT: Get participant info first to know student details
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    user_id = participant.identity  
    student_name = participant.name
    
    logger.info("=" * 80)
    logger.info(f"🎓 VOICE AGENT SESSION STARTED")
    logger.info(f"Participant: {student_name} (ID: {user_id})")
    
    # Get subject and topic from participant metadata (sent by frontend in token)
    subject = None
    topic = None
    subject_id = None
    auth_token = None # Extract auth token for reporting
    prompt_text = None
    prompt_file_used = "default_fallback"
    
    if participant.metadata:
        try:
            metadata = json.loads(participant.metadata)
            subject = metadata.get("subject")
            topic = metadata.get("topic")
            subject_id = metadata.get("subjectId")
            auth_token = metadata.get("authToken")
            
            logger.info(f"📚 Subject: {subject or 'Not specified'}")
            logger.info(f"🆔 Subject ID: {subject_id or 'Not specified'}")
            logger.info(f"📖 Topic: {topic or 'Not specified (will use core)'}")
            
            # Load prompt from file BEFORE creating agent
            if subject:
                prompt_text = await load_prompt_from_file(subject, topic)
                if prompt_text:
                    if topic:
                        prompt_file_used = f"prompts/{subject}/{topic}.txt"
                        logger.info(f"✅ USING TOPIC PROMPT: prompts/{subject}/{topic}.txt")
                    else:
                        prompt_file_used = f"prompts/{subject}/core_{subject}_tutor.txt"
                        logger.info(f"✅ USING CORE PROMPT: prompts/{subject}/core_{subject}_tutor.txt")
                    logger.info(f"📝 Prompt preview (first 200 chars): {prompt_text[:200]}...")
                else:
                    logger.warning(f"⚠️  No prompt file found for {subject}/{topic}, using fallback")
            else:
                logger.warning(f"⚠️  No subject specified, using default prompt")
        except Exception as e:
            logger.error(f"❌ Error parsing participant metadata: {e}")
    else:
        logger.warning("⚠️  No participant metadata found, using default prompt")
    
    logger.info("=" * 80)
      
    temp_agent = WAECExamAgent(user_id=user_id, student_name=student_name, subject=subject)
    chat_ctx = await temp_agent.load_chat_history()

    agent = WAECExamAgent(
        user_id=user_id,
        student_name=student_name,
        subject=subject,
        chat_ctx=chat_ctx,
        instructions_text=prompt_text
    )
    
    logger.info("=" * 80)
    logger.info(f"🤖 AGENT INITIALIZED")
    logger.info(f"Prompt file used: {prompt_file_used}")
    logger.info(f"Instructions loaded: {'YES ✅' if prompt_text else 'NO - Using fallback ⚠️'}")
    if prompt_text:
        logger.info(f"Instructions length: {len(prompt_text)} characters")
    logger.info("=" * 80)  
      

    session = AgentSession(
    vad=ctx.proc.userdata["vad"],
   
    llm=openai.LLM(base_url=os.getenv("PUBLICAAI_BASE_URL"),model="google/gemma-4-E4B-it", api_key=os.getenv("API_KEY")),
    # llm=openai.LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    stt=openai.STT(language="en", model="whisper-1"),
    tts=openai.TTS(
            base_url=os.getenv("PUBLICAAI_TTS_BASE_URL"),
            # model="tts-1",
            voice="voice5",
            response_format="wav",
        ),
    )


    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    @session.on("conversation_item_added")
    def on_conversation_item_added(ev):
        """Log conversation items for debugging."""
        item = ev.item if hasattr(ev, 'item') else None
        if item is None:
            return
        role = getattr(item, 'role', None)
        text = getattr(item, 'text_content', None) or ""
        if text.strip():
            if role == "user":
                logger.info(f"⌨️  User: {text[:80]}")
            elif role == "assistant":
                logger.info(f"🤖 Agent: {text[:80]}...")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev):
        """Log agent state changes."""
        old_state = str(getattr(ev, 'old_state', ''))
        new_state = str(getattr(ev, 'new_state', ''))
        logger.info(f"🔄 Agent state: {old_state} → {new_state}")

    # Start session with standard options
    await session.start(  
        agent=agent,  
        room=ctx.room,
        # Using RoomInputOptions for text/audio input
        room_input_options=RoomInputOptions(
            text_enabled=True,
            audio_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(
            transcription_enabled=True,
            audio_enabled=True
        )
    )  
      
    ctx.log_context_fields["user_id"] = user_id

    # Assign session to agent to allow manual triggering
    agent.session = session

    logger.info(f"Agent initialized with prompt for subject: {subject or 'default'}, topic: {topic or 'core'}")  
    
    # --- Vision Logic Restoration ---
    def handle_vision_message(reader, participant_identity):
        """Handle vision messages sent via text stream"""
        async def _process():
            try:
                # Use read_all() as per correct API
                message_text = await reader.read_all()
                logger.info(f"cam📸 Received vision text message from {participant_identity}")
                
                try:
                    message_data = json.loads(message_text)
                    await agent.process_vision_message(message_data)
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON vision message: {message_text[:100]}...")
            except Exception as e:
                logger.error(f"Error in vision message handler: {str(e)}", exc_info=True)
        
        asyncio.create_task(_process())

    # Register the handler
    room = ctx.room
    room.register_text_stream_handler('lk.chat.vision', handle_vision_message)
    logger.info("✅ Registered text stream handler for 'lk.chat.vision'")
    # --------------------------------
      
    async def log_usage():  
        summary = usage_collector.get_summary()  
        logger.info(f"Usage summary: {summary}")
        
        # Save usage directly to MongoDB (No external API call needed)
        try:
            # Calculate total tokens (LLM + STT + TTS)
            # We combine all metrics into a single "usage unit" for simplicity, 
            # or you might want to store them separately if the DB schema allowed.
            # For now, we sum: LLM Tokens + TTS Characters + STT Seconds
            llm_total = summary.llm_prompt_tokens + summary.llm_completion_tokens
            tts_total = summary.tts_characters_count
            stt_total = int(summary.stt_audio_duration)
            
            total_usage_units = llm_total + tts_total + stt_total
            
            # Prepare payload for external API
            # Note: studentId is omitted as the server uses the auth token for identification
            external_request_id = str(uuid.uuid4())
            payload = {
                "studentId": user_id,
                "tokensUsed": total_usage_units,
                "requestId": external_request_id,
                "token": auth_token,
                "name": student_name,
                "subject": subject,
                "topic": topic
             
            }
            
            # Send usage to external API
            external_api_url = os.getenv("BACKEND_USAGE_URL")
            mask_token = f"{auth_token[:10]}...{auth_token[-10:]}" if auth_token else "NONE"
            logger.info(f"📤 Reporting usage to external API: {external_api_url}")
            logger.info(f"📦 Payload: studentId={user_id}, tokens={total_usage_units}, token={mask_token}")
            
            # Set up headers with authorization
            headers = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
            else:
                logger.warning("⚠️ No auth token available for external API reporting")
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        external_api_url, 
                        json=payload, 
                        headers=headers,
                        timeout=10
                    ) as response:
                        if response.status in [200, 201]:
                            logger.info(f"✅ Successfully reported usage to external API. Status: {response.status}")
                        else:
                            resp_text = await response.text()
                            logger.warning(f"⚠️ External API returned error status {response.status}: {resp_text}")
                            logger.info(f"💡 Hint: If 403, check if the token/subjectId match an active allocation on the server.")
                except Exception as api_err:
                    logger.error(f"❌ Failed to reach external API: {api_err}")

            # Keep existing MongoDB storage as backup/local log
            store_token_usage(
                student_id=user_id,
                tokens=total_usage_units,
                request_id=external_request_id,
                model_name="voice-agent-combined",
                subject_id=subject_id,
                session_type="voice_tutoring"
            )
            
        except Exception as e:
            logger.error(f"❌ Error in token usage reporting: {e}", exc_info=True)
  
  
    async def save_conversation_on_shutdown():
        try:
            logger.info("=" * 60)
            logger.info(f"💬 SAVING CONVERSATION for user: {user_id}")
            logger.info(f"   Student    : {student_name}")
            logger.info(f"   Subject    : {subject or 'N/A'} | Topic: {topic or 'N/A'}")
            await agent.save_chat_history(agent.chat_ctx)
            logger.info(f"✅ Conversation summary saved successfully for user: {user_id}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"❌ Error saving conversation: {str(e)}")  
  
    async def delete_room():  
        try:  
            async with api.LiveKitAPI(  
                os.getenv("LIVEKIT_URL"),  
                os.getenv("LIVEKIT_API_KEY"),  
                os.getenv("LIVEKIT_API_SECRET")  
            ) as api_client:  
                await api_client.room.delete_room(api.DeleteRoomRequest(  
                    room=ctx.room.name,  
                ))  
                logger.info("Room deleted successfully")  
        except Exception as e:  
            logger.error(f"Error deleting room: {str(e)}")  
  
    ctx.add_shutdown_callback(save_conversation_on_shutdown)
    ctx.add_shutdown_callback(log_usage)
    ctx.add_shutdown_callback(delete_room)  
       
    def on_data_received(data: rtc.DataPacket):  
        async def _process_data():  
            try:  
                if data.topic == "lk-rpc-request":  
                    payload = json.loads(data.data.decode('utf-8'))  
                    if payload.get("method") == "start_turn":  
                        logger.info("Interrupting agent via data channel")  
                        session.interrupt()  
  
                        await ctx.room.local_participant.publish_data(  
                            json.dumps({"status": "interrupted"}).encode(),  
                            topic="agent_control"  
                        )  
            except Exception as e:  
                logger.error(f"Data handling error: {str(e)}", exc_info=True)  
  
        asyncio.create_task(_process_data())  
  
    ctx.room.on("data_received", on_data_received)  
  
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))