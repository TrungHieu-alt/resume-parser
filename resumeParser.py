import google.generativeai as gemini
import fitz
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import json
import os
import re

# ---------- Setup Gemini ----------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
gemini.configure(api_key=api_key)

# ---------- STEP 1: Extract text ----------
def extract_text_from_pdf(path): 
    doc = fitz.open(path)
    text = ""
    for page in doc: 
        text += page.get_text("text") + "\n"
    return text

def extract_text_from_img(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img)

# ---------- STEP 2: Use LLM to Extract info ----------
def extract_with_gemini(text):
    prompt = f'''
You are a strict resume parser. Extract ONLY information explicitly written in the resume text. 
Do not infer, generate, or add any content that is not present in the resume. 
If a field is missing, leave it empty ("" for strings, [] for lists, 0.0 for numbers).

Return a valid JSON with exactly these keys (no extra keys, no missing keys):

{{
  "name": "",
  "summary": "",
  "education": [
    {{"degree": "", "school": "", "gpa": ""}}
  ],
  "experiences": [
    {{
      "role": "",
      "years": 0.0,
      "highlights": [],
      "tech_stack": []
    }}
  ],
  "projects": [
    {{
      "role": "",
      "tech_stack": [],
      "highlights": []
    }}
  ],
  "skills": [],
  "languages": []
}}

General normalization rules:
- All extracted text must be in English. Translate if needed. Do not leave any non-English words.
- Output must be strictly valid JSON.
- Always include all keys. If missing: "" for strings, [] for lists, 0.0 for numbers.
- "summary": only fill if resume explicitly contains it. Otherwise "".
- "education.degree": only major/field of study (e.g. "Computer Science"). Do not include faculty/department/school names.
- "education.school": only the university/college/school name.
- "skills" and "tech_stack": 
  * unique arrays, no duplicates
  * remove vague terms like "API", "Cloud Services" unless explicitly named in resume
  * normalize names to most common English form with correct casing (JavaScript, Node.js, React.js, PostgreSQL)
- "experiences[].years": numeric float. If not found use 0.0.
- "experiences": only for real work (internships, part-time, full-time jobs). 
- "projects": only for academic, personal, coursework, research. 
- If no work experience: keep "experiences" as array with one empty object. 
- Never put project info in "experiences".
- "highlights": extract exactly as written, translate to English if needed. Do not invent or expand.
- "languages": 
  * normalize to CEFR (English) or keep JLPT/HSK/TOPIK levels
  * if only descriptors like Fluent, Intermediate, Basic → keep as is
  * always format "Language - Level" (e.g. "English - C1")
- Remove redundant or repeated information across all fields.

Resume text:
{text}
'''
    model = gemini.GenerativeModel("gemini-2.5-flash")


    response = model.generate_content(prompt)

    raw_output = ""
    try:
        raw_output = response.candidates[0].content.parts[0].text
    except Exception:
        raw_output = response.text if hasattr(response, "text") else ""

    raw_output = raw_output.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        raw_output = raw_output.replace("json", "", 1).strip()

    try:
        return json.loads(raw_output)
    except Exception as e:
        print("JSON parse error:", e)
        print("Raw output:", raw_output[:300]) 
        return {}

# ---------- STEP 3: Validate ----------
def validate_json(data):
    if not isinstance(data, dict):
        return {}

    schema = {
        "name": "",
        "summary": "",
        "education": [
            {"degree": "", "gpa": ""}
        ],
        "experiences": [
            {"role": "", "years": 0.0, "highlights": [], "tech_stack": []}
        ],
        "projects": [
            {"role": "", "tech_stack": [], "highlights": []}
        ],
        "skills": [],
        "languages": []
    }

    clean_data = {k: data.get(k, v) for k, v in schema.items()}

    # --- Chuẩn hoá education ---
    fixed_edu = []
    for edu in clean_data.get("education", []):
        if isinstance(edu, dict):
            fixed = {"degree": edu.get("degree", ""), "gpa": edu.get("gpa", "")}
            if fixed["gpa"] and not re.match(r"^\d+(\.\d+)?(/\d+(\.\d+)?)?$", str(fixed["gpa"])):
                fixed["gpa"] = ""
            fixed_edu.append(fixed)
    clean_data["education"] = fixed_edu if fixed_edu else schema["education"]

    # --- Chuẩn hoá experiences ---
    fixed_exps = []
    for exp in clean_data.get("experiences", []):
        if isinstance(exp, dict):
            fixed = {
                "role": exp.get("role", ""),
                "years": float(exp.get("years", 0.0)) if str(exp.get("years", "")).replace(".","",1).isdigit() else 0.0,
                "highlights": exp.get("highlights", []) if isinstance(exp.get("highlights", []), list) else [],
                "tech_stack": exp.get("tech_stack", []) if isinstance(exp.get("tech_stack", []), list) else []
            }
            fixed_exps.append(fixed)
    clean_data["experiences"] = fixed_exps if fixed_exps else schema["experiences"]

    # --- Chuẩn hoá projects ---
    fixed_projs = []
    for proj in clean_data.get("projects", []):
        if isinstance(proj, dict):
            fixed = {
                "role": proj.get("role", ""),
                "tech_stack": proj.get("tech_stack", []) if isinstance(proj.get("tech_stack", []), list) else [],
                "highlights": proj.get("highlights", []) if isinstance(proj.get("highlights", []), list) else []
            }
            fixed_projs.append(fixed)
    clean_data["projects"] = fixed_projs if fixed_projs else schema["projects"]

    # --- Chuẩn hoá skills ---
    skills = clean_data.get("skills", [])
    if isinstance(skills, str):
        clean_data["skills"] = [s.strip() for s in skills.split(",") if s.strip()]
    elif isinstance(skills, list):
        clean_data["skills"] = [str(s).strip() for s in skills if str(s).strip()]
    else:
        clean_data["skills"] = []

    # --- Chuẩn hoá languages ---
    languages = clean_data.get("languages", [])
    if not isinstance(languages, list):
        languages = []
    clean_data["languages"] = [str(l).strip() for l in languages if str(l).strip()]

    return clean_data

# ---------- Wrapper ----------
def parse_resume(file_path, is_pdf=True):
    raw_text = extract_text_from_pdf(file_path) if is_pdf else extract_text_from_img(file_path)
    llm_data = extract_with_gemini(raw_text)
    final_data = validate_json(llm_data)
    return final_data

# ---------- Run ----------
if __name__ == "__main__":
    pdf_file = "public/resume.pdf"
    if not os.path.exists(pdf_file):
        print("❌ File not found:", pdf_file)
    else:
        result = parse_resume(pdf_file, is_pdf=True)
        print(json.dumps(result, indent=2, ensure_ascii=False))
