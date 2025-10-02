import google.generativeai as gemini
import fitz
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import json
import os
import re
from datetime import datetime

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
You are a STRICT resume parser.
Your job is to extract ONLY information that is explicitly written in the resume text. 
Do NOT infer, guess, or add any information that is not present.

Return a valid JSON with EXACTLY these keys (no extra keys, no missing keys):

{{
  "name": "",
  "summary": "",
  "education": [
    {{"degree": "", "school": "", "gpa": "", "year": ""}}
  ],
  "experiences": [
    {{
      "role": "",
      "organization": "",
      "start_date": "",
      "end_date": "",
      "years": 0.0,
      "location": "",
      "highlights": []
    }}
  ],
  "projects": [
    {{
      "role": "",
      "highlights": []
    }}
  ],
  "skills": [],
  "languages": [],
  "certifications": [
    {{"name": "", "issuer": "", "year": ""}}
  ],
  "awards": [
    {{"title": "", "issuer": "", "year": ""}}
  ],
  "activities": [
    {{
      "role": "",
      "organization": "",
      "start_date": "",
      "end_date": "",
      "years": 0.0,
      "highlights": []
    }}
  ],
  "publications": [
    {{"title": "", "journal": "", "year": "", "doi": ""}}
  ],
  "licenses": [
    {{"name": "", "issuer": "", "year": ""}}
  ]
}}

Strict rules you MUST follow:
1. Output must be strictly valid JSON. No extra commentary, no markdown code fences.
2. Always include ALL keys, even if empty.
   - "" for strings
   - [] for arrays
   - 0.0 for numbers
3. "summary": only fill if explicitly written. Otherwise "".
4. "education.degree": only major/field of study (e.g. "Computer Science").
5. "education.school": only the university/college/school name.
6. "education.year": extract graduation year or study period if available. Otherwise "".
7. "education.gpa": must match pattern like "3.5/4.0". If invalid, leave "".
8. "experiences": only for real work (internship, part-time, full-time).
   - "role": job title
   - "organization": company/organization
   - "start_date": extract as YYYY-MM or YYYY if available
   - "end_date": extract as YYYY-MM, YYYY or "Present"
   - "years": numeric float (will be validated again in code)
   - "location": location if available
   - "highlights": bullet points or short action statements only
9. "projects": only academic/personal/research projects. Never mix with experiences.
10. "skills" and "languages":
    - Arrays, unique, no duplicates
    - Normalize casing (JavaScript, Node.js, PostgreSQL…)
11. "languages": format "English - C1" if level available, else keep descriptor.
12. "certifications", "awards", "activities", "publications", "licenses": extract if present.
13. Remove duplicates across fields.
14. All extracted text must be in English.

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


# ---------- STEP 3a: Filter highlights ----------
def filter_highlights(raw_list):
    filtered = []
    action_verbs = [
        "developed", "designed", "implemented", "led", "analyzed",
        "created", "managed", "optimized", "conducted", "built",
        "researched", "improved", "organized", "presented", "trained"
    ]
    for item in raw_list:
        text = str(item).strip("•- \n\t").strip()
        if not text:
            continue
        if len(text.split()) < 3:
            continue
        if text.split()[0].lower() in action_verbs:
            filtered.append(text)
        else:
            filtered.append(text)
    return list(dict.fromkeys(filtered))


# ---------- STEP 3b: Refine highlights ----------
def refine_highlights(highlights, max_words=18):
    refined = []
    for hl in highlights:
        text = str(hl).strip("•- \n\t").strip()
        if not text:
            continue
        words = text.split()
        if len(words) > max_words:
            parts = re.split(r',| and ', text)
            parts = [p.strip() for p in parts if len(p.strip().split()) >= 3]
            for p in parts:
                refined.append(p)
        else:
            refined.append(text)
    normalized = []
    for item in refined:
        item = item.strip()
        if not item:
            continue
        item = item[0].upper() + item[1:] if item else item
        if not item.endswith('.'):
            item += '.'
        normalized.append(item)
    unique = list(dict.fromkeys(normalized))
    return unique


# ---------- STEP 3c: Compute years ----------
def compute_years(start_date: str, end_date: str) -> float:
    if not start_date:
        return 0.0

    def parse_date(d):
        if not d: return None
        d = d.strip().lower()
        if d in ["present", "hiện tại", "current", "now"]:
            return datetime.today()
        for fmt in ("%Y-%m", "%Y"):
            try:
                return datetime.strptime(d, fmt)
            except:
                continue
        return None

    start = parse_date(start_date)
    end = parse_date(end_date) or datetime.today()
    if not start: return 0.0

    diff_years = (end.year - start.year) + (end.month - start.month) / 12
    return round(diff_years, 2)


# ---------- STEP 4: Validate JSON ----------
def validate_json(data):
    if not isinstance(data, dict):
        return {}

    schema = {
        "name": "",
        "summary": "",
        "education": [{"degree": "", "school": "", "gpa": "", "year": ""}],
        "experiences": [{"role": "", "organization": "", "years": 0.0, "location": "", "highlights": []}],
        "projects": [{"role": "", "highlights": []}],
        "skills": [],
        "languages": [],
        "certifications": [{"name": "", "issuer": "", "year": ""}],
        "awards": [{"title": "", "issuer": "", "year": ""}],
        "activities": [{"role": "", "organization": "", "years": 0.0, "highlights": []}],
        "publications": [{"title": "", "journal": "", "year": "", "doi": ""}],
        "licenses": [{"name": "", "issuer": "", "year": ""}]
    }

    clean_data = {k: data.get(k, v) for k, v in schema.items()}

    # --- normalize education ---
    fixed_edu = []
    for edu in clean_data.get("education", []):
        if isinstance(edu, dict):
            fixed = {
                "degree": edu.get("degree", ""),
                "school": edu.get("school", ""),
                "gpa": edu.get("gpa", ""),
                "year": edu.get("year", "")
            }
            if fixed["gpa"] and not re.match(r"^\d+(\.\d+)?(/\d+(\.\d+)?)?$", str(fixed["gpa"])):
                fixed["gpa"] = ""
            fixed_edu.append(fixed)
    clean_data["education"] = fixed_edu if fixed_edu else schema["education"]

    # --- normalize experiences ---
    fixed_exps = []
    for exp in data.get("experiences", []):
        if isinstance(exp, dict):
            years_val = compute_years(exp.get("start_date", ""), exp.get("end_date", ""))
            fixed = {
                "role": exp.get("role", ""),
                "organization": exp.get("organization", ""),
                "years": years_val,
                "location": exp.get("location", ""),
                "highlights": refine_highlights(filter_highlights(exp.get("highlights", [])))
            }
            if fixed["role"] or fixed["organization"] or fixed["highlights"]:
                fixed_exps.append(fixed)
    clean_data["experiences"] = fixed_exps if fixed_exps else []

    # --- normalize projects ---
    fixed_projs = []
    for proj in data.get("projects", []):
        if isinstance(proj, dict):
            fixed = {
                "role": proj.get("role", ""),
                "highlights": refine_highlights(filter_highlights(proj.get("highlights", [])))
            }
            if fixed["role"] or fixed["highlights"]:
                fixed_projs.append(fixed)
    clean_data["projects"] = fixed_projs if fixed_projs else []

    # --- normalize activities ---
    fixed_acts = []
    for act in data.get("activities", []):
        if isinstance(act, dict):
            years_val = compute_years(act.get("start_date", ""), act.get("end_date", ""))
            fixed = {
                "role": act.get("role", ""),
                "organization": act.get("organization", ""),
                "years": years_val,
                "highlights": refine_highlights(filter_highlights(act.get("highlights", [])))
            }
            if fixed["role"] or fixed["organization"] or fixed["highlights"]:
                fixed_acts.append(fixed)
    clean_data["activities"] = fixed_acts if fixed_acts else []

    # --- normalize skills & languages ---
    for key in ["skills", "languages"]:
        val = clean_data.get(key, [])
        if isinstance(val, str):
            clean_data[key] = [s.strip() for s in val.split(",") if s.strip()]
        elif isinstance(val, list):
            clean_data[key] = [str(s).strip() for s in val if str(s).strip()]
        else:
            clean_data[key] = []

    return clean_data


# ---------- Wrapper ----------
def parse_resume(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        raw_text = extract_text_from_img(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    llm_data = extract_with_gemini(raw_text)
    final_data = validate_json(llm_data)
    return final_data


# ---------- Run ----------
if __name__ == "__main__":
    file_path = "public/resume.pdf"
    
    if not os.path.exists(file_path):
        print("❌ File not found:", file_path)
    else:
        try:
            result = parse_resume(file_path)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except ValueError as e:
            print("❌", e)
