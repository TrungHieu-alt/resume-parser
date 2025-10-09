import os
import json
from dotenv import load_dotenv

# --- Phần 1: Import logic từ resumeParser và LlamaIndex/OpenAI ---
from resumeParser import parse_resume # Giả sử file resumeParser.py ở cùng thư mục
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI

# --- Tải các biến môi trường ---
load_dotenv()

# ==============================================================================
# BƯỚC 1: PARSE TẤT CẢ CV TRONG THƯ MỤC VÀ TẠO DATABASE
# ==============================================================================
def create_cv_database(folder_path: str):
    """
    Quét một thư mục, parse tất cả các file CV và trả về một danh sách các dict.
    """
    print("--- BƯỚC 1: Đang parse các CV từ thư mục... ---")
    database = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            print(f"  > Đang xử lý file: {filename}")
            try:
                # Sử dụng hàm parse_resume từ file resumeParser.py
                parsed_data = parse_resume(file_path)
                
                # Thêm một ID duy nhất cho mỗi CV
                parsed_data['id'] = os.path.splitext(filename)[0]
                database.append(parsed_data)
                print(f"  > Xử lý thành công: {parsed_data.get('name', 'N/A')}")
            except Exception as e:
                print(f"  > Lỗi khi xử lý {filename}: {e}")
    print("--- HOÀN THÀNH BƯỚC 1 ---\n")
    return database

# ==============================================================================
# HÀM HỖ TRỢ: TẠO NỘI DUNG ĐỂ EMBEDDING TỪ JSON ĐÃ PARSE
# ==============================================================================
def create_embedding_content_from_json(cv_data: dict):
    """
    Tạo một chuỗi văn bản giàu thông tin từ CV JSON để có kết quả embedding tốt nhất.
    """
    parts = []
    if cv_data.get("name"):
        parts.append(f"Name: {cv_data['name']}")
    if cv_data.get("summary"):
        parts.append(f"Summary: {cv_data['summary']}")
    
    exp_parts = []
    for exp in cv_data.get("experiences", []):
        exp_str = f"{exp.get('role', '')} at {exp.get('organization', '')}."
        exp_parts.append(exp_str)
    if exp_parts:
        parts.append("Experience: " + " ".join(exp_parts))

    if cv_data.get("skills"):
        parts.append("Skills: " + ", ".join(cv_data.get("skills", [])))
        
    return "\n".join(parts)


# ==============================================================================
# HÀM CHÍNH ĐỂ THỰC HIỆN TOÀN BỘ QUY TRÌNH
# ==============================================================================
def main():
    # --- Dữ liệu JD (giữ nguyên) ---
    job_description = {
        "id": "jd01",
        "content_for_embedding": "Tuyển dụng vị trí Senior Python Backend Developer. Yêu cầu ít nhất 4 năm kinh nghiệm làm việc với Python và các framework như Django hoặc Flask. Có kinh nghiệm sâu về thiết kế API RESTful, làm việc với cơ sở dữ liệu PostgreSQL. Ưu tiên ứng viên có kinh nghiệm với Docker và các dịch vụ đám mây như AWS."
    }

    # --- Thực hiện Bước 1 ---
    CV_FOLDER = "cv_folder"
    if not os.path.exists(CV_FOLDER):
        print(f"Lỗi: Thư mục '{CV_FOLDER}' không tồn tại. Vui lòng tạo và đặt CV vào đó.")
        return
        
    cv_database = create_cv_database(CV_FOLDER)
    if not cv_database:
        print("Không có CV nào được xử lý. Dừng chương trình.")
        return

    # ==========================================================================
    # BƯỚC 2: FILTER & RETRIEVAL (Sử dụng LlamaIndex)
    # ==========================================================================
    print("--- BƯỚC 2: Đang lập chỉ mục và truy xuất ứng viên (Retrieval) ---")
    
    # Chuẩn bị các Document cho LlamaIndex
    documents = []
    for cv in cv_database:
        embedding_content = create_embedding_content_from_json(cv)
        # Lưu toàn bộ JSON vào metadata để sử dụng ở bước sau
        documents.append(Document(text=embedding_content, metadata={"full_cv_json": cv}))

    # Cấu hình model embedding của OpenAI
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Lập chỉ mục
    index = VectorStoreIndex.from_documents(documents)
    
    # Truy xuất top 3 ứng viên phù hợp nhất
    retriever = index.as_retriever(similarity_top_k=3)
    retrieved_nodes = retriever.retrieve(job_description["content_for_embedding"])

    print("--- HOÀN THÀNH BƯỚC 2 ---\n")

    # ==========================================================================
    # BƯỚC 3: LLM RE-RANKER / MATCHING AGENT (Nâng cấp)
    # ==========================================================================
    print("--- BƯỚC 3: Đang đánh giá chi tiết bằng LLM (GPT-4o) ---")
    
    # Prompt (giữ nguyên từ demo.ipynb)
    SYSTEM_PROMPT = """
    Bạn là một chuyên gia tuyển dụng kỹ thuật (Tech Recruiter) rất kinh nghiệm và tỉ mỉ.
    Nhiệm vụ của bạn là đánh giá một CV của ứng viên dựa trên một Bản mô tả công việc (JD) được cung cấp.
    Hãy phân tích sâu và trả về kết quả đánh giá DUY NHẤT dưới dạng một đối tượng JSON.

    Đối tượng JSON phải có các trường sau:
    - "score": một con số từ 0 đến 100, thể hiện mức độ phù hợp tổng thể. 100 là hoàn toàn khớp.
    - "skills_checklist": một đối tượng chứa hai danh sách:
    - "matched_skills": danh sách các kỹ năng trong JD mà ứng viên có.
    - "missing_skills": danh sách các kỹ năng quan trọng trong JD mà ứng viên không đề cập.
    - "experience_match": một chuỗi ngắn để đánh giá kinh nghiệm (ví dụ: "Rất phù hợp", "Phù hợp một phần", "Không đủ kinh nghiệm").
    - "risk_points": một danh sách các điểm rủi ro hoặc không phù hợp cần lưu ý (ví dụ: "Chuyên môn chính là frontend, không phải backend").
    - "rationale": một đoạn văn ngắn (2-3 câu) giải thích lý do cho điểm số và nhận định của bạn.
    """
    USER_PROMPT_TEMPLATE = """
    Dưới đây là Bản mô tả công việc (JD) và CV của ứng viên. Vui lòng đánh giá.

    --- JD ---
    {jd_text}

    --- CV ---
    {cv_text}
    """

    client = OpenAI()
    evaluation_results = []

    for node in retrieved_nodes:
        full_cv_data = node.metadata["full_cv_json"]
        cv_name = full_cv_data.get('name', 'N/A')
        
        # Để LLM có thông tin tốt nhất, chúng ta cung cấp phiên bản JSON đẹp
        cv_text_for_llm = json.dumps(full_cv_data, indent=2, ensure_ascii=False)
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            jd_text=job_description['content_for_embedding'], 
            cv_text=cv_text_for_llm
        )

        print(f"  > Đang đánh giá ứng viên: {cv_name}...")
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Sử dụng model mạnh nhất
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        result_json = json.loads(response.choices[0].message.content)
        
        evaluation_results.append({
            "name": cv_name,
            "initial_score": node.get_score(),
            "detailed_evaluation": result_json
        })
        print(f"  > Đánh giá hoàn tất!")
    print("--- HOÀN THÀNH BƯỚC 3 ---\n")

    # ==========================================================================
    # BƯỚC 4: AGGREGATE & PRESENT (Hoàn thiện với Re-ranking)
    # ==========================================================================
    sorted_results = sorted(evaluation_results, key=lambda x: x['detailed_evaluation'].get('score', 0), reverse=True)



    print("PDF đã được xử lý: ")
    
    print("\n\n--- BẢNG XẾP HẠNG ỨNG VIÊN CUỐI CÙNG ---")
    
    for i, result in enumerate(sorted_results):
        eval_data = result['detailed_evaluation']
        print("\n==============================================")
        print(f"HẠNG {i+1}: {result['name']}")
        print("==============================================")
        print(f"Điểm phù hợp (LLM Score): {eval_data.get('score', 'N/A')} / 100")
        print(f"Lý do (Rationale): {eval_data.get('rationale', 'N/A')}")
        print("----------------------------------------------")
        print(f"Đánh giá kinh nghiệm: {eval_data.get('experience_match', 'N/A')}")
        print(f"Điểm rủi ro: {', '.join(eval_data.get('risk_points', [])) or 'Không có'}")
        print("==============================================\n")

if __name__ == "__main__":
    main()