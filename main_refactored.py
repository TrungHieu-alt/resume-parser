# main_refactored.py

import os
import json
from dotenv import load_dotenv

# --- Phần 1: Import logic từ các thư viện cần thiết ---

# Import hàm parse CV từ file resumeParser.py
# Đảm bảo file resumeParser.py nằm cùng thư mục với file này.
from resumeParser import parse_resume

# Import các thành phần cần thiết từ LlamaIndex và OpenAI
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI

# --- Tải các biến môi trường từ file .env ---
# Đảm bảo file .env của bạn có OPENAI_API_KEY
load_dotenv()

# ==============================================================================
# BƯỚC 1: PARSE TẤT CẢ CV TRONG THƯ MỤC VÀ TẠO DATABASE
# ==============================================================================
def create_cv_database(folder_path: str):
    """
    Quét một thư mục, parse tất cả các file CV (PDF, ảnh) bằng resumeParser
    và trả về một danh sách các đối tượng JSON chứa thông tin CV.
    """
    print("--- BƯỚC 1: Đang parse các CV từ thư mục... ---")
    database = []
    # Kiểm tra xem thư mục có tồn tại không
    if not os.path.exists(folder_path):
        print(f"Lỗi: Thư mục '{folder_path}' không tồn tại.")
        return []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Chỉ xử lý file, bỏ qua thư mục con
        if os.path.isfile(file_path):
            print(f"  > Đang xử lý file: {filename}")
            try:
                # Sử dụng hàm parse_resume đã import
                parsed_data = parse_resume(file_path)

                # Thêm một ID duy nhất cho mỗi CV, lấy từ tên file
                parsed_data['id'] = os.path.splitext(filename)[0]
                database.append(parsed_data)
                print(f"  > Xử lý thành công CV của: {parsed_data.get('name', 'N/A')}")
            except Exception as e:
                print(f"  > Lỗi khi xử lý file {filename}: {e}")
    print(f"--- HOÀN THÀNH BƯỚC 1: Đã parse được {len(database)} CV ---\n")
    return database

# ==============================================================================
# HÀM HỖ TRỢ: TẠO NỘI DUNG VĂN BẢN ĐỂ EMBEDDING TỪ JSON
# ==============================================================================
def create_embedding_content_from_json(cv_data: dict) -> str:
    """
    Tạo một chuỗi văn bản giàu thông tin từ CV JSON để có kết quả embedding tốt nhất.
    Hàm này tổng hợp các thông tin quan trọng nhất vào một chuỗi duy nhất.
    """
    parts = []
    if cv_data.get("name"):
        parts.append(f"Name: {cv_data['name']}")
    if cv_data.get("summary"):
        parts.append(f"Summary: {cv_data['summary']}")

    # Tổng hợp kinh nghiệm làm việc
    exp_parts = []
    for exp in cv_data.get("experiences", []):
        role = exp.get('role', '')
        org = exp.get('organization', '')
        if role and org:
            exp_parts.append(f"{role} at {org}")
    if exp_parts:
        parts.append("Experience: " + ", ".join(exp_parts))

    # Nối các kỹ năng
    if cv_data.get("skills"):
        parts.append("Skills: " + ", ".join(cv_data.get("skills", [])))

    return "\n".join(parts)

# ==============================================================================
# HÀM CHÍNH ĐỂ THỰC HIỆN TOÀN BỘ QUY TRÌNH RAG
# ==============================================================================
def find_best_candidates(job_description_text: str):
    """
    Hàm này nhận đầu vào là một chuỗi văn bản mô tả công việc (JD),
    thực hiện toàn bộ quy trình RAG (Retrieval-Augmented Generation)
    và trả về một danh sách các ứng viên đã được đánh giá và xếp hạng.
    """
    # --- Bước 1: Tạo hoặc tải cơ sở dữ liệu CV ---
    CV_FOLDER = "cv_folder"
    cv_database = create_cv_database(CV_FOLDER)
    if not cv_database:
        print("Không có CV nào trong cơ sở dữ liệu để xử lý. Dừng lại.")
        return []

    # ==========================================================================
    # BƯỚC 2: FILTER & RETRIEVAL (Sử dụng LlamaIndex)
    # ==========================================================================
    print("--- BƯỚC 2: Đang lập chỉ mục và truy xuất ứng viên (Retrieval) ---")

    # Chuẩn bị các đối tượng Document cho LlamaIndex
    documents = []
    for cv in cv_database:
        embedding_content = create_embedding_content_from_json(cv)
        # Lưu toàn bộ JSON của CV vào metadata để có thể truy xuất lại ở bước sau
        documents.append(Document(text=embedding_content, metadata={"full_cv_json": cv}))

    # Cấu hình model embedding của OpenAI để tạo vector
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Xây dựng chỉ mục vector từ các document
    index = VectorStoreIndex.from_documents(documents)

    # Tạo một retriever để tìm kiếm các ứng viên phù hợp nhất (top 3)
    retriever = index.as_retriever(similarity_top_k=3)
    retrieved_nodes = retriever.retrieve(job_description_text)

    print(f"--- HOÀN THÀNH BƯỚC 2: Đã tìm thấy {len(retrieved_nodes)} ứng viên tiềm năng ---\n")

    # ==========================================================================
    # BƯỚC 3: LLM RE-RANKER / MATCHING AGENT (Phân tích sâu)
    # ==========================================================================
    print("--- BƯỚC 3: Đang đánh giá chi tiết từng ứng viên bằng LLM (GPT-4o) ---")

    # Prompt hệ thống để hướng dẫn LLM hoạt động như một nhà tuyển dụng
    SYSTEM_PROMPT = """
    Bạn là một chuyên gia tuyển dụng kỹ thuật (Tech Recruiter) rất kinh nghiệm và tỉ mỉ.
    Nhiệm vụ của bạn là đánh giá một CV của ứng viên dựa trên một Bản mô tả công việc (JD) được cung cấp.
    Hãy phân tích sâu và trả về kết quả đánh giá DUY NHẤT dưới dạng một đối tượng JSON.

    Đối tượng JSON phải có các trường sau:
    - "score": một con số từ 0 đến 100, thể hiện mức độ phù hợp tổng thể.
    - "skills_checklist": một đối tượng chứa hai danh sách: "matched_skills" và "missing_skills".
    - "experience_match": một chuỗi ngắn để đánh giá kinh nghiệm (ví dụ: "Rất phù hợp", "Phù hợp", "Không đủ kinh nghiệm").
    - "risk_points": một danh sách các điểm rủi ro hoặc không phù hợp cần lưu ý.
    - "rationale": một đoạn văn ngắn (2-3 câu) giải thích lý do cho điểm số của bạn.
    """
    # Mẫu prompt cho người dùng, sẽ được điền JD và CV vào
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
        # Lấy lại toàn bộ dữ liệu JSON của CV từ metadata đã lưu ở bước 2
        full_cv_data = node.metadata["full_cv_json"]
        cv_name = full_cv_data.get('name', 'N/A')

        # Chuyển đổi CV dạng JSON thành một chuỗi đẹp mắt để LLM dễ đọc
        cv_text_for_llm = json.dumps(full_cv_data, indent=2, ensure_ascii=False)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            jd_text=job_description_text,
            cv_text=cv_text_for_llm
        )

        print(f"  > Đang đánh giá ứng viên: {cv_name}...")

        try:
            # Gửi yêu cầu đến API của OpenAI để đánh giá
            response = client.chat.completions.create(
                model="gpt-4o",  # Sử dụng model mạnh nhất để có kết quả phân tích tốt
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"} # Yêu cầu trả về định dạng JSON
            )

            # Parse kết quả JSON từ phản hồi của API
            result_json = json.loads(response.choices[0].message.content)

            # Lưu kết quả đánh giá
            evaluation_results.append({
                "name": cv_name,
                "initial_score": node.get_score(), # Điểm tương đồng vector ban đầu
                "detailed_evaluation": result_json # Kết quả đánh giá sâu từ LLM
            })
            print(f"  > Đánh giá hoàn tất cho {cv_name}!")
        except Exception as e:
            print(f"  > Lỗi khi đánh giá ứng viên {cv_name}: {e}")

    print("--- HOÀN THÀNH BƯỚC 3 ---\n")

    # ==========================================================================
    # BƯỚC 4: SẮP XẾP VÀ TRẢ VỀ KẾT QUẢ CUỐI CÙNG
    # ==========================================================================
    # Sắp xếp danh sách ứng viên dựa trên điểm số ("score") từ LLM theo thứ tự giảm dần
    sorted_results = sorted(
        evaluation_results,
        key=lambda x: x.get('detailed_evaluation', {}).get('score', 0),
        reverse=True
    )

    return sorted_results

# ==============================================================================
# KHỐI LỆNH ĐỂ CHẠY TEST ĐỘC LẬP
# ==============================================================================
if __name__ == "__main__":
    # Đây là một ví dụ về JD để bạn có thể chạy file này trực tiếp và kiểm tra
    sample_job_description = """
    Tuyển dụng vị trí Senior Python Backend Developer.
    Yêu cầu:
    - Ít nhất 4 năm kinh nghiệm làm việc chuyên sâu với Python.
    - Kinh nghiệm vững vàng với framework Django hoặc Flask.
    - Hiểu biết sâu về thiết kế và xây dựng API RESTful.
    - Thành thạo làm việc với cơ sở dữ liệu PostgreSQL.
    - Có kinh nghiệm với Docker và các dịch vụ đám mây như AWS là một lợi thế lớn.
    """

    print("--- BẮT ĐẦU CHẠY THỬ NGHIỆM QUY TRÌNH RAG ---")
    final_ranking = find_best_candidates(sample_job_description)

    if final_ranking:
        print("\n\n--- BẢNG XẾP HẠNG ỨNG VIÊN CUỐI CÙNG ---")
        for i, result in enumerate(final_ranking):
            eval_data = result['detailed_evaluation']
            print("\n==============================================")
            print(f"HẠNG {i+1}: {result['name']}")
            print("==============================================")
            print(f"Điểm phù hợp (LLM Score): {eval_data.get('score', 'N/A')} / 100")
            print(f"Lý do (Rationale): {eval_data.get('rationale', 'N/A')}")
            print("----------------------------------------------")
            print(f"Đánh giá kinh nghiệm: {eval_data.get('experience_match', 'N/A')}")
            matched = eval_data.get('skills_checklist', {}).get('matched_skills', [])
            missing = eval_data.get('skills_checklist', {}).get('missing_skills', [])
            print(f"Kỹ năng khớp: {', '.join(matched) or 'Không có'}")
            print(f"Kỹ năng thiếu: {', '.join(missing) or 'Không có'}")
            print(f"Điểm rủi ro: {', '.join(eval_data.get('risk_points', [])) or 'Không có'}")
            print("==============================================\n")
    else:
        print("\n--- Không có kết quả nào để hiển thị. ---")