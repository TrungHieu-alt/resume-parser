# mastodon_bot.py
import os
from mastodon import Mastodon, StreamListener
from dotenv import load_dotenv
import re
import time

# Import hàm xử lý chính từ file đã tái cấu trúc
from main_refactored import find_best_candidates

# --- Cấu hình ---
load_dotenv()
MASTODON_CLIENT_KEY = os.getenv("MASTODON_CLIENT_KEY")
MASTODON_CLIENT_SECRET = os.getenv("MASTODON_CLIENT_SECRET")
MASTODON_ACCESS_TOKEN = os.getenv("MASTODON_ACCESS_TOKEN")
# Lấy URL instance từ file .env hoặc dùng giá trị mặc định
MASTODON_API_BASE_URL = os.getenv("MASTODON_API_BASE_URL", "https://mastodonuet.duckdns.org/")
LISTEN_HASHTAG = "tuyendungAI" # Hashtag để bot lắng nghe

# --- Khởi tạo API ---
mastodon = Mastodon(
    client_id=MASTODON_CLIENT_KEY,
    client_secret=MASTODON_CLIENT_SECRET,
    access_token=MASTODON_ACCESS_TOKEN,
    api_base_url=MASTODON_API_BASE_URL
)

# --- Lớp lắng nghe sự kiện từ Mastodon ---
class RecruitmentListener(StreamListener):
    def on_update(self, status):
        # Kiểm tra xem bài đăng có phải là của chính bot không để tránh lặp vô hạn
        try:
            my_username = mastodon.me()['username']
            if status['account']['username'] == my_username:
                return
        except Exception as e:
            print(f"Lỗi khi lấy thông tin bot: {e}")
            return # Bỏ qua nếu không thể xác thực

        # Kiểm tra xem bài đăng có chứa hashtag không
        hashtags = [tag['name'] for tag in status['tags']]
        if LISTEN_HASHTAG.lower() in [h.lower() for h in hashtags]:
            print(f"🔥 Phát hiện bài đăng tuyển dụng từ @{status['account']['acct']}")

            # Trích xuất nội dung JD (loại bỏ HTML tags và hashtag)
            jd_text = re.sub(r'<.*?>', '', status['content']).strip()
            jd_text = jd_text.replace(f"#{LISTEN_HASHTAG}", "").strip()


            # Lấy thông tin người đăng để gửi DM
            poster_account = status['account']

            # Gửi tin nhắn xác nhận
            mastodon.status_post(
                f"@{poster_account['acct']} Đã nhận yêu cầu tuyển dụng. Bắt đầu quá trình phân tích và sàng lọc. Vui lòng chờ kết quả trong DM!",
                in_reply_to_id=status['id']
            )

            try:
                # Gọi hàm xử lý cốt lõi
                print(f"   > Đang xử lý JD: {jd_text[:100]}...")
                final_ranking = find_best_candidates(jd_text)
                
                # ===================================================================
                # PHẦN SỬA ĐỔI CHÍNH - GỬI NHIỀU DM THAY VÌ MỘT
                # ===================================================================
                
                if not final_ranking:
                    # Gửi một DM nếu không tìm thấy ứng viên nào
                    mastodon.status_post(
                        f"@{poster_account['acct']} Rất tiếc, không tìm thấy ứng viên phù hợp nào trong cơ sở dữ liệu.",
                        visibility='direct'
                    )
                else:
                    # 1. Gửi tin nhắn giới thiệu đầu tiên
                    mastodon.status_post(
                        f"@{poster_account['acct']} ✅ Đã xử lý xong! Dưới đây là bảng xếp hạng các ứng viên phù hợp nhất:",
                        visibility='direct'
                    )
                    time.sleep(1) # Chờ 1 giây để đảm bảo thứ tự tin nhắn

                    # 2. Lặp qua từng ứng viên và gửi một DM riêng cho mỗi người
                    for i, result in enumerate(final_ranking):
                        eval_data = result['detailed_evaluation']
                        
                        # Tạo nội dung tin nhắn cho CHỈ MỘT ứng viên
                        # Tin nhắn này sẽ ngắn và không vượt quá giới hạn
                        single_result_message = (
                            f"🏆 HẠNG {i+1}: {result.get('name', 'N/A')}\n"
                            f"Điểm: {eval_data.get('score', 'N/A')}/100\n"
                            f"Đánh giá kinh nghiệm: {eval_data.get('experience_match', 'N/A')}\n\n"
                            f"Lý do: {eval_data.get('rationale', 'N/A')}"
                        )
                        
                        # Gửi DM cho ứng viên này
                        mastodon.status_post(
                            f"@{poster_account['acct']} {single_result_message}",
                            visibility='direct' # Quan trọng: chỉ gửi cho người nhận!
                        )
                        print(f"   > Đã gửi DM cho Hạng {i+1}: {result.get('name')}")
                        time.sleep(1) # Chờ giữa các tin nhắn để tránh bị coi là spam

                print(f"✅ Đã gửi toàn bộ kết quả DM cho @{poster_account['acct']}")

            except Exception as e:
                print(f"❌ Lỗi trong quá trình xử lý: {e}")
                mastodon.status_post(
                    f"@{poster_account['acct']} Rất tiếc, đã có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại sau.",
                    visibility='direct'
                )

# --- Chạy bot ---
if __name__ == "__main__":
    try:
        my_info = mastodon.me()
        print(f"🤖 Bot tuyển dụng '{my_info['display_name']}' (@{my_info['username']}) đang chạy...")
        print(f"   > Lắng nghe hashtag #{LISTEN_HASHTAG} trên instance {MASTODON_API_BASE_URL}")
        # Lắng nghe các bài đăng công khai có chứa hashtag
        mastodon.stream_hashtag(LISTEN_HASHTAG, RecruitmentListener(), reconnect_async=True)
    except Exception as e:
        print(f"❌ Không thể kết nối đến Mastodon. Vui lòng kiểm tra lại thông tin trong file .env")
        print(f"   > Lỗi chi tiết: {e}")